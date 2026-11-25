#![cfg(feature = "ash-llm")]
//! CPU and opt-in Vulkan proofs for the request-local autoregressive state.

use honkbox_ash_backend::{
    VkError,
    attention::Attention,
    llm::state::{
        ConvolutionPlan, KvPlan, LayerPlan, LayerUpdate, MaskMode, MatrixPlan, RecurrentPlan,
        SequenceState, StatePlan, TokenPosition, TokenUpdate, build_mask,
    },
};
use honkbox_llm_worker::{LlmError, RequestCancel};
use honkbox_residency::Demand;

fn kv_plan(name: &str, query_heads: usize, kv_heads: usize, capacity: usize) -> KvPlan {
    KvPlan {
        name: name.into(),
        query_heads,
        kv_heads,
        head_dim: 2,
        value_dim: 3,
        context_capacity: capacity,
        sliding_window: None,
        element_bytes: 4,
    }
}

fn kv_update(
    segment: u32,
    index: usize,
    head_dim: usize,
    value_dim: usize,
    kv_heads: usize,
) -> TokenUpdate {
    let key = (0..kv_heads * head_dim)
        .map(|i| (index * 9 + i) as f32 * 0.125)
        .collect();
    let value = (0..kv_heads * value_dim)
        .map(|i| (index * 5 + i) as f32 * 0.25)
        .collect();
    TokenUpdate {
        segment,
        layers: vec![LayerUpdate {
            key,
            value,
            ..LayerUpdate::default()
        }],
    }
}

fn recurrent_plan() -> RecurrentPlan {
    RecurrentPlan {
        name: "recurrent".into(),
        convolution: Some(ConvolutionPlan {
            name: "conv".into(),
            channels: 1,
            history: 2,
        }),
        matrix_slots: vec![MatrixPlan {
            name: "delta".into(),
            rows: 2,
            cols: 2,
        }],
        element_bytes: 4,
    }
}

fn recurrent_update(index: usize) -> LayerUpdate {
    LayerUpdate {
        convolution: vec![index as f32, (index + 1) as f32],
        matrices: vec![vec![
            index as f32,
            (index + 1) as f32,
            (index + 2) as f32,
            (index + 3) as f32,
        ]],
        ..LayerUpdate::default()
    }
}

fn two_layer_update(index: usize, kv_heads: usize) -> TokenUpdate {
    let mut update = kv_update(0, index, 2, 3, kv_heads);
    update.layers.push(recurrent_update(index));
    update
}

fn assert_error(result: Result<(), LlmError>, op: &str) {
    if op == "cancelled" {
        assert!(matches!(result, Err(LlmError::Cancelled)));
        return;
    }
    match result {
        Err(LlmError::Context(message)) => assert!(message.contains(op), "{message}"),
        other => panic!("expected named context error for {op}: {other:?}"),
    }
}

fn cpu_logits(state: &SequenceState, layer: &str, query: &[f32]) -> Vec<f32> {
    let plan = state.plan().kv(layer).expect("kv layer");
    let views = state.attention_views(layer).expect("views");
    let positions = state.positions();
    let tokens = positions.len();
    let scale = 1.0 / (plan.head_dim as f32).sqrt();
    let mut output = vec![0.0; tokens * plan.query_heads * plan.value_dim];
    for token in 0..tokens {
        for head in 0..plan.query_heads {
            let kv_head = plan.kv_head(head);
            let q_offset = (token * plan.query_heads + head) * plan.head_dim;
            let mut scores: Vec<f64> = (0..tokens)
                .map(|key| {
                    let visible = positions[key].segment == positions[token].segment
                        && positions[key].position <= positions[token].position;
                    if !visible {
                        return f64::NEG_INFINITY;
                    }
                    let dot: f32 = (0..plan.head_dim)
                        .map(|dim| {
                            query[q_offset + dim] * views.keys[kv_head][key * plan.head_dim + dim]
                        })
                        .sum();
                    f64::from(dot * scale)
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for score in &mut scores {
                *score = (*score - max).exp();
            }
            let total: f64 = scores.iter().sum();
            for (key, weight) in scores.iter().enumerate() {
                let probability = *weight / total;
                let out_base = (token * plan.query_heads + head) * plan.value_dim;
                for dim in 0..plan.value_dim {
                    output[out_base + dim] +=
                        probability * f64::from(views.values[kv_head][key * plan.value_dim + dim]);
                }
            }
        }
    }
    output.into_iter().map(|value| value as f32).collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "element {index}: {got} vs {want}"
        );
    }
}

#[test]
fn autoregressive_state_cpu_plan_validation() {
    let bad_ratio = kv_plan("bad-ratio", 3, 2, 4);
    assert!(StatePlan::new(vec![LayerPlan::Kv(bad_ratio)]).is_err());
    let zero_width = kv_plan("zero", 2, 2, 0);
    assert!(StatePlan::new(vec![LayerPlan::Kv(zero_width)]).is_err());
    let mut bad_window = kv_plan("window", 2, 2, 2);
    bad_window.sliding_window = Some(3);
    assert!(StatePlan::new(vec![LayerPlan::Kv(bad_window)]).is_err());
    let duplicate = kv_plan("dup", 2, 2, 2);
    assert!(
        StatePlan::new(vec![
            LayerPlan::Kv(duplicate.clone()),
            LayerPlan::Kv(duplicate),
        ])
        .is_err()
    );
    assert!(StatePlan::new(vec![]).is_err());
}

#[test]
fn autoregressive_state_cpu_kv_gqa_append_and_window() {
    let plan = kv_plan("attn", 4, 2, 4);
    assert_eq!(plan.group_size(), 2);
    assert_eq!(plan.kv_head(3), 1);
    let mut state =
        SequenceState::new(StatePlan::new(vec![LayerPlan::Kv(plan)]).unwrap(), 10).unwrap();
    let cancel = RequestCancel::new();
    for index in 0..4 {
        state
            .append_prefill(vec![kv_update(0, index, 2, 3, 2)], &cancel)
            .unwrap();
    }
    assert_eq!(state.next_position(), 14);
    assert_eq!(
        state.positions()[3],
        TokenPosition {
            segment: 0,
            position: 13
        }
    );
    assert_error(
        state.decode(kv_update(0, 4, 2, 3, 2), &cancel),
        "context_capacity",
    );
    assert_eq!(state.len(), 4);

    let mut windowed = kv_plan("windowed", 4, 2, 4);
    windowed.sliding_window = Some(2);
    let mut sliding =
        SequenceState::new(StatePlan::new(vec![LayerPlan::Kv(windowed)]).unwrap(), 0).unwrap();
    for index in 0..4 {
        sliding
            .append_prefill(vec![kv_update(0, index, 2, 3, 2)], &cancel)
            .unwrap();
    }
    let visible = sliding.visible_positions("windowed").unwrap();
    assert_eq!(visible.len(), 2);
    assert_eq!(visible[0].position, 2);
    assert_eq!(visible[1].position, 3);
    sliding.rewind(2).unwrap();
    assert_error(sliding.rewind(1), "evicted");
}

#[test]
fn autoregressive_state_cpu_masks() {
    let plan = kv_plan("attn", 2, 2, 4);
    let mut state =
        SequenceState::new(StatePlan::new(vec![LayerPlan::Kv(plan)]).unwrap(), 0).unwrap();
    let cancel = RequestCancel::new();
    let mut updates = Vec::new();
    for (segment, index) in [(0, 0), (0, 1), (0, 2), (1, 0)] {
        updates.push(kv_update(segment, index, 2, 3, 2));
    }
    state.append_prefill(updates, &cancel).unwrap();

    let causal = state.mask("attn", 0, 4, MaskMode::Causal).unwrap();
    assert_eq!(causal.values[0], 0.0);
    assert_eq!(causal.values[1], f32::NEG_INFINITY);
    assert_eq!(causal.values[2 * 4 + 3], f32::NEG_INFINITY);
    assert_eq!(causal.values[3 * 4 + 3], 0.0);

    let sliding = state
        .mask("attn", 2, 1, MaskMode::Sliding { window: 2 })
        .unwrap();
    assert_eq!(
        sliding.values,
        vec![f32::NEG_INFINITY, 0.0, 0.0, f32::NEG_INFINITY]
    );

    let packed = state
        .mask("attn", 3, 1, MaskMode::Packed { window: None })
        .unwrap();
    assert_eq!(
        packed.values,
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]
    );

    let kv = match state.plan().layer("attn").unwrap() {
        LayerPlan::Kv(kv) => kv,
        _ => unreachable!(),
    };
    assert!(
        build_mask(
            kv,
            &[TokenPosition {
                segment: 2,
                position: 0
            }],
            state.visible_positions("attn").unwrap().as_slice(),
            MaskMode::Packed { window: None },
        )
        .is_err()
    );
}

#[test]
fn autoregressive_state_cpu_prefill_decode_equivalence() {
    let plan = StatePlan::new(vec![
        LayerPlan::Kv(kv_plan("attn", 4, 2, 8)),
        LayerPlan::Recurrent(recurrent_plan()),
    ])
    .unwrap();
    let updates: Vec<TokenUpdate> = (0..4).map(|i| two_layer_update(i, 2)).collect();
    let query: Vec<f32> = (0..4 * 4 * 2).map(|index| index as f32 * 0.125).collect();
    let cancel = RequestCancel::new();

    let mut full = SequenceState::new(plan.clone(), 0).unwrap();
    full.append_prefill(updates.clone(), &cancel).unwrap();
    let mut chunked = SequenceState::new(plan, 0).unwrap();
    chunked
        .append_prefill(updates[..2].to_vec(), &cancel)
        .unwrap();
    for update in &updates[2..] {
        chunked.decode(update.clone(), &cancel).unwrap();
    }

    assert_close(
        &cpu_logits(&full, "attn", &query),
        &cpu_logits(&chunked, "attn", &query),
    );
    assert_eq!(format!("{full:?}"), format!("{chunked:?}"));

    let mut rewound = chunked.copy();
    rewound.rewind(2).unwrap();
    assert_eq!(rewound.len(), 2);
    assert_ne!(format!("{rewound:?}"), format!("{chunked:?}"));
}

#[test]
fn autoregressive_state_cpu_two_sequence_isolation() {
    let plan = StatePlan::new(vec![LayerPlan::Kv(kv_plan("attn", 2, 2, 4))]).unwrap();
    let cancel = RequestCancel::new();
    let mut left = SequenceState::new(plan.clone(), 0).unwrap();
    let mut right = SequenceState::new(plan, 0).unwrap();
    left.append_prefill(vec![kv_update(0, 1, 2, 3, 2)], &cancel)
        .unwrap();
    right
        .append_prefill(vec![kv_update(0, 9, 2, 3, 2)], &cancel)
        .unwrap();
    let right_before = format!("{right:?}");
    let snapshot = left.copy();
    left.rewind(0).unwrap();
    left.clear();
    assert_eq!(format!("{right:?}"), right_before);
    assert_eq!(snapshot.len(), 1);
    assert_eq!(
        snapshot.attention_views("attn").unwrap().keys[0],
        vec![1.125, 1.25]
    );
}

#[test]
fn autoregressive_state_cpu_cancellation_boundaries() {
    let plan = StatePlan::new(vec![LayerPlan::Kv(kv_plan("attn", 2, 2, 4))]).unwrap();
    let mut state = SequenceState::new(plan, 0).unwrap();
    let cancel = RequestCancel::new();
    state
        .append_prefill(vec![kv_update(0, 0, 2, 3, 2)], &cancel)
        .unwrap();
    cancel.cancel();
    assert_error(
        state.append_prefill(vec![kv_update(0, 1, 2, 3, 2)], &cancel),
        "cancelled",
    );
    assert_error(state.decode(kv_update(0, 1, 2, 3, 2), &cancel), "cancelled");
    assert_eq!(state.len(), 1);
    assert!(matches!(
        honkbox_llm_worker::LlmError::Cancelled,
        LlmError::Cancelled
    ));
}

#[test]
fn autoregressive_state_cpu_byte_accounting() {
    let plan = StatePlan::new(vec![LayerPlan::Kv(kv_plan("attn", 4, 2, 4))]).unwrap();
    let state = SequenceState::new(plan, 0).unwrap();
    let bytes = state.bytes_for(1).unwrap();
    assert_eq!(bytes.kv_bytes, 4 * 2 * (2 + 3) * 4);
    assert_eq!(bytes.recurrent_bytes, 0);
    assert_eq!(
        bytes.ram_bytes,
        bytes.kv_bytes + bytes.position_bytes + bytes.metadata_bytes + bytes.scratch_bytes
    );
    assert_eq!(bytes.vram_bytes, bytes.scratch_bytes);
    let demand = Demand {
        gpu_slots: 1,
        vram_bytes: bytes.vram_bytes,
        ram_bytes: bytes.ram_bytes,
        cpu_threads: 1,
    };
    assert_eq!(demand.vram_bytes, bytes.vram_bytes);
    assert_eq!(demand.ram_bytes, bytes.ram_bytes);
}

#[test]
#[ignore = "requires HONKBOX_ASH_TEST_DEVICE and an idle named Vulkan device"]
fn autoregressive_state_cpu_named_vulkan_gqa_proof() {
    let expected = match std::env::var("HONKBOX_ASH_TEST_DEVICE") {
        Ok(name) => name,
        Err(_) => return,
    };
    let device = match honkbox_ash_backend::device::VulkanDevice::headless() {
        Ok(device) => std::sync::Arc::new(device),
        Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => return,
        Err(error) => panic!("vulkan api failed: {error:?}"),
    };
    assert_eq!(device.info.name, expected, "unexpected Vulkan device");
    let attention = Attention::new(device).expect("attention pipelines");
    let plan = StatePlan::new(vec![LayerPlan::Kv(kv_plan("attn", 4, 2, 8))]).unwrap();
    let mut state = SequenceState::new(plan, 0).unwrap();
    let cancel = RequestCancel::new();
    let updates: Vec<TokenUpdate> = (0..4).map(|i| kv_update(0, i, 2, 3, 2)).collect();
    let query: Vec<f32> = (0..4 * 4 * 2).map(|index| index as f32 * 0.125).collect();
    state.append_prefill(updates, &cancel).unwrap();
    let actual = state
        .attend("attn", 0, &query, MaskMode::Causal, &attention, &cancel)
        .unwrap();
    assert_close(&actual, &cpu_logits(&state, "attn", &query));
    assert_eq!(state.query_to_kv_head("attn", 3).unwrap(), 1);
}
