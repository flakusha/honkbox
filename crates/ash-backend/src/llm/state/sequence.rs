//! Request-local KV/recurrent storage, exact rewind and attention dispatch.

use super::mask::{AdditiveMask, MaskMode, TokenPosition, build_mask};
use super::plan::{
    LayerPlan, MatrixPlan, RecurrentPlan, StatePlan, checked_add, checked_mul, error, nonzero,
    u32_shape,
};
use crate::attention::{Attention, AttentionShape};
use honkbox_llm_worker::{LlmError, RequestCancel};
use std::collections::VecDeque;

/// Caller-produced state for one layer at one token.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerUpdate {
    /// Unique-head K in `[kv_head, head_dim]` order.
    pub key: Vec<f32>,
    /// Unique-head V in `[kv_head, value_dim]` order.
    pub value: Vec<f32>,
    /// Complete post-token convolution history in `[channel, history]` order.
    pub convolution: Vec<f32>,
    /// Complete post-token row-major matrix slots in plan order.
    pub matrices: Vec<Vec<f32>>,
}

/// Caller-produced updates for one logical token.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenUpdate {
    /// Packed segment containing this token.
    pub segment: u32,
    /// One update per plan layer, in plan order.
    pub layers: Vec<LayerUpdate>,
}

/// Unique-head attention inputs in transient token-major layout.
#[derive(Debug, Clone, PartialEq)]
pub struct KvAttentionViews {
    /// One `[token, head_dim]` buffer per unique KV head.
    pub keys: Vec<Vec<f32>>,
    /// One `[token, value_dim]` buffer per unique KV head.
    pub values: Vec<Vec<f32>>,
    /// Absolute positions of the retained keys.
    pub positions: Vec<TokenPosition>,
}

/// Exact weight-independent state and state-owned scratch byte accounting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StateBytes {
    /// Allocated unique-head K/V element storage.
    pub kv_bytes: u64,
    /// Current recurrent slots plus retained exact rewind snapshots.
    pub recurrent_bytes: u64,
    /// Token positions and per-segment position counters.
    pub position_bytes: u64,
    /// Plan/container metadata and diagnostic names.
    pub metadata_bytes: u64,
    /// State-owned host/device-visible scratch for the requested query shape.
    pub scratch_bytes: u64,
    /// Persistent state plus state-owned scratch resident in host memory.
    pub ram_bytes: u64,
    /// State-owned scratch crossing the device residency seam.
    pub vram_bytes: u64,
}

#[derive(Debug, Clone)]
enum LayerState {
    Stateless,
    Kv(KvState),
    Recurrent(RecurrentState),
}

#[derive(Debug, Clone)]
struct KvState {
    keys: Vec<VecDeque<Vec<f32>>>,
    values: Vec<VecDeque<Vec<f32>>>,
}

#[derive(Debug, Clone)]
struct RecurrentState {
    convolution: Vec<f32>,
    matrices: Vec<Vec<f32>>,
    convolution_history: Vec<Vec<f32>>,
    matrix_history: Vec<Vec<Vec<f32>>>,
}

/// One sequence's complete request-local state.
///
/// Cloning is a deep copy. No buffer, position counter or rewind snapshot is
/// shared with another sequence. The runtime stores caller-supplied values; it
/// deliberately contains no family update equations, tokenizer or sampler.
#[derive(Debug, Clone)]
pub struct SequenceState {
    plan: StatePlan,
    position_offset: u64,
    positions: Vec<TokenPosition>,
    segment_positions: Vec<(u32, u64)>,
    layers: Vec<LayerState>,
}

impl SequenceState {
    /// Allocate validated empty state for one request.
    pub fn new(plan: StatePlan, position_offset: u64) -> Result<Self, LlmError> {
        position_offset
            .checked_add(u64::from(u32::MAX))
            .ok_or_else(|| error("construct", "sequence", "position_offset", "overflow"))?;
        let layers = empty_layers(&plan)?;
        Ok(Self {
            plan,
            position_offset,
            positions: Vec::new(),
            segment_positions: Vec::new(),
            layers,
        })
    }

    /// The immutable validated plan.
    #[must_use]
    pub fn plan(&self) -> &StatePlan {
        &self.plan
    }

    /// Number of committed logical tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether no token has been committed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Next sequence-wide absolute position, independent of physical slots.
    #[must_use]
    pub fn next_position(&self) -> u64 {
        self.position_offset + self.len() as u64
    }

    /// Base offset applied independently inside every packed segment.
    #[must_use]
    pub fn position_offset(&self) -> u64 {
        self.position_offset
    }

    /// Committed token positions in logical order.
    #[must_use]
    pub fn positions(&self) -> &[TokenPosition] {
        &self.positions
    }

    /// Deep-copy this sequence, including all buffers and rewind snapshots.
    #[must_use]
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Reset this sequence while retaining its validated plan and allocations.
    pub fn clear(&mut self) {
        self.positions.clear();
        self.segment_positions.clear();
        for state in &mut self.layers {
            match state {
                LayerState::Stateless => {}
                LayerState::Kv(kv) => {
                    for head in &mut kv.keys {
                        head.clear();
                    }
                    for head in &mut kv.values {
                        head.clear();
                    }
                }
                LayerState::Recurrent(recurrent) => {
                    recurrent.convolution.fill(0.0);
                    for matrix in &mut recurrent.matrices {
                        matrix.fill(0.0);
                    }
                    recurrent.convolution_history.clear();
                    recurrent.matrix_history.clear();
                    recurrent
                        .convolution_history
                        .push(vec![0.0; recurrent.convolution.len()]);
                    recurrent.matrix_history.push(
                        recurrent
                            .matrices
                            .iter()
                            .map(|matrix| vec![0.0; matrix.len()])
                            .collect(),
                    );
                }
            }
        }
    }

    /// Append one decode token atomically.
    pub fn decode(&mut self, update: TokenUpdate, cancel: &RequestCancel) -> Result<(), LlmError> {
        self.append_chunk(vec![update], "decode", cancel, |_| {})
    }

    /// Append a prefill chunk atomically, checking cancellation before every token.
    pub fn append_prefill(
        &mut self,
        updates: Vec<TokenUpdate>,
        cancel: &RequestCancel,
    ) -> Result<(), LlmError> {
        self.append_prefill_impl(updates, cancel, |_| {})
    }

    fn append_prefill_impl(
        &mut self,
        updates: Vec<TokenUpdate>,
        cancel: &RequestCancel,
        before_append: impl FnMut(usize),
    ) -> Result<(), LlmError> {
        self.append_chunk(updates, "prefill", cancel, before_append)
    }

    fn append_chunk(
        &mut self,
        updates: Vec<TokenUpdate>,
        op: &str,
        cancel: &RequestCancel,
        mut before_append: impl FnMut(usize),
    ) -> Result<(), LlmError> {
        check_cancel(cancel)?;
        nonzero(updates.len(), op, "sequence", "updates")?;
        self.validate_updates(&updates, op)?;
        // ponytail: whole-chunk deep checkpoint; replace with an undo log if
        // profiling shows large sliding chunks make this copy dominate.
        let checkpoint = self.clone();
        for (index, update) in updates.into_iter().enumerate() {
            check_cancel(cancel)?;
            before_append(index);
            check_cancel(cancel)?;
            if let Err(failure) = self.append_validated(update, op) {
                *self = checkpoint;
                return Err(failure);
            }
        }
        Ok(())
    }

    fn validate_updates(&self, updates: &[TokenUpdate], op: &str) -> Result<(), LlmError> {
        let count = checked_add(self.len(), updates.len(), op, "sequence", "token count")?;
        u32_shape(count, op, "sequence", "token count")?;
        self.next_position()
            .checked_add(updates.len() as u64)
            .ok_or_else(|| error(op, "sequence", "position", "overflow"))?;
        for update in updates {
            if update.layers.len() != self.plan.layers().len() {
                return Err(error(
                    op,
                    "sequence",
                    "layers",
                    &format!(
                        "update count {} must be {}",
                        update.layers.len(),
                        self.plan.layers().len()
                    ),
                ));
            }
            for (layer_index, (layer_plan, values)) in
                self.plan.layers().iter().zip(&update.layers).enumerate()
            {
                let name = layer_plan.name();
                match (layer_plan, &self.layers[layer_index]) {
                    (LayerPlan::Stateless { .. }, LayerState::Stateless) => {
                        require_empty(values, op, name)?;
                    }
                    (LayerPlan::Kv(plan), LayerState::Kv(kv)) => {
                        require_empty_recurrent(values, op, name)?;
                        let key_len = checked_mul(plan.kv_heads, plan.head_dim, op, name, "key")?;
                        let value_len =
                            checked_mul(plan.kv_heads, plan.value_dim, op, name, "value")?;
                        require_values(&values.key, key_len, op, name, "key")?;
                        require_values(&values.value, value_len, op, name, "value")?;
                        let remaining = updates.len();
                        let retained = kv.keys[0].len();
                        let final_len =
                            checked_add(retained, remaining, op, name, "context length")?;
                        if plan.sliding_window.is_none() && final_len > plan.context_capacity {
                            return Err(error(
                                op,
                                name,
                                "context_capacity",
                                &format!("{final_len} exceeds {}", plan.context_capacity),
                            ));
                        }
                    }
                    (LayerPlan::Recurrent(plan), LayerState::Recurrent(_)) => {
                        require_empty_kv(values, op, name)?;
                        validate_recurrent_update(values, plan, op)?;
                    }
                    _ => return Err(error(op, name, "layer", "plan/storage kind mismatch")),
                }
            }
        }
        Ok(())
    }

    fn append_validated(&mut self, update: TokenUpdate, op: &str) -> Result<(), LlmError> {
        let position = TokenPosition {
            segment: update.segment,
            position: self.segment_next(update.segment),
        };
        for (layer_index, (layer_plan, values)) in
            self.plan.layers().iter().zip(update.layers).enumerate()
        {
            let state = &mut self.layers[layer_index];
            match (layer_plan, state, values) {
                (LayerPlan::Stateless { .. }, LayerState::Stateless, _) => {}
                (LayerPlan::Kv(plan), LayerState::Kv(kv), LayerUpdate { key, value, .. }) => {
                    let retention = plan.sliding_window.unwrap_or(plan.context_capacity);
                    if kv.keys[0].len() >= retention {
                        for head in &mut kv.keys {
                            head.pop_front();
                        }
                        for head in &mut kv.values {
                            head.pop_front();
                        }
                    }
                    for (values, head) in key.chunks_exact(plan.head_dim).zip(&mut kv.keys) {
                        head.try_reserve(1)
                            .map_err(|_| error(op, &plan.name, "key", "allocation failed"))?;
                        head.push_back(values.to_vec());
                    }
                    for (values, head) in value.chunks_exact(plan.value_dim).zip(&mut kv.values) {
                        head.try_reserve(1)
                            .map_err(|_| error(op, &plan.name, "value", "allocation failed"))?;
                        head.push_back(values.to_vec());
                    }
                }
                (
                    LayerPlan::Recurrent(plan),
                    LayerState::Recurrent(recurrent),
                    LayerUpdate {
                        convolution,
                        matrices,
                        ..
                    },
                ) => {
                    recurrent.convolution = convolution;
                    recurrent.matrices = matrices;
                    recurrent.convolution_history.try_reserve(1).map_err(|_| {
                        error(op, &plan.name, "convolution snapshot", "allocation failed")
                    })?;
                    recurrent.matrix_history.try_reserve(1).map_err(|_| {
                        error(op, &plan.name, "matrix snapshot", "allocation failed")
                    })?;
                    recurrent
                        .convolution_history
                        .push(recurrent.convolution.clone());
                    recurrent.matrix_history.push(recurrent.matrices.clone());
                }
                _ => {
                    return Err(error(
                        op,
                        layer_plan.name(),
                        "layer",
                        "plan/storage kind mismatch",
                    ));
                }
            }
        }
        self.positions
            .try_reserve(1)
            .map_err(|_| error(op, "sequence", "positions", "allocation failed"))?;
        if !self
            .segment_positions
            .iter()
            .any(|(segment, _)| *segment == position.segment)
        {
            self.segment_positions
                .try_reserve(1)
                .map_err(|_| error(op, "sequence", "segment positions", "allocation failed"))?;
            self.segment_positions
                .push((position.segment, self.position_offset));
        }
        if let Some((_, next)) = self
            .segment_positions
            .iter_mut()
            .find(|(segment, _)| *segment == position.segment)
        {
            *next = next
                .checked_add(1)
                .ok_or_else(|| error(op, "sequence", "segment position", "overflow"))?;
        }
        self.positions.push(position);
        Ok(())
    }

    fn segment_next(&self, segment: u32) -> u64 {
        self.segment_positions
            .iter()
            .find(|(candidate, _)| *candidate == segment)
            .map_or(self.position_offset, |(_, next)| *next)
    }

    /// Restore exactly `target` committed tokens.
    ///
    /// KV rewind is exact within each layer's retained context. Rewinding to an
    /// already-evicted KV token is rejected by layer name rather than guessed;
    /// recurrent state always replays an exact retained caller snapshot.
    pub fn rewind(&mut self, target: usize) -> Result<(), LlmError> {
        if target > self.len() {
            return Err(error(
                "rewind",
                "sequence",
                "target",
                &format!("{} exceeds committed {}", target, self.len()),
            ));
        }
        for (layer_plan, state) in self.plan.layers().iter().zip(&mut self.layers) {
            match (layer_plan, state) {
                (LayerPlan::Kv(plan), LayerState::Kv(kv)) => {
                    let oldest = self.positions.len() - kv.keys[0].len();
                    if target < oldest {
                        return Err(error(
                            "rewind",
                            &plan.name,
                            "target",
                            &format!("token {target} was evicted; oldest retained is {oldest}"),
                        ));
                    }
                    let keep = target - oldest;
                    for head in &mut kv.keys {
                        while head.len() > keep {
                            head.pop_back();
                        }
                    }
                    for head in &mut kv.values {
                        while head.len() > keep {
                            head.pop_back();
                        }
                    }
                }
                (LayerPlan::Recurrent(_), LayerState::Recurrent(recurrent)) => {
                    recurrent.convolution = recurrent.convolution_history[target].clone();
                    recurrent.matrices = recurrent.matrix_history[target].clone();
                    recurrent.convolution_history.truncate(target + 1);
                    recurrent.matrix_history.truncate(target + 1);
                }
                _ => {}
            }
        }
        self.positions.truncate(target);
        self.segment_positions.clear();
        for position in &self.positions {
            let next = position
                .position
                .checked_add(1)
                .ok_or_else(|| error("rewind", "sequence", "segment position", "overflow"))?;
            if let Some((_, existing)) = self
                .segment_positions
                .iter_mut()
                .find(|(segment, _)| *segment == position.segment)
            {
                *existing = next;
            } else {
                self.segment_positions.push((position.segment, next));
            }
        }
        Ok(())
    }

    /// Build a mask for committed query rows against this layer's retained keys.
    pub fn mask(
        &self,
        layer: &str,
        query_start: usize,
        query_len: usize,
        mode: MaskMode,
    ) -> Result<AdditiveMask, LlmError> {
        check_cancel_boundary(query_start, query_len, self.len(), "mask", layer)?;
        let plan = self.plan.kv(layer)?;
        let queries = &self.positions[query_start..query_start + query_len];
        let keys = self.visible_positions(layer)?;
        build_mask(plan, queries, &keys, mode)
    }

    /// Materialize unique-head K/V views without expanding grouped heads.
    pub fn attention_views(&self, layer: &str) -> Result<KvAttentionViews, LlmError> {
        let plan = self.plan.kv(layer)?;
        let layer_index = self
            .plan
            .layers()
            .iter()
            .position(|plan| plan.name() == layer)
            .ok_or_else(|| error("views", layer, "layer", "is not in the plan"))?;
        let LayerState::Kv(kv) = &self.layers[layer_index] else {
            return Err(error("views", layer, "layer", "must be KV"));
        };
        let len = kv.keys[0].len();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        keys.try_reserve_exact(plan.kv_heads)
            .map_err(|_| error("views", layer, "keys", "allocation failed"))?;
        values
            .try_reserve_exact(plan.kv_heads)
            .map_err(|_| error("views", layer, "values", "allocation failed"))?;
        for head in 0..plan.kv_heads {
            let key_len = checked_mul(len, plan.head_dim, "views", layer, "key")?;
            let value_len = checked_mul(len, plan.value_dim, "views", layer, "value")?;
            u32_shape(key_len, "views", layer, "key")?;
            u32_shape(value_len, "views", layer, "value")?;
            let mut key = Vec::new();
            let mut value = Vec::new();
            key.try_reserve_exact(key_len)
                .map_err(|_| error("views", layer, "key", "allocation failed"))?;
            value
                .try_reserve_exact(value_len)
                .map_err(|_| error("views", layer, "value", "allocation failed"))?;
            for token in 0..len {
                key.extend_from_slice(&kv.keys[head][token]);
                value.extend_from_slice(&kv.values[head][token]);
            }
            keys.push(key);
            values.push(value);
        }
        Ok(KvAttentionViews {
            keys,
            values,
            positions: self.visible_positions(layer)?,
        })
    }

    /// Map one query head to the single unique KV head it reads.
    pub fn query_to_kv_head(&self, layer: &str, query_head: usize) -> Result<usize, LlmError> {
        let plan = self.plan.kv(layer)?;
        if query_head >= plan.query_heads {
            return Err(error(
                "attend",
                layer,
                "query_head",
                &format!("{} exceeds {}", query_head, plan.query_heads),
            ));
        }
        Ok(plan.kv_head(query_head))
    }

    /// Run existing masked attention once per KV group over unique-head views.
    ///
    /// Query layout is `[token, query_head, head_dim]`; output layout is
    /// `[token, query_head, value_dim]`. No expanded K/V copy is retained.
    pub fn attend(
        &self,
        layer: &str,
        query_start: usize,
        query: &[f32],
        mode: MaskMode,
        attention: &Attention,
        cancel: &RequestCancel,
    ) -> Result<Vec<f32>, LlmError> {
        self.attend_impl(layer, query_start, query, mode, attention, cancel, || {})
    }

    #[allow(clippy::too_many_arguments)]
    fn attend_impl(
        &self,
        layer: &str,
        query_start: usize,
        query: &[f32],
        mode: MaskMode,
        attention: &Attention,
        cancel: &RequestCancel,
        after_readback: impl FnOnce(),
    ) -> Result<Vec<f32>, LlmError> {
        check_cancel(cancel)?;
        let plan = self.plan.kv(layer)?;
        let per_token = checked_mul(plan.query_heads, plan.head_dim, "attend", layer, "query")?;
        if per_token == 0 || !query.len().is_multiple_of(per_token) {
            return Err(error(
                "attend",
                layer,
                "query",
                "length must be token_count * query_heads * head_dim",
            ));
        }
        let query_len = query.len() / per_token;
        check_cancel_boundary(query_start, query_len, self.len(), "attend", layer)?;
        require_values(query, query.len(), "attend", layer, "query")?;
        let mask = self.mask(layer, query_start, query_len, mode)?;
        let views = self.attention_views(layer)?;
        let key_len = views.positions.len();
        let output_len = checked_mul(query_len, plan.query_heads, "attend", layer, "output")?
            .checked_mul(plan.value_dim)
            .ok_or_else(|| error("attend", layer, "output", "element count overflow"))?;
        u32_shape(output_len, "attend", layer, "output")?;
        let mut output = zero_vec(output_len, "attend", layer, "output")?;
        for kv_head in 0..plan.kv_heads {
            check_cancel(cancel)?;
            let first_query_head = kv_head * plan.group_size();
            let q_len = checked_mul(query_len, plan.group_size(), "attend", layer, "query group")?
                .checked_mul(plan.head_dim)
                .ok_or_else(|| error("attend", layer, "query group", "element count overflow"))?;
            u32_shape(q_len, "attend", layer, "query group")?;
            let mut q_group = Vec::new();
            q_group
                .try_reserve_exact(q_len)
                .map_err(|_| error("attend", layer, "query group", "allocation failed"))?;
            for token in 0..query_len {
                for query_head in first_query_head..first_query_head + plan.group_size() {
                    let start = (token * plan.query_heads + query_head) * plan.head_dim;
                    q_group.extend_from_slice(&query[start..start + plan.head_dim]);
                }
            }
            let k_stride = checked_mul(plan.head_dim, key_len, "attend", layer, "key stride")?;
            let k_t_len = checked_mul(
                k_stride,
                plan.group_size(),
                "attend",
                layer,
                "key transpose",
            )?;
            u32_shape(k_t_len, "attend", layer, "key transpose")?;
            let mut k_t = zero_vec(k_t_len, "attend", layer, "key transpose")?;
            for head in 0..plan.group_size() {
                for dim in 0..plan.head_dim {
                    for token in 0..key_len {
                        k_t[head * k_stride + dim * key_len + token] =
                            views.keys[kv_head][token * plan.head_dim + dim];
                    }
                }
            }
            let v_len = checked_mul(key_len, plan.group_size(), "attend", layer, "values")?
                .checked_mul(plan.value_dim)
                .ok_or_else(|| error("attend", layer, "values", "element count overflow"))?;
            u32_shape(v_len, "attend", layer, "values")?;
            let mut v_group = zero_vec(v_len, "attend", layer, "values")?;
            for token in 0..key_len {
                for head in 0..plan.group_size() {
                    for dim in 0..plan.value_dim {
                        v_group[(token * plan.group_size() + head) * plan.value_dim + dim] =
                            views.values[kv_head][token * plan.value_dim + dim];
                    }
                }
            }
            let shape = AttentionShape {
                seq_q: query_len,
                seq_k: key_len,
                head_dim: plan.head_dim,
                value_dim: plan.value_dim,
                heads: plan.group_size(),
            };
            let group_output = attention
                .run_masked(&q_group, &k_t, &v_group, shape, Some(&mask.values))
                .map_err(|failure| {
                    LlmError::Decode(format!(
                        "state attend: layer '{layer}': attention {failure}"
                    ))
                })?;
            for token in 0..query_len {
                for head in 0..plan.group_size() {
                    let src = (token * plan.group_size() + head) * plan.value_dim;
                    let dst = (token * plan.query_heads + first_query_head + head) * plan.value_dim;
                    output[dst..dst + plan.value_dim]
                        .copy_from_slice(&group_output[src..src + plan.value_dim]);
                }
            }
        }
        after_readback();
        check_cancel(cancel)?;
        Ok(output)
    }

    /// Append and attend one decode token; cancellation after readback rolls it back.
    pub fn decode_attention(
        &mut self,
        layer: &str,
        update: TokenUpdate,
        query: &[f32],
        mode: MaskMode,
        attention: &Attention,
        cancel: &RequestCancel,
    ) -> Result<Vec<f32>, LlmError> {
        check_cancel(cancel)?;
        let start = self.len();
        let checkpoint = self.clone();
        self.decode(update, cancel)?;
        match self.attend(layer, start, query, mode, attention, cancel) {
            Ok(output) => Ok(output),
            Err(failure) => {
                *self = checkpoint;
                Err(failure)
            }
        }
    }

    /// Append and attend one prefill chunk atomically.
    pub fn prefill_attention(
        &mut self,
        layer: &str,
        updates: Vec<TokenUpdate>,
        queries: &[f32],
        mode: MaskMode,
        attention: &Attention,
        cancel: &RequestCancel,
    ) -> Result<Vec<f32>, LlmError> {
        check_cancel(cancel)?;
        let start = self.len();
        let checkpoint = self.clone();
        self.append_prefill(updates, cancel)?;
        match self.attend(layer, start, queries, mode, attention, cancel) {
            Ok(output) => Ok(output),
            Err(failure) => {
                *self = checkpoint;
                Err(failure)
            }
        }
    }

    /// Retained positions for one KV layer in physical key order.
    pub fn visible_positions(&self, layer: &str) -> Result<Vec<TokenPosition>, LlmError> {
        let layer_index = self
            .plan
            .layers()
            .iter()
            .position(|plan| plan.name() == layer)
            .ok_or_else(|| error("positions", layer, "layer", "is not in the plan"))?;
        let LayerState::Kv(kv) = &self.layers[layer_index] else {
            return Err(error("positions", layer, "layer", "must be KV"));
        };
        let len = kv.keys[0].len();
        Ok(self.positions[self.len() - len..].to_vec())
    }

    /// Exact allocated state bytes plus scratch for `query_tokens` rows.
    #[allow(clippy::too_many_lines)]
    pub fn bytes_for(&self, query_tokens: usize) -> Result<StateBytes, LlmError> {
        nonzero(query_tokens, "bytes", "sequence", "query_tokens")?;
        u32_shape(query_tokens, "bytes", "sequence", "query_tokens")?;
        let mut kv_bytes = 0;
        let mut recurrent_bytes = 0;
        for (layer_plan, state) in self.plan.layers().iter().zip(&self.layers) {
            let name = layer_plan.name();
            match (layer_plan, state) {
                (LayerPlan::Kv(plan), LayerState::Kv(kv)) => {
                    for head in &kv.keys {
                        kv_bytes = add_bytes(
                            kv_bytes,
                            allocated_bytes(
                                head.capacity(),
                                plan.head_dim * size_of::<f32>(),
                                "bytes",
                                name,
                                "key",
                            )?,
                            name,
                            "key",
                        )?;
                    }
                    for head in &kv.values {
                        kv_bytes = add_bytes(
                            kv_bytes,
                            allocated_bytes(
                                head.capacity(),
                                plan.value_dim * size_of::<f32>(),
                                "bytes",
                                name,
                                "value",
                            )?,
                            name,
                            "value",
                        )?;
                    }
                }
                (LayerPlan::Recurrent(_), LayerState::Recurrent(recurrent)) => {
                    recurrent_bytes = add_bytes(
                        recurrent_bytes,
                        allocated_bytes(
                            recurrent.convolution.capacity(),
                            size_of::<f32>(),
                            "bytes",
                            name,
                            "convolution",
                        )?,
                        name,
                        "convolution",
                    )?;
                    for matrix in &recurrent.matrices {
                        recurrent_bytes = add_bytes(
                            recurrent_bytes,
                            allocated_bytes(
                                matrix.capacity(),
                                size_of::<f32>(),
                                "bytes",
                                name,
                                "matrix",
                            )?,
                            name,
                            "matrix",
                        )?;
                    }
                    for snapshot in &recurrent.convolution_history {
                        recurrent_bytes = add_bytes(
                            recurrent_bytes,
                            allocated_bytes(
                                snapshot.capacity(),
                                size_of::<f32>(),
                                "bytes",
                                name,
                                "convolution snapshot",
                            )?,
                            name,
                            "convolution snapshot",
                        )?;
                    }
                    for snapshots in &recurrent.matrix_history {
                        for snapshot in snapshots {
                            recurrent_bytes = add_bytes(
                                recurrent_bytes,
                                allocated_bytes(
                                    snapshot.capacity(),
                                    size_of::<f32>(),
                                    "bytes",
                                    name,
                                    "matrix snapshot",
                                )?,
                                name,
                                "matrix snapshot",
                            )?;
                        }
                    }
                }
                _ => {}
            }
        }
        let position_bytes = allocated_bytes(
            self.positions.capacity(),
            size_of::<TokenPosition>(),
            "bytes",
            "sequence",
            "positions",
        )?
        .checked_add(allocated_bytes(
            self.segment_positions.capacity(),
            size_of::<(u32, u64)>(),
            "bytes",
            "sequence",
            "segment positions",
        )?)
        .ok_or_else(|| error("bytes", "sequence", "position_bytes", "overflow"))?;
        let metadata_bytes = self.metadata_bytes()?;
        let scratch_bytes = self.scratch_bytes(query_tokens)?;
        let persistent = [kv_bytes, recurrent_bytes, position_bytes, metadata_bytes]
            .into_iter()
            .try_fold(0_u64, u64::checked_add)
            .ok_or_else(|| error("bytes", "sequence", "ram_bytes", "overflow"))?;
        let ram_bytes = persistent
            .checked_add(scratch_bytes)
            .ok_or_else(|| error("bytes", "sequence", "ram_bytes", "overflow"))?;
        Ok(StateBytes {
            kv_bytes,
            recurrent_bytes,
            position_bytes,
            metadata_bytes,
            scratch_bytes,
            ram_bytes,
            vram_bytes: scratch_bytes,
        })
    }

    /// Exact bytes for a full-context scratch bound.
    pub fn bytes(&self) -> Result<StateBytes, LlmError> {
        let query_tokens = self
            .plan
            .layers()
            .iter()
            .filter_map(|layer| match layer {
                LayerPlan::Kv(plan) => Some(plan.context_capacity),
                _ => None,
            })
            .max()
            .unwrap_or(1);
        self.bytes_for(query_tokens)
    }

    fn metadata_bytes(&self) -> Result<u64, LlmError> {
        let mut bytes =
            allocated_bytes(size_of::<Self>(), 1, "bytes", "sequence", "inline metadata")?;
        bytes = add_bytes(
            bytes,
            allocated_bytes(
                self.layers.capacity(),
                size_of::<LayerState>(),
                "bytes",
                "sequence",
                "layers",
            )?,
            "sequence",
            "layers",
        )?;
        bytes = add_bytes(bytes, self.plan.metadata_bytes()?, "sequence", "plan")?;
        for state in &self.layers {
            match state {
                LayerState::Stateless => {}
                LayerState::Kv(kv) => {
                    bytes = add_bytes(
                        bytes,
                        allocated_bytes(
                            kv.keys.capacity(),
                            size_of::<VecDeque<Vec<f32>>>(),
                            "bytes",
                            "sequence",
                            "key heads",
                        )?,
                        "sequence",
                        "key heads",
                    )?;
                    bytes = add_bytes(
                        bytes,
                        allocated_bytes(
                            kv.values.capacity(),
                            size_of::<VecDeque<Vec<f32>>>(),
                            "bytes",
                            "sequence",
                            "value heads",
                        )?,
                        "sequence",
                        "value heads",
                    )?;
                }
                LayerState::Recurrent(recurrent) => {
                    bytes = add_bytes(
                        bytes,
                        allocated_bytes(
                            recurrent.matrices.capacity(),
                            size_of::<Vec<f32>>(),
                            "bytes",
                            "sequence",
                            "matrices",
                        )?,
                        "sequence",
                        "matrices",
                    )?;
                    bytes = add_bytes(
                        bytes,
                        allocated_bytes(
                            recurrent.convolution_history.capacity(),
                            size_of::<Vec<f32>>(),
                            "bytes",
                            "sequence",
                            "convolution history",
                        )?,
                        "sequence",
                        "convolution history",
                    )?;
                    bytes = add_bytes(
                        bytes,
                        allocated_bytes(
                            recurrent.matrix_history.capacity(),
                            size_of::<Vec<Vec<f32>>>(),
                            "bytes",
                            "sequence",
                            "matrix history",
                        )?,
                        "sequence",
                        "matrix history",
                    )?;
                    for matrices in &recurrent.matrix_history {
                        bytes = add_bytes(
                            bytes,
                            allocated_bytes(
                                matrices.capacity(),
                                size_of::<Vec<f32>>(),
                                "bytes",
                                "sequence",
                                "matrix snapshot slots",
                            )?,
                            "sequence",
                            "matrix snapshot slots",
                        )?;
                    }
                }
            }
        }
        Ok(bytes)
    }

    fn scratch_bytes(&self, query_tokens: usize) -> Result<u64, LlmError> {
        let mut total = 0;
        for layer in self.plan.layers() {
            let LayerPlan::Kv(plan) = layer else { continue };
            let name = plan.name.as_str();
            let visible = query_tokens.min(plan.context_capacity);
            let heads = u64::from(u32::try_from(plan.kv_heads).expect("validated u32"));
            let query = u64::from(u32::try_from(query_tokens).expect("validated u32"));
            let keys = u64::from(u32::try_from(visible).expect("validated u32"));
            let mut scratch = 0;
            scratch = add_bytes(
                scratch,
                keys * heads
                    * u64::from(u32::try_from(plan.head_dim).expect("validated u32"))
                    * size_of::<f32>() as u64,
                name,
                "key views",
            )?;
            scratch = add_bytes(
                scratch,
                keys * heads
                    * u64::from(u32::try_from(plan.value_dim).expect("validated u32"))
                    * size_of::<f32>() as u64,
                name,
                "value views",
            )?;
            scratch = add_bytes(
                scratch,
                query * keys * size_of::<f32>() as u64,
                name,
                "mask",
            )?;
            scratch = add_bytes(
                scratch,
                (query + keys) * size_of::<TokenPosition>() as u64,
                name,
                "mask/view positions",
            )?;
            let group = u64::from(u32::try_from(plan.group_size()).expect("validated u32"));
            scratch = add_bytes(
                scratch,
                query
                    * group
                    * u64::from(u32::try_from(plan.head_dim).expect("validated u32"))
                    * size_of::<f32>() as u64,
                name,
                "query group",
            )?;
            scratch = add_bytes(
                scratch,
                keys * u64::from(u32::try_from(plan.head_dim).expect("validated u32"))
                    * size_of::<f32>() as u64,
                name,
                "key transpose",
            )?;
            scratch = add_bytes(
                scratch,
                query
                    * group
                    * u64::from(u32::try_from(plan.value_dim).expect("validated u32"))
                    * size_of::<f32>() as u64,
                name,
                "group output",
            )?;
            scratch = add_bytes(
                scratch,
                query
                    * u64::from(u32::try_from(plan.query_heads).expect("validated u32"))
                    * u64::from(u32::try_from(plan.value_dim).expect("validated u32"))
                    * size_of::<f32>() as u64,
                name,
                "output",
            )?;
            total = add_bytes(total, scratch, name, "scratch")?;
        }
        Ok(total)
    }
}

impl StatePlan {
    fn metadata_bytes(&self) -> Result<u64, LlmError> {
        let mut bytes = allocated_bytes(
            self.layers().len(),
            size_of::<LayerPlan>(),
            "bytes",
            "sequence",
            "plan layers",
        )?;
        for layer in self.layers() {
            bytes = add_bytes(
                bytes,
                allocated_bytes(layer.name().len(), 1, "bytes", layer.name(), "name")?,
                layer.name(),
                "name",
            )?;
            match layer {
                LayerPlan::Stateless { .. } | LayerPlan::Kv(_) => {}
                LayerPlan::Recurrent(plan) => {
                    if let Some(convolution) = &plan.convolution {
                        bytes = add_bytes(
                            bytes,
                            allocated_bytes(
                                convolution.name.len(),
                                1,
                                "bytes",
                                &plan.name,
                                "convolution name",
                            )?,
                            &plan.name,
                            "convolution name",
                        )?;
                    }
                    bytes = add_bytes(
                        bytes,
                        allocated_bytes(
                            plan.matrix_slots.capacity(),
                            size_of::<MatrixPlan>(),
                            "bytes",
                            &plan.name,
                            "matrix slots",
                        )?,
                        &plan.name,
                        "matrix slots",
                    )?;
                    for slot in &plan.matrix_slots {
                        bytes = add_bytes(
                            bytes,
                            allocated_bytes(
                                slot.name.len(),
                                1,
                                "bytes",
                                &plan.name,
                                "matrix name",
                            )?,
                            &plan.name,
                            "matrix name",
                        )?;
                    }
                }
            }
        }
        Ok(bytes)
    }
}

fn empty_layers(plan: &StatePlan) -> Result<Vec<LayerState>, LlmError> {
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(plan.layers().len())
        .map_err(|_| error("construct", "sequence", "layers", "allocation failed"))?;
    for layer in plan.layers() {
        let state = match layer {
            LayerPlan::Stateless { .. } => LayerState::Stateless,
            LayerPlan::Kv(plan) => {
                let mut keys = Vec::new();
                let mut values = Vec::new();
                keys.try_reserve_exact(plan.kv_heads).map_err(|_| {
                    error("construct", &plan.name, "key heads", "allocation failed")
                })?;
                values.try_reserve_exact(plan.kv_heads).map_err(|_| {
                    error("construct", &plan.name, "value heads", "allocation failed")
                })?;
                for _ in 0..plan.kv_heads {
                    let mut key = VecDeque::new();
                    let mut value = VecDeque::new();
                    key.try_reserve(plan.context_capacity)
                        .map_err(|_| error("construct", &plan.name, "key", "allocation failed"))?;
                    value.try_reserve(plan.context_capacity).map_err(|_| {
                        error("construct", &plan.name, "value", "allocation failed")
                    })?;
                    keys.push(key);
                    values.push(value);
                }
                LayerState::Kv(KvState { keys, values })
            }
            LayerPlan::Recurrent(plan) => {
                let convolution_len = plan
                    .convolution
                    .as_ref()
                    .map_or(0, |convolution| convolution.channels * convolution.history);
                let convolution =
                    zero_vec(convolution_len, "construct", &plan.name, "convolution")?;
                let mut matrices = Vec::new();
                matrices
                    .try_reserve_exact(plan.matrix_slots.len())
                    .map_err(|_| error("construct", &plan.name, "matrices", "allocation failed"))?;
                for slot in &plan.matrix_slots {
                    let len = checked_mul(slot.rows, slot.cols, "construct", &plan.name, "matrix")?;
                    matrices.push(zero_vec(len, "construct", &plan.name, "matrix")?);
                }
                let mut convolution_history = Vec::new();
                convolution_history.try_reserve_exact(1).map_err(|_| {
                    error(
                        "construct",
                        &plan.name,
                        "convolution history",
                        "allocation failed",
                    )
                })?;
                convolution_history.push(convolution.clone());
                let mut matrix_history = Vec::new();
                matrix_history.try_reserve_exact(1).map_err(|_| {
                    error(
                        "construct",
                        &plan.name,
                        "matrix history",
                        "allocation failed",
                    )
                })?;
                matrix_history.push(matrices.clone());
                LayerState::Recurrent(RecurrentState {
                    convolution,
                    matrices,
                    convolution_history,
                    matrix_history,
                })
            }
        };
        layers.push(state);
    }
    Ok(layers)
}

fn validate_recurrent_update(
    values: &LayerUpdate,
    plan: &RecurrentPlan,
    op: &str,
) -> Result<(), LlmError> {
    let name = plan.name.as_str();
    let convolution_len = match &plan.convolution {
        Some(convolution) => checked_mul(
            convolution.channels,
            convolution.history,
            op,
            name,
            "convolution",
        )?,
        None => 0,
    };
    require_values(
        &values.convolution,
        convolution_len,
        op,
        name,
        "convolution",
    )?;
    if values.matrices.len() != plan.matrix_slots.len() {
        return Err(error(
            op,
            name,
            "matrices",
            &format!(
                "slot count {} must be {}",
                values.matrices.len(),
                plan.matrix_slots.len()
            ),
        ));
    }
    for (slot, matrix) in plan.matrix_slots.iter().zip(&values.matrices) {
        let len = checked_mul(slot.rows, slot.cols, op, name, &slot.name)?;
        require_values(matrix, len, op, name, &slot.name)?;
    }
    Ok(())
}

fn require_empty(values: &LayerUpdate, op: &str, layer: &str) -> Result<(), LlmError> {
    if values.key.is_empty()
        && values.value.is_empty()
        && values.convolution.is_empty()
        && values.matrices.is_empty()
    {
        Ok(())
    } else {
        Err(error(
            op,
            layer,
            "update",
            "stateless layer requires empty state",
        ))
    }
}

fn require_empty_kv(values: &LayerUpdate, op: &str, layer: &str) -> Result<(), LlmError> {
    if values.key.is_empty() && values.value.is_empty() {
        Ok(())
    } else {
        Err(error(
            op,
            layer,
            "key/value",
            "recurrent layer requires empty KV state",
        ))
    }
}

fn require_empty_recurrent(values: &LayerUpdate, op: &str, layer: &str) -> Result<(), LlmError> {
    if values.convolution.is_empty() && values.matrices.is_empty() {
        Ok(())
    } else {
        Err(error(
            op,
            layer,
            "recurrent state",
            "KV layer requires empty recurrent state",
        ))
    }
}

fn require_values(
    values: &[f32],
    expected: usize,
    op: &str,
    layer: &str,
    field: &str,
) -> Result<(), LlmError> {
    if values.len() != expected {
        return Err(error(
            op,
            layer,
            field,
            &format!("length {} must be {expected}", values.len()),
        ));
    }
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(error(op, layer, field, "values must be finite"))
    }
}

fn zero_vec(len: usize, op: &str, layer: &str, field: &str) -> Result<Vec<f32>, LlmError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| error(op, layer, field, "allocation failed"))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn check_cancel_boundary(
    start: usize,
    len: usize,
    committed: usize,
    op: &str,
    layer: &str,
) -> Result<(), LlmError> {
    nonzero(len, op, layer, "query rows")?;
    let end = checked_add(start, len, op, layer, "query range")?;
    if end > committed {
        return Err(error(
            op,
            layer,
            "query range",
            &format!("{start}..{end} exceeds committed {committed}"),
        ));
    }
    Ok(())
}

/// Check the shared request token at a runtime boundary.
pub fn check_cancel(cancel: &RequestCancel) -> Result<(), LlmError> {
    if cancel.is_cancelled() {
        Err(LlmError::Cancelled)
    } else {
        Ok(())
    }
}

fn allocated_bytes(
    count: usize,
    size: usize,
    op: &str,
    layer: &str,
    field: &str,
) -> Result<u64, LlmError> {
    let count = u64::try_from(count).map_err(|_| error(op, layer, field, "size exceeds u64"))?;
    let size =
        u64::try_from(size).map_err(|_| error(op, layer, field, "element size exceeds u64"))?;
    count
        .checked_mul(size)
        .ok_or_else(|| error(op, layer, field, "byte count overflow"))
}

fn add_bytes(total: u64, value: u64, layer: &str, field: &str) -> Result<u64, LlmError> {
    total
        .checked_add(value)
        .ok_or_else(|| error("bytes", layer, field, "byte count overflow"))
}
