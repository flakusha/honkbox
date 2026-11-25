//! Additive causal, sliding-window and packed-segment attention masks.

use super::plan::{KvPlan, checked_mul, error, nonzero, u32_shape};
use honkbox_llm_worker::LlmError;

/// Absolute position within one packed segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenPosition {
    /// Packed segment identifier. Equal identifiers share causal visibility.
    pub segment: u32,
    /// Zero-based position inside the segment, including any plan offset.
    pub position: u64,
}

/// Visibility applied to one query/key position matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    /// All earlier keys in the same segment.
    Causal,
    /// Earlier keys in the same segment within `window` tokens.
    Sliding {
        /// Non-zero inclusive lookback distance (`query - key < window`).
        window: usize,
    },
    /// Block-diagonal packed visibility with an optional sliding window.
    Packed {
        /// `None` permits the whole segment; `Some` bounds lookback.
        window: Option<usize>,
    },
}

/// Row-major additive mask plus the positions that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct AdditiveMask {
    /// `0.0` allows a key; `-INFINITY` blocks it.
    pub values: Vec<f32>,
    /// Query positions in row order.
    pub queries: Vec<TokenPosition>,
    /// Retained key positions in column order.
    pub keys: Vec<TokenPosition>,
}

/// Build and validate a mask before attention dispatch.
pub fn build_mask(
    plan: &KvPlan,
    queries: &[TokenPosition],
    keys: &[TokenPosition],
    mode: MaskMode,
) -> Result<AdditiveMask, LlmError> {
    nonzero(queries.len(), "mask", &plan.name, "query rows")?;
    u32_shape(queries.len(), "mask", &plan.name, "query rows")?;
    u32_shape(keys.len(), "mask", &plan.name, "key rows")?;
    let window = match mode {
        MaskMode::Causal => None,
        MaskMode::Sliding { window } => Some(window),
        MaskMode::Packed { window } => window,
    };
    if let Some(window) = window {
        nonzero(window, "mask", &plan.name, "window")?;
        u32_shape(window, "mask", &plan.name, "window")?;
        if window > plan.context_capacity {
            return Err(error(
                "mask",
                &plan.name,
                "window",
                "must not exceed context_capacity",
            ));
        }
    }
    let elements = checked_mul(queries.len(), keys.len(), "mask", &plan.name, "elements")?;
    u32_shape(elements, "mask", &plan.name, "elements")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| error("mask", &plan.name, "values", "allocation failed"))?;
    for (row, query) in queries.iter().enumerate() {
        let mut visible = false;
        for key in keys {
            let same_segment = query.segment == key.segment;
            let causal = same_segment && query.position >= key.position;
            let in_window = window.is_none_or(|window| {
                query.position - key.position < u64::try_from(window).expect("u32 window")
            });
            if causal && in_window {
                values.push(0.0);
                visible = true;
            } else {
                values.push(f32::NEG_INFINITY);
            }
        }
        if !visible {
            return Err(error(
                "mask",
                &plan.name,
                &format!("query row {row}"),
                "has no visible key (segment/position/window malformed or all masked)",
            ));
        }
    }
    Ok(AdditiveMask {
        values,
        queries: queries.to_vec(),
        keys: keys.to_vec(),
    })
}
