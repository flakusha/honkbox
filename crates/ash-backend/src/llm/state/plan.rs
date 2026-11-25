//! Validated, family-independent descriptions of per-layer sequence state.

use honkbox_llm_worker::LlmError;

/// One named layer in a [`StatePlan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerPlan {
    /// A layer that contributes no persistent sequence state.
    Stateless {
        /// Stable diagnostic name, unique within the plan.
        name: String,
    },
    /// Unique-head key/value storage and grouped-query mapping.
    Kv(KvPlan),
    /// Caller-updated convolution history and matrix-state slots.
    Recurrent(RecurrentPlan),
}

impl LayerPlan {
    /// The layer's stable diagnostic name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Stateless { name }
            | Self::Kv(KvPlan { name, .. })
            | Self::Recurrent(RecurrentPlan { name, .. }) => name,
        }
    }
}

/// Geometry for one KV layer. K/V are stored once per unique KV head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPlan {
    /// Stable diagnostic name.
    pub name: String,
    /// Number of query heads; must be divisible by `kv_heads`.
    pub query_heads: usize,
    /// Number of unique key/value heads.
    pub kv_heads: usize,
    /// Width of each query/key head.
    pub head_dim: usize,
    /// Width of each value head.
    pub value_dim: usize,
    /// Maximum retained tokens. Non-sliding layers reject growth beyond it.
    pub context_capacity: usize,
    /// Optional causal window; must not exceed `context_capacity`.
    pub sliding_window: Option<usize>,
    /// Stored element width. This f32 runtime supports exactly four bytes.
    pub element_bytes: usize,
}

impl KvPlan {
    /// Query heads sharing one unique KV head.
    #[must_use]
    pub fn group_size(&self) -> usize {
        self.query_heads / self.kv_heads
    }

    /// Map a query head to its unique KV head.
    #[must_use]
    pub fn kv_head(&self, query_head: usize) -> usize {
        query_head / self.group_size()
    }
}

/// Geometry for caller-updated convolution history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionPlan {
    /// Stable slot name, unique within this recurrent layer.
    pub name: String,
    /// Number of independent convolution channels.
    pub channels: usize,
    /// Number of prior channel values retained by the caller's equation.
    pub history: usize,
}

/// One dense matrix-state slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixPlan {
    /// Stable slot name, unique within this recurrent layer.
    pub name: String,
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub cols: usize,
}

/// Recurrent state geometry. The runtime owns values, not update equations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentPlan {
    /// Stable diagnostic name.
    pub name: String,
    /// Optional flattened `[channel, history]` convolution state.
    pub convolution: Option<ConvolutionPlan>,
    /// Flattened row-major matrix slots.
    pub matrix_slots: Vec<MatrixPlan>,
    /// Stored element width. This f32 runtime supports exactly four bytes.
    pub element_bytes: usize,
}

/// A validated layer plan. Construct with [`StatePlan::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatePlan {
    layers: Vec<LayerPlan>,
}

impl StatePlan {
    /// Validate every layer before any sequence allocates or dispatches.
    ///
    /// All dimensions and per-buffer element counts must fit `u32`, matching
    /// the decoder dispatch boundary. Recurrent history is replayed from exact
    /// caller-supplied snapshots; no family equations live here.
    pub fn new(layers: Vec<LayerPlan>) -> Result<Self, LlmError> {
        if layers.is_empty() {
            return Err(error("plan", "sequence", "layers", "must not be empty"));
        }
        u32_shape(layers.len(), "plan", "sequence", "layers")?;
        for (index, layer) in layers.iter().enumerate() {
            let name = layer.name();
            if name.trim().is_empty() {
                return Err(error("plan", name, "name", "must not be empty"));
            }
            if layers[..index].iter().any(|other| other.name() == name) {
                return Err(error("plan", name, "name", "must be unique"));
            }
            match layer {
                LayerPlan::Stateless { .. } => {}
                LayerPlan::Kv(plan) => validate_kv(plan)?,
                LayerPlan::Recurrent(plan) => validate_recurrent(plan)?,
            }
        }
        Ok(Self { layers })
    }

    /// Validated layers in dispatch order.
    #[must_use]
    pub fn layers(&self) -> &[LayerPlan] {
        &self.layers
    }

    /// Find a layer by its exact diagnostic name.
    pub fn layer(&self, name: &str) -> Result<&LayerPlan, LlmError> {
        self.layers
            .iter()
            .find(|layer| layer.name() == name)
            .ok_or_else(|| error("lookup", name, "layer", "is not in the plan"))
    }

    /// Find a KV layer, rejecting another layer kind by name.
    pub fn kv(&self, name: &str) -> Result<&KvPlan, LlmError> {
        match self.layer(name)? {
            LayerPlan::Kv(plan) => Ok(plan),
            _ => Err(error("lookup", name, "layer", "must be KV")),
        }
    }
}

pub(crate) fn error(op: &str, layer: &str, field: &str, detail: &str) -> LlmError {
    LlmError::Context(format!("state {op}: layer '{layer}': {field} {detail}"))
}

pub(crate) fn u32_shape(value: usize, op: &str, layer: &str, field: &str) -> Result<u32, LlmError> {
    u32::try_from(value).map_err(|_| error(op, layer, field, "exceeds u32 range"))
}

pub(crate) fn nonzero(value: usize, op: &str, layer: &str, field: &str) -> Result<(), LlmError> {
    if value == 0 {
        Err(error(op, layer, field, "must be non-zero"))
    } else {
        Ok(())
    }
}

pub(crate) fn checked_mul(
    left: usize,
    right: usize,
    op: &str,
    layer: &str,
    field: &str,
) -> Result<usize, LlmError> {
    left.checked_mul(right)
        .ok_or_else(|| error(op, layer, field, "element count overflow"))
}

pub(crate) fn checked_add(
    left: usize,
    right: usize,
    op: &str,
    layer: &str,
    field: &str,
) -> Result<usize, LlmError> {
    left.checked_add(right)
        .ok_or_else(|| error(op, layer, field, "index overflow"))
}

fn require_element_bytes(value: usize, op: &str, layer: &str) -> Result<(), LlmError> {
    if value == size_of::<f32>() {
        Ok(())
    } else {
        Err(error(
            op,
            layer,
            "element_bytes",
            "must be 4 for the f32 state runtime",
        ))
    }
}

fn validate_kv(plan: &KvPlan) -> Result<(), LlmError> {
    let name = plan.name.as_str();
    for (field, value) in [
        ("query_heads", plan.query_heads),
        ("kv_heads", plan.kv_heads),
        ("head_dim", plan.head_dim),
        ("value_dim", plan.value_dim),
        ("context_capacity", plan.context_capacity),
    ] {
        nonzero(value, "plan", name, field)?;
        u32_shape(value, "plan", name, field)?;
    }
    require_element_bytes(plan.element_bytes, "plan", name)?;
    if !plan.query_heads.is_multiple_of(plan.kv_heads) {
        return Err(error(
            "plan",
            name,
            "query_heads / kv_heads",
            "must be an integer GQA ratio",
        ));
    }
    if let Some(window) = plan.sliding_window {
        nonzero(window, "plan", name, "sliding_window")?;
        u32_shape(window, "plan", name, "sliding_window")?;
        if window > plan.context_capacity {
            return Err(error(
                "plan",
                name,
                "sliding_window",
                "must not exceed context_capacity",
            ));
        }
    }
    let heads = checked_mul(
        plan.kv_heads,
        plan.context_capacity,
        "plan",
        name,
        "KV tokens",
    )?;
    let keys = checked_mul(heads, plan.head_dim, "plan", name, "key elements")?;
    let values = checked_mul(heads, plan.value_dim, "plan", name, "value elements")?;
    u32_shape(keys, "plan", name, "key elements")?;
    u32_shape(values, "plan", name, "value elements")?;
    let per_token = checked_mul(
        plan.query_heads,
        plan.head_dim,
        "plan",
        name,
        "query elements per token",
    )?;
    u32_shape(per_token, "plan", name, "query elements per token")?;
    let mask = checked_mul(
        plan.context_capacity,
        plan.context_capacity,
        "plan",
        name,
        "mask elements",
    )?;
    u32_shape(mask, "plan", name, "mask elements")?;
    Ok(())
}

fn validate_recurrent(plan: &RecurrentPlan) -> Result<(), LlmError> {
    let name = plan.name.as_str();
    require_element_bytes(plan.element_bytes, "plan", name)?;
    if plan.convolution.is_none() && plan.matrix_slots.is_empty() {
        return Err(error(
            "plan",
            name,
            "slots",
            "must declare convolution history or a matrix slot",
        ));
    }
    let mut elements = 0;
    if let Some(convolution) = &plan.convolution {
        if convolution.name.trim().is_empty() {
            return Err(error("plan", name, "convolution.name", "must not be empty"));
        }
        nonzero(convolution.channels, "plan", name, "convolution.channels")?;
        nonzero(convolution.history, "plan", name, "convolution.history")?;
        u32_shape(convolution.channels, "plan", name, "convolution.channels")?;
        u32_shape(convolution.history, "plan", name, "convolution.history")?;
        elements = checked_mul(
            convolution.channels,
            convolution.history,
            "plan",
            name,
            "convolution elements",
        )?;
    }
    u32_shape(plan.matrix_slots.len(), "plan", name, "matrix_slots")?;
    for (index, slot) in plan.matrix_slots.iter().enumerate() {
        let field = format!("matrix_slots[{index}].{}", slot.name);
        if slot.name.trim().is_empty() {
            return Err(error("plan", name, &field, "name must not be empty"));
        }
        if plan.matrix_slots[..index]
            .iter()
            .any(|other| other.name == slot.name)
        {
            return Err(error("plan", name, &field, "name must be unique"));
        }
        nonzero(slot.rows, "plan", name, &format!("{field}.rows"))?;
        nonzero(slot.cols, "plan", name, &format!("{field}.cols"))?;
        u32_shape(slot.rows, "plan", name, &format!("{field}.rows"))?;
        u32_shape(slot.cols, "plan", name, &format!("{field}.cols"))?;
        let slot_elements = checked_mul(slot.rows, slot.cols, "plan", name, &field)?;
        elements = checked_add(elements, slot_elements, "plan", name, "recurrent elements")?;
    }
    u32_shape(elements, "plan", name, "recurrent elements")?;
    Ok(())
}
