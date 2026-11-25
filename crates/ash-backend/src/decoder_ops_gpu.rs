//! Vulkan implementations of the portable decoder primitives.
//!
//! QK norm deliberately reuses the RMSNorm pipeline by flattening
//! `[rows, heads, head_dim]` to `[rows * heads, head_dim]`.

// size-allow: 420
use crate::{
    VkError,
    compute::{
        GpuBuffer, Kernel, RMS_NORM_SPIRV, ROPE_SPIRV, SHORT_CONV_SPIRV, SWIGLU_SPIRV, f32_bytes,
        f32_from_bytes, spirv_words,
    },
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

const LOCAL_SIZE: usize = 64;

fn invalid(op: &str, field: &str, detail: &str) -> VkError {
    VkError::Device(format!("{op}: {field} {detail}"))
}

fn nonzero(value: usize, op: &str, field: &str) -> Result<(), VkError> {
    if value == 0 {
        Err(invalid(op, field, "must be non-zero"))
    } else {
        Ok(())
    }
}

fn checked_mul(a: usize, b: usize, op: &str, field: &str) -> Result<usize, VkError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid(op, field, "element count overflow"))
}

fn require_len(actual: usize, expected: usize, op: &str, field: &str) -> Result<(), VkError> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid(
            op,
            field,
            &format!("length {actual} must be {expected}"),
        ))
    }
}

fn require_finite(values: &[f32], op: &str, field: &str) -> Result<(), VkError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid(op, field, "must be finite"))
    }
}

fn require_positive(value: f32, op: &str, field: &str) -> Result<(), VkError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(op, field, "must be finite and positive"))
    }
}

fn u32_dim(value: usize, op: &str, field: &str) -> Result<u32, VkError> {
    u32::try_from(value).map_err(|_| invalid(op, field, "exceeds u32 range"))
}

fn element_count(len: usize, op: &str) -> Result<u32, VkError> {
    u32::try_from(len).map_err(|_| invalid(op, "element count", "exceeds u32 range"))
}

fn byte_len(len: usize, op: &str) -> Result<usize, VkError> {
    len.checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid(op, "byte length", "overflow"))
}

fn workgroups(count: usize, op: &str) -> Result<u32, VkError> {
    u32::try_from(count.div_ceil(LOCAL_SIZE))
        .map_err(|_| invalid(op, "workgroup count", "exceeds u32 range"))
}

fn shape_bytes<const N: usize>(dims: [u32; N]) -> Vec<u8> {
    dims.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn admit(
    kernel: &Kernel,
    op: &str,
    workgroup_count: u32,
    ranges: [usize; 4],
) -> Result<[u64; 4], VkError> {
    let converted = ranges.map(|len| u64::try_from(len).ok());
    let [Some(a), Some(b), Some(c), Some(d)] = converted else {
        return Err(invalid(op, "buffer size", "exceeds u64 range"));
    };
    let sizes = [a, b, c, d];
    let storage_range = *sizes.iter().max().expect("four buffer ranges");
    kernel.device.info.admit(&CapacityRequirements {
        workgroup_count: [workgroup_count, 1, 1],
        storage_range,
        ..CapacityRequirements::default()
    })?;
    Ok(sizes)
}

fn validate_rms(
    values: &[f32],
    rows: usize,
    cols: usize,
    gain: &[f32],
    eps: f32,
    op: &str,
) -> Result<usize, VkError> {
    nonzero(rows, op, "rows")?;
    nonzero(cols, op, "cols")?;
    let len = checked_mul(rows, cols, op, "rows * cols")?;
    require_len(values.len(), len, op, "values")?;
    require_len(gain.len(), cols, op, "gain")?;
    require_positive(eps, op, "eps")?;
    require_finite(values, op, "values")?;
    require_finite(gain, op, "gain")?;
    element_count(len, op)?;
    Ok(len)
}

fn validate_qk(
    values: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    gain: &[f32],
    eps: f32,
) -> Result<(usize, usize), VkError> {
    nonzero(rows, "qk_norm", "rows")?;
    nonzero(heads, "qk_norm", "heads")?;
    nonzero(head_dim, "qk_norm", "head_dim")?;
    let per_row = checked_mul(heads, head_dim, "qk_norm", "heads * head_dim")?;
    let len = checked_mul(rows, per_row, "qk_norm", "rows * heads * head_dim")?;
    require_len(values.len(), len, "qk_norm", "values")?;
    require_len(gain.len(), head_dim, "qk_norm", "gain")?;
    require_positive(eps, "qk_norm", "eps")?;
    require_finite(values, "qk_norm", "values")?;
    require_finite(gain, "qk_norm", "gain")?;
    element_count(len, "qk_norm")?;
    Ok((len, checked_mul(rows, heads, "qk_norm", "rows * heads")?))
}

fn validate_rope(
    values: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    position: u64,
    rotary_dim: usize,
    theta: f32,
) -> Result<(usize, Vec<f32>), VkError> {
    nonzero(rows, "rope", "rows")?;
    nonzero(heads, "rope", "heads")?;
    nonzero(head_dim, "rope", "head_dim")?;
    let per_row = checked_mul(heads, head_dim, "rope", "heads * head_dim")?;
    let len = checked_mul(rows, per_row, "rope", "rows * heads * head_dim")?;
    require_len(values.len(), len, "rope", "values")?;
    if rotary_dim == 0 || rotary_dim % 2 != 0 {
        return Err(invalid("rope", "rotary_dim", "must be even and non-zero"));
    }
    if rotary_dim > head_dim {
        return Err(invalid("rope", "rotary_dim", "must not exceed head_dim"));
    }
    let position =
        u32::try_from(position).map_err(|_| invalid("rope", "position", "exceeds u32 range"))?;
    require_positive(theta, "rope", "theta")?;
    require_finite(values, "rope", "values")?;
    let trig = rope_trig(rotary_dim, position, theta)?;
    element_count(len, "rope")?;
    Ok((len, trig))
}

fn rope_trig(rotary_dim: usize, position: u32, theta: f32) -> Result<Vec<f32>, VkError> {
    let frequencies: Vec<f32> = (0..rotary_dim / 2)
        .map(|i| theta.powf(-2.0 * i as f32 / rotary_dim as f32))
        .collect();
    if !frequencies.iter().all(|frequency| frequency.is_finite()) {
        return Err(invalid("rope", "rotary frequencies", "must be finite"));
    }
    #[allow(clippy::cast_precision_loss)] // u32 positions are exact for RoPE angles.
    let position = position as f32;
    let mut trig = Vec::with_capacity(frequencies.len() * 2);
    for frequency in frequencies {
        let (sin, cos) = (position * frequency).sin_cos();
        if !sin.is_finite() || !cos.is_finite() {
            return Err(invalid("rope", "rotary angles", "must be finite"));
        }
        trig.push(cos);
        trig.push(sin);
    }
    Ok(trig)
}

fn validate_swiglu(gate: &[f32], up: &[f32]) -> Result<usize, VkError> {
    nonzero(gate.len(), "swiglu", "gate")?;
    require_len(up.len(), gate.len(), "swiglu", "up")?;
    require_finite(gate, "swiglu", "gate")?;
    require_finite(up, "swiglu", "up")?;
    element_count(gate.len(), "swiglu")?;
    Ok(gate.len())
}

fn validate_short_conv(
    values: &[f32],
    weights: &[f32],
    rows: usize,
    channels: usize,
    time: usize,
    kernel: usize,
) -> Result<usize, VkError> {
    nonzero(rows, "short_conv", "rows")?;
    nonzero(channels, "short_conv", "channels")?;
    nonzero(time, "short_conv", "time")?;
    nonzero(kernel, "short_conv", "kernel")?;
    let per_row = checked_mul(channels, time, "short_conv", "channels * time")?;
    let len = checked_mul(rows, per_row, "short_conv", "rows * channels * time")?;
    let weight_len = checked_mul(channels, kernel, "short_conv", "channels * kernel")?;
    let padded = time
        .checked_add(kernel)
        .ok_or_else(|| invalid("short_conv", "time + kernel", "index overflow"))?;
    u32::try_from(padded)
        .map_err(|_| invalid("short_conv", "time + kernel", "exceeds u32 range"))?;
    require_len(values.len(), len, "short_conv", "values")?;
    require_len(weights.len(), weight_len, "short_conv", "weights")?;
    require_finite(values, "short_conv", "values")?;
    require_finite(weights, "short_conv", "weights")?;
    element_count(len, "short_conv")?;
    element_count(weight_len, "short_conv")?;
    Ok(len)
}

/// Reusable RMSNorm pipeline; also dispatches per-head QK norm.
pub struct RmsNorm {
    kernel: Kernel,
}

impl RmsNorm {
    /// Create the shader module, descriptor layout and compute pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(RMS_NORM_SPIRV))?,
        })
    }

    /// Compute `x * rsqrt(mean(x^2) + eps) * gain` independently per row.
    pub fn run(
        &self,
        values: &[f32],
        rows: usize,
        cols: usize,
        gain: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, VkError> {
        let len = validate_rms(values, rows, cols, gain, eps, "rms_norm")?;
        self.dispatch_rms(values, len, rows, cols, gain, eps, "rms_norm")
    }

    /// Apply RMSNorm independently to every `[row, head]` vector.
    pub fn qk_norm(
        &self,
        values: &[f32],
        rows: usize,
        heads: usize,
        head_dim: usize,
        gain: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, VkError> {
        let (len, flat_rows) = validate_qk(values, rows, heads, head_dim, gain, eps)?;
        self.dispatch_rms(values, len, flat_rows, head_dim, gain, eps, "qk_norm")
    }

    #[allow(clippy::too_many_arguments)] // shared validated RMS/QK dispatch
    fn dispatch_rms(
        &self,
        values: &[f32],
        len: usize,
        rows: usize,
        cols: usize,
        gain: &[f32],
        eps: f32,
        op: &str,
    ) -> Result<Vec<f32>, VkError> {
        let groups = workgroups(rows, op)?;
        let bytes = byte_len(len, op)?;
        let gain_bytes = byte_len(gain.len(), op)?;
        let shape = shape_bytes([
            u32_dim(rows, op, "rows")?,
            u32_dim(cols, op, "cols")?,
            eps.to_bits(),
        ]);
        let sizes = admit(
            &self.kernel,
            op,
            groups,
            [bytes, gain_bytes, bytes, shape.len()],
        )?;
        let input = GpuBuffer::new(&self.kernel.device, &f32_bytes(values))?;
        let gain_buffer = GpuBuffer::new(&self.kernel.device, &f32_bytes(gain))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; bytes])?;
        let params = GpuBuffer::new(&self.kernel.device, &shape)?;
        let result = self.kernel.dispatch(
            &[input, gain_buffer, output, params],
            sizes,
            2,
            bytes,
            [groups, 1, 1],
        )?;
        Ok(f32_from_bytes(&result))
    }
}

/// Reusable half-split rotary position embedding pipeline.
pub struct Rope {
    kernel: Kernel,
}

impl Rope {
    /// Create the shader module, descriptor layout and compute pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(ROPE_SPIRV))?,
        })
    }

    /// Rotate `[rows, heads, head_dim]` half-split pairs and preserve the tail.
    #[allow(clippy::too_many_arguments)] // full shape plus rotary parameters
    pub fn run(
        &self,
        values: &[f32],
        rows: usize,
        heads: usize,
        head_dim: usize,
        position: u64,
        rotary_dim: usize,
        theta: f32,
    ) -> Result<Vec<f32>, VkError> {
        let (len, trig) =
            validate_rope(values, rows, heads, head_dim, position, rotary_dim, theta)?;
        let count = checked_mul(rows, heads, "rope", "rows * heads")?;
        let groups = workgroups(count, "rope")?;
        let bytes = byte_len(len, "rope")?;
        let trig_bytes = byte_len(trig.len(), "rope")?;
        let position = u32::try_from(position)
            .map_err(|_| invalid("rope", "position", "exceeds u32 range"))?;
        let shape = shape_bytes([
            u32_dim(rows, "rope", "rows")?,
            u32_dim(heads, "rope", "heads")?,
            u32_dim(head_dim, "rope", "head_dim")?,
            position,
            u32_dim(rotary_dim, "rope", "rotary_dim")?,
        ]);
        let sizes = admit(
            &self.kernel,
            "rope",
            groups,
            [bytes, trig_bytes, bytes, shape.len()],
        )?;
        let input = GpuBuffer::new(&self.kernel.device, &f32_bytes(values))?;
        let trig_buffer = GpuBuffer::new(&self.kernel.device, &f32_bytes(&trig))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; bytes])?;
        let params = GpuBuffer::new(&self.kernel.device, &shape)?;
        let result = self.kernel.dispatch(
            &[input, trig_buffer, output, params],
            sizes,
            2,
            bytes,
            [groups, 1, 1],
        )?;
        Ok(f32_from_bytes(&result))
    }
}

/// Reusable elementwise SwiGLU pipeline.
pub struct SwiGlu {
    kernel: Kernel,
}

impl SwiGlu {
    /// Create the shader module, descriptor layout and compute pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(SWIGLU_SPIRV))?,
        })
    }

    /// Compute `silu(gate[i]) * up[i]` for equal-length non-empty vectors.
    pub fn run(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, VkError> {
        let len = validate_swiglu(gate, up)?;
        let groups = workgroups(len, "swiglu")?;
        let bytes = byte_len(len, "swiglu")?;
        let shape = shape_bytes([element_count(len, "swiglu")?]);
        let sizes = admit(
            &self.kernel,
            "swiglu",
            groups,
            [bytes, bytes, bytes, shape.len()],
        )?;
        let gate_buffer = GpuBuffer::new(&self.kernel.device, &f32_bytes(gate))?;
        let up_buffer = GpuBuffer::new(&self.kernel.device, &f32_bytes(up))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; bytes])?;
        let params = GpuBuffer::new(&self.kernel.device, &shape)?;
        let result = self.kernel.dispatch(
            &[gate_buffer, up_buffer, output, params],
            sizes,
            2,
            bytes,
            [groups, 1, 1],
        )?;
        Ok(f32_from_bytes(&result))
    }
}

/// Reusable one-shot depthwise causal convolution pipeline.
pub struct ShortConv {
    kernel: Kernel,
}

impl ShortConv {
    /// Create the shader module, descriptor layout and compute pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(SHORT_CONV_SPIRV))?,
        })
    }

    /// Convolve `[rows, channels, time]` with `[channels, kernel]` weights.
    pub fn run(
        &self,
        values: &[f32],
        weights: &[f32],
        rows: usize,
        channels: usize,
        time: usize,
        kernel: usize,
    ) -> Result<Vec<f32>, VkError> {
        let len = validate_short_conv(values, weights, rows, channels, time, kernel)?;
        let groups = workgroups(len, "short_conv")?;
        let bytes = byte_len(len, "short_conv")?;
        let weight_len = checked_mul(channels, kernel, "short_conv", "channels * kernel")?;
        let weight_bytes = byte_len(weight_len, "short_conv")?;
        let shape = shape_bytes([
            u32_dim(rows, "short_conv", "rows")?,
            u32_dim(channels, "short_conv", "channels")?,
            u32_dim(time, "short_conv", "time")?,
            u32_dim(kernel, "short_conv", "kernel")?,
        ]);
        let sizes = admit(
            &self.kernel,
            "short_conv",
            groups,
            [bytes, weight_bytes, bytes, shape.len()],
        )?;
        let input = GpuBuffer::new(&self.kernel.device, &f32_bytes(values))?;
        let weight_buffer = GpuBuffer::new(&self.kernel.device, &f32_bytes(weights))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; bytes])?;
        let params = GpuBuffer::new(&self.kernel.device, &shape)?;
        let result = self.kernel.dispatch(
            &[input, weight_buffer, output, params],
            sizes,
            2,
            bytes,
            [groups, 1, 1],
        )?;
        Ok(f32_from_bytes(&result))
    }
}

#[cfg(test)]
#[path = "tests/decoder_ops_gpu.rs"]
mod tests;
