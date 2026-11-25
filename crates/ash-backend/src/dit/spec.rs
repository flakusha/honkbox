//! Validated geometry for one synthetic single-stream flow-DiT block.

use crate::VkError;

fn invalid(field: &str, detail: &str) -> VkError {
    VkError::Device(format!("flow_dit: {field} {detail}"))
}

/// Axes of the patch grid. Image grids use two axes; video grids use three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionAxes {
    /// Two-dimensional patch grid.
    Image {
        /// Row count.
        height: usize,
        /// Column count.
        width: usize,
    },
    /// Three-dimensional patch grid.
    Video {
        /// Frame count.
        frames: usize,
        /// Row count.
        height: usize,
        /// Column count.
        width: usize,
    },
}

impl PositionAxes {
    /// Axes in rotation order: image is `[row, column, 0]`, video adds frames.
    pub fn dims(self) -> [usize; 3] {
        match self {
            Self::Image { height, width } => [height, width, 1],
            Self::Video {
                frames,
                height,
                width,
            } => [frames, height, width],
        }
    }

    /// Number of patches in the grid, checked for overflow.
    pub fn product(self) -> Result<usize, VkError> {
        let [a, b, c] = self.dims();
        a.checked_mul(b)
            .and_then(|value| value.checked_mul(c))
            .ok_or_else(|| invalid("axes", "element count overflow"))
    }
}

/// Geometry of the shared synthetic `DiT` substrate. No family is named here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowDitSpec {
    /// Model width `W`.
    pub width: usize,
    /// Attention head count.
    pub heads: usize,
    /// Per-head query/key width.
    pub head_dim: usize,
    /// Feed-forward hidden width.
    pub ffn_width: usize,
    /// Flattened elements per patch.
    pub patch_volume: usize,
    /// Latent channel count.
    pub latent_channels: usize,
    /// Patch-grid axes.
    pub axes: PositionAxes,
}

impl FlowDitSpec {
    /// Reject zero, odd or inconsistent dimensions before any allocation.
    pub fn validate(&self) -> Result<(), VkError> {
        for (field, value) in [
            ("width", self.width),
            ("heads", self.heads),
            ("head_dim", self.head_dim),
            ("ffn_width", self.ffn_width),
            ("patch_volume", self.patch_volume),
            ("latent_channels", self.latent_channels),
        ] {
            if value == 0 {
                return Err(invalid(field, "must be non-zero"));
            }
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(invalid("head_dim", "must be even"));
        }
        let expected = self
            .heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid("width", "head product overflow"))?;
        if expected != self.width {
            return Err(invalid("width", "must equal heads * head_dim"));
        }
        for (field, value) in self.axes.dims().into_iter().enumerate() {
            if value == 0 {
                return Err(invalid("axes", &format!("axis {field} must be non-zero")));
            }
        }
        if self.axes.product()? != self.patch_volume {
            return Err(invalid("axes", "product must equal patch_volume"));
        }
        Ok(())
    }

    /// Token count implied by the patch grid.
    pub fn tokens(&self) -> Result<usize, VkError> {
        self.validate()?;
        self.axes.product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> FlowDitSpec {
        FlowDitSpec {
            width: 4,
            heads: 2,
            head_dim: 2,
            ffn_width: 4,
            patch_volume: 2,
            latent_channels: 1,
            axes: PositionAxes::Image {
                height: 1,
                width: 2,
            },
        }
    }

    #[test]
    fn flow_dit_substrate_spec_accepts_image_and_video_geometry() {
        assert!(spec().validate().is_ok());
        let mut video = spec();
        video.patch_volume = 6;
        video.axes = PositionAxes::Video {
            frames: 2,
            height: 1,
            width: 3,
        };
        assert!(video.validate().is_ok());
        assert_eq!(video.tokens().unwrap(), 6);
    }

    #[test]
    fn flow_dit_substrate_spec_rejects_bad_geometry_by_name() {
        let cases = [
            ("width", 0),
            ("head_dim", 3),
            ("ffn_width", 0),
            ("patch_volume", 3),
        ];
        for (field, value) in cases {
            let mut bad = spec();
            match field {
                "width" => bad.width = value,
                "head_dim" => {
                    bad.head_dim = value;
                    bad.width = value * bad.heads;
                }
                "ffn_width" => bad.ffn_width = value,
                _ => bad.patch_volume = value,
            }
            let error = bad.validate().unwrap_err().to_string();
            assert!(error.contains(field), "{error}");
        }
        let mut bad = spec();
        bad.axes = PositionAxes::Image {
            height: 1,
            width: 1,
        };
        assert!(bad.validate().unwrap_err().to_string().contains("axes"));
    }
}
