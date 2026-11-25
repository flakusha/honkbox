//! Centered patch-grid positions for the flow-DiT substrate.
//!
//! ponytail: host-side grid generation; add a shader only if profiling shows
//! position lookup on the GPU matters.

#![allow(clippy::cast_precision_loss)]

use crate::VkError;

use super::invalid;
use super::spec::PositionAxes;

/// Row-major positions in `[row, column, 0]` or video axis order.
pub fn position_grid(axes: PositionAxes) -> Result<Vec<[f32; 3]>, VkError> {
    let dims = axes.dims();
    for (axis, size) in dims.into_iter().enumerate() {
        if size == 0 {
            return Err(invalid("axes", &format!("axis {axis} must be non-zero")));
        }
    }
    let count = axes.product()?;
    let centers = dims.map(|size| (size - 1) as f32 / 2.0);

    let mut out = Vec::with_capacity(count);
    for first in 0..dims[0] {
        for second in 0..dims[1] {
            for third in 0..dims[2] {
                out.push([
                    first as f32 - centers[0],
                    second as f32 - centers[1],
                    third as f32 - centers[2],
                ]);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_dit_substrate_position_grid_centers_image_and_video() {
        let image = position_grid(PositionAxes::Image {
            height: 2,
            width: 3,
        })
        .unwrap();
        assert_eq!(
            image,
            vec![
                [-0.5, -1.0, 0.0],
                [-0.5, 0.0, 0.0],
                [-0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        );
        let video = position_grid(PositionAxes::Video {
            frames: 2,
            height: 1,
            width: 2,
        })
        .unwrap();
        assert_eq!(
            video,
            vec![
                [-0.5, 0.0, -0.5],
                [-0.5, 0.0, 0.5],
                [0.5, 0.0, -0.5],
                [0.5, 0.0, 0.5],
            ]
        );
    }

    #[test]
    fn flow_dit_substrate_position_grid_axis_independence() {
        let narrow = position_grid(PositionAxes::Image {
            height: 2,
            width: 2,
        })
        .unwrap();
        let wide = position_grid(PositionAxes::Image {
            height: 2,
            width: 3,
        })
        .unwrap();
        for (base, changed) in narrow.chunks(2).zip(wide.chunks(3)) {
            assert_eq!(base[0][0], changed[0][0]);
            assert_eq!(base[1][0], changed[1][0]);
            assert_ne!(base[0][1], changed[0][1]);
            assert_eq!(changed[0][2], 0.0);
        }
    }

    #[test]
    fn flow_dit_substrate_position_grid_contains_origin_for_odd_image() {
        let grid = position_grid(PositionAxes::Image {
            height: 3,
            width: 3,
        })
        .unwrap();
        assert!(grid.contains(&[0.0, 0.0, 0.0]));
    }

    #[test]
    fn flow_dit_substrate_position_grid_rejects_zero_and_overflow() {
        let zero = position_grid(PositionAxes::Image {
            height: 0,
            width: 1,
        })
        .unwrap_err()
        .to_string();
        assert!(zero.contains("axes"), "{zero}");
        let overflow = position_grid(PositionAxes::Image {
            height: usize::MAX,
            width: 2,
        })
        .unwrap_err()
        .to_string();
        assert!(overflow.contains("overflow"), "{overflow}");
    }
}
