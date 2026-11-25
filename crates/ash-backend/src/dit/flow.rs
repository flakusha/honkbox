//! Flow-sigma conditioning: Fourier embedding, shifted schedule, Euler step.
//!
//! ponytail: host-side f32 math; promote to kernels only if profiling shows it
//! dominates a step.

#![allow(clippy::cast_precision_loss)]

use crate::VkError;

use super::invalid;

/// Even-width half-cos/half-sin embedding of a sigma in `[0, 1]`.
pub fn sigma_embedding(sigma: f32, width: usize, theta: f32) -> Result<Vec<f32>, VkError> {
    if !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) {
        return Err(invalid("sigma", "must be finite in [0, 1]"));
    }
    if width == 0 || !width.is_multiple_of(2) {
        return Err(invalid("width", "must be even and non-zero"));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(invalid("theta", "must be finite and positive"));
    }

    let half = width / 2;
    let mut out = Vec::with_capacity(width);
    for i in 0..half {
        let angle = sigma * theta.powf(-2.0 * i as f32 / half as f32);
        out.push(angle.cos());
    }
    for i in 0..half {
        let angle = sigma * theta.powf(-2.0 * i as f32 / half as f32);
        out.push(angle.sin());
    }
    if out.iter().any(|value| !value.is_finite()) {
        return Err(invalid("embedding", "must be finite"));
    }
    Ok(out)
}

/// Descending sigma grid with `t / (t + shift * (1 - t))`, ending at zero.
pub fn shifted_schedule(steps: usize, shift: f32) -> Result<Vec<f32>, VkError> {
    if steps < 2 {
        return Err(invalid("steps", "must be at least 2"));
    }
    if !shift.is_finite() || shift <= 0.0 {
        return Err(invalid("shift", "must be finite and positive"));
    }
    if steps.checked_mul(std::mem::size_of::<f32>()).is_none() {
        return Err(invalid("steps", "allocation overflow"));
    }

    let denominator = (steps - 1) as f32;
    let mut out = Vec::with_capacity(steps);
    let mut previous = f32::INFINITY;
    for index in 0..steps {
        let t = 1.0 - index as f32 / denominator;
        let sigma = t / (t + shift * (1.0 - t));
        if !sigma.is_finite() || sigma > previous {
            return Err(invalid("sigma", "must be finite and descending"));
        }
        previous = sigma;
        out.push(sigma);
    }
    Ok(out)
}

/// Explicit Euler update with a descending flow sigma.
pub fn euler_step(
    latent: &[f32],
    velocity: &[f32],
    sigma: f32,
    sigma_next: f32,
) -> Result<Vec<f32>, VkError> {
    if latent.is_empty() {
        return Err(invalid("latent", "must be non-empty"));
    }
    if latent.len() != velocity.len() {
        return Err(invalid("velocity", "length must match latent"));
    }
    if let Some(index) = latent.iter().position(|value| !value.is_finite()) {
        return Err(invalid(
            "latent",
            &format!("element {index} must be finite"),
        ));
    }
    if let Some(index) = velocity.iter().position(|value| !value.is_finite()) {
        return Err(invalid(
            "velocity",
            &format!("element {index} must be finite"),
        ));
    }
    if !sigma.is_finite() {
        return Err(invalid("sigma", "must be finite"));
    }
    if !sigma_next.is_finite() {
        return Err(invalid("sigma_next", "must be finite"));
    }

    let delta = sigma_next - sigma;
    let mut out = Vec::with_capacity(latent.len());
    for (value, speed) in latent.iter().zip(velocity) {
        out.push(value + delta * speed);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_dit_substrate_sigma_embedding_half_split_and_finiteness() {
        let values = sigma_embedding(0.5, 6, 1.0).unwrap();
        let expected = [0.5_f32.cos(); 3].into_iter().chain([0.5_f32.sin(); 3]);
        for (got, want) in values.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        let swept = sigma_embedding(1.0, 8, 100.0).unwrap();
        assert_eq!(swept.len(), 8);
        assert!(swept.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn flow_dit_substrate_sigma_embedding_zero_sigma_shape() {
        let values = sigma_embedding(0.0, 6, 100.0).unwrap();
        assert_eq!(&values[..3], &[1.0, 1.0, 1.0]);
        assert_eq!(&values[3..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn flow_dit_substrate_sigma_embedding_rejects_bad_inputs_by_name() {
        let cases = [
            (f32::NAN, 4, 10.0, "sigma"),
            (0.5, 0, 10.0, "width"),
            (0.5, 3, 10.0, "width"),
            (0.5, 4, 0.0, "theta"),
            (0.5, 4, f32::INFINITY, "theta"),
        ];
        for (sigma, width, theta, field) in cases {
            let error = sigma_embedding(sigma, width, theta)
                .unwrap_err()
                .to_string();
            assert!(error.contains(field), "{error}");
        }
    }

    #[test]
    fn flow_dit_substrate_shifted_schedule_endpoints_and_descent() {
        let sigmas = shifted_schedule(5, 1.0).unwrap();
        assert_eq!(sigmas[0], 1.0);
        assert_eq!(sigmas[4], 0.0);
        for pair in sigmas.windows(2) {
            assert!(pair[0] > pair[1], "{pair:?}");
        }
        let expected = [1.0, 0.75, 0.5, 0.25, 0.0];
        for (got, want) in sigmas.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn flow_dit_substrate_shifted_schedule_shift_effect_and_rejections() {
        let unit = shifted_schedule(3, 1.0).unwrap();
        let shifted = shifted_schedule(3, 3.0).unwrap();
        assert!(shifted[1] < unit[1]);
        assert_eq!(shifted[0], 1.0);
        assert_eq!(shifted[2], 0.0);
        for (steps, shift, field) in [(1, 1.0, "steps"), (4, 0.0, "shift"), (4, f32::NAN, "shift")]
        {
            let error = shifted_schedule(steps, shift).unwrap_err().to_string();
            assert!(error.contains(field), "{error}");
        }
    }

    #[test]
    fn flow_dit_substrate_euler_step_numeric_result() {
        let next = euler_step(&[1.0, 2.0], &[0.5, -1.0], 0.2, 0.5).unwrap();
        for (got, want) in next.iter().zip([1.15_f32, 1.7]) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        let held = euler_step(&[1.0, 2.0], &[0.5, -1.0], 0.5, 0.5).unwrap();
        assert_eq!(held, vec![1.0, 2.0]);
    }

    #[test]
    fn flow_dit_substrate_euler_step_rejects_bad_inputs_by_name() {
        let error = euler_step(&[], &[], 0.0, 1.0).unwrap_err().to_string();
        assert!(error.contains("latent"), "{error}");
        let error = euler_step(&[1.0], &[1.0, 2.0], 0.0, 1.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("velocity"), "{error}");
        let error = euler_step(&[f32::NAN], &[1.0], 0.0, 1.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("latent"), "{error}");
        let error = euler_step(&[1.0], &[f32::INFINITY], 0.0, 1.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("velocity"), "{error}");
        let error = euler_step(&[1.0], &[1.0], f32::NAN, 1.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("sigma"), "{error}");
        let error = euler_step(&[1.0], &[1.0], 0.0, f32::NAN)
            .unwrap_err()
            .to_string();
        assert!(error.contains("sigma_next"), "{error}");
    }
}
