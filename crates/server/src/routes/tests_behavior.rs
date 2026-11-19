//! Behavior tests for image cancellation, admission control, and recognition.

use super::image_ops_exec::image_op_pre_admission_join_failure;
use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A backend that ignores the cancel flag must not answer 200 with a
/// full image after the token was marked.
struct CancelIgnoringPipeline;

impl honkbox_sd_core::SdPipeline for CancelIgnoringPipeline {
    fn generate(
        &self,
        _req: &honkbox_sd_core::Txt2ImgRequest,
        _cancel: &AtomicBool,
    ) -> Result<honkbox_sd_core::GeneratedImage, honkbox_sd_core::SdError> {
        Ok(honkbox_sd_core::GeneratedImage {
            width: 1,
            height: 1,
            rgba: vec![1, 2, 3, 255],
        })
    }

    fn backend_id(&self) -> &'static str {
        "cancel-ignoring-test"
    }
}

fn one_pixel_request() -> honkbox_sd_core::Txt2ImgRequest {
    honkbox_sd_core::Txt2ImgRequest {
        prompt: String::new(),
        seed: 1,
        steps: 1,
        width: 1,
        height: 1,
        model_slug: String::new(),
        clip: None,
        vae: None,
        sampler: None,
        scheduler: None,
        cfg_scale: None,
        negative_prompt: None,
        loras: None,
        init_image: None,
        mask_image: None,
        strength: None,
        hires: None,
        control_net: None,
        control_image: None,
        controlnet_strength: None,
        control_type: None,
    }
}

#[tokio::test]
async fn cancelled_token_beats_a_backend_that_returns_ok() {
    let token = tokio_util::sync::CancellationToken::new();
    token.cancel();
    let outcome = generate_image_bytes(
        Arc::new(CancelIgnoringPipeline),
        one_pixel_request(),
        honkbox_sd_core::OutputFormat::Png,
        None,
        new_image_cancel_flag(),
        token,
    )
    .await;
    assert!(matches!(outcome, Err(GenerateFailure::Cancelled)));
}

#[test]
fn each_image_gets_a_fresh_cancel_flag() {
    let first = new_image_cancel_flag();
    let second = new_image_cancel_flag();
    first.store(true, Ordering::Release);
    assert!(!second.load(Ordering::Acquire));
    assert!(!Arc::ptr_eq(&first, &second));
}

#[test]
fn image_cancel_messages_name_the_phase() {
    let before = image_cancel_before_start("job_1", 2, 3, 1);
    let during = image_cancel_mid_generation("job_1", 2, 3, 1);
    assert!(before.contains("before image 2 of 3 started"), "{before}");
    assert!(!before.contains("while generating"), "{before}");
    assert!(during.contains("while generating image 2 of 3"), "{during}");
    assert!(!during.contains("before image"), "{during}");
    assert!(before.contains("job_1") && during.contains("job_1"));
}

#[test]
fn admission_rejection_registers_no_job_and_has_no_job_header() {
    let registry = crate::jobs_registry::Registry::default();
    let mut events = registry.subscribe();
    let rejection = (StatusCode::TOO_MANY_REQUESTS, "slot busy".to_owned());
    let result = admit_then_register(
        &registry,
        Err(rejection.clone()),
        crate::jobs_registry::JobKind::Chat,
        Some("m".to_owned()),
        true,
    );
    let error = result.expect_err("admission rejected");
    assert_eq!(error, rejection);
    assert!(registry.list().is_empty());
    assert!(events.try_recv().is_err(), "no job_registered on a 429");
    let response: Response = (error.0, error.1).into_response();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(response.headers().get(JOB_ID_HEADER).is_none());
}

#[test]
fn pre_admission_join_failure_stamps_a_job_id() {
    let settled = image_op_pre_admission_join_failure("worker died".to_owned());
    match settled.response {
        Err((status, message, code)) => {
            assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(message, "worker died");
            assert_eq!(code, None);
        }
        Ok(_) => panic!("join failure must be a 500"),
    }
    let id = settled.job_id.expect("id stamped");
    assert!(id.starts_with("job_"), "{id}");
}

#[test]
fn recognition_501_points_at_the_top_level_capability() {
    let message = recognition_501_message("unsupported kind");
    assert!(
        message.contains("GET /v1/capabilities recognition"),
        "{message}"
    );
    assert!(!message.contains("features.recognition"), "{message}");
}
