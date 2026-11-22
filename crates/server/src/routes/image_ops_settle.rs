//! Image-op job settlement: map blocking-pool outcomes onto units or named
//! failures, and execute one command end to end.

use std::time::Instant;

use axum::http::StatusCode;
use base64::Engine as _;
use honkbox_api_types::{ImageOpOutput, ImageOpRequest, ImageOpResponse};

use super::AppConfig;
use super::chat_prep::admit_then_register;
use super::image_ops_dedup::{ImageOpSettled, op_label, sha256_hex};
use super::image_ops_exec::{
    ImageOpWork, ImageOpWorkError, admit_op_inputs, image_op_pre_admission_join_failure,
    run_image_op_units, settled_without_job,
};

/// Map the blocking-pool work outcome onto units or a settleable failure:
/// cancel → 409 between units, bad input → named 400, work/join failure →
/// 500. Every failure stamps the job id so the replay keeps `X-Job-Id`
/// (boxed: `ImageOpSettled` carries the full response and is big).
pub(super) fn settle_image_op_work(
    job_id: &str,
    total_inputs: usize,
    work: Result<Result<ImageOpWork, ImageOpWorkError>, tokio::task::JoinError>,
) -> Result<ImageOpWork, Box<ImageOpSettled>> {
    match work {
        Ok(Ok(pair)) => Ok(pair),
        Ok(Err(ImageOpWorkError::Cancelled { completed })) => Err(Box::new(ImageOpSettled {
            response: Err((
                StatusCode::CONFLICT,
                format!(
                    "job {job_id} cancelled before completion ({completed} of {total_inputs} input(s) processed)"
                ),
                Some("job_cancelled".to_owned()),
            )),
            job_id: Some(job_id.to_owned()),
        })),
        Ok(Err(ImageOpWorkError::Bad { message, code })) => Err(Box::new(ImageOpSettled {
            response: Err((StatusCode::BAD_REQUEST, message, Some(code.to_owned()))),
            job_id: Some(job_id.to_owned()),
        })),
        Ok(Err(ImageOpWorkError::Failed(message))) => Err(Box::new(ImageOpSettled {
            response: Err((StatusCode::INTERNAL_SERVER_ERROR, message, None)),
            job_id: Some(job_id.to_owned()),
        })),
        Err(join) => Err(Box::new(ImageOpSettled {
            response: Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("image-op worker failed: {join}"),
                None,
            )),
            job_id: Some(job_id.to_owned()),
        })),
    }
}

/// Execute one image-op command: register the job, acquire CPU admission,
/// run the units on the blocking pool (cancel checked between inputs),
/// write the bundle (best-effort, like generation), and build the response.
/// NEVER returns without a settleable outcome — every failure path returns
/// the exact error the caller must replay.
#[allow(clippy::too_many_lines)] // one linear execution: admit, register, run, settle
pub(super) async fn execute_image_op(
    config: &AppConfig,
    req: &ImageOpRequest,
    command_sha256: [u8; 32],
    format: honkbox_sd_core::OutputFormat,
    jpeg_quality: Option<u8>,
) -> ImageOpSettled {
    let started = Instant::now();
    // Pre-admission: every input is resolved and safety-checked BEFORE a
    // job is registered, so a hostile container is a named 400 and the job
    // registry never sees it.
    let inputs = req.inputs.clone();
    let data_dir = config.data_dir.clone();
    let admitted =
        match tokio::task::spawn_blocking(move || admit_op_inputs(&inputs, data_dir.as_deref()))
            .await
        {
            Ok(Ok(admitted)) => admitted,
            Ok(Err(err)) => return settled_without_job(err),
            Err(join) => {
                return image_op_pre_admission_join_failure(format!(
                    "image-op worker failed: {join}"
                ));
            }
        };
    // Admission (TASK-image-ops-job-route): CPU-only work takes the
    // ledger's CPU-only bucket (`rag`, 0 GPU slots) — the fixed 9-kind
    // vocabulary has no image-ops kind, the same nearest-bucket mapping the
    // audio routes use with `llm`. Image ops therefore contend with rag
    // pipelines on CPU threads instead of bypassing the ledger entirely.
    // Admission first, registration second: a rejection carries no job id.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Rag).await,
        crate::jobs_registry::JobKind::ImageOps,
        None,
        req.inputs.len() > 1,
    ) {
        Ok(pair) => pair,
        Err((status, message)) => {
            return ImageOpSettled {
                response: Err((status, message, None)),
                job_id: None,
            };
        }
    };
    let command = req.command.clone();
    let token = job.token().clone();
    let artifacts = config.artifacts.clone();
    let job_id = job.id().to_owned();
    let work = tokio::task::spawn_blocking(move || {
        run_image_op_units(
            &command,
            &admitted,
            format,
            jpeg_quality,
            &token,
            &artifacts,
            &job_id,
        )
    })
    .await;
    drop(lease);
    let (units, input_records) = match settle_image_op_work(job.id(), req.inputs.len(), work) {
        Ok(pair) => pair,
        Err(settled) => return *settled,
    };
    // Bundle (best-effort, exactly like generation: a bundle write failure
    // never discards completed work — the response gains a warning).
    let mut warnings: Vec<String> = Vec::new();
    let bundle = config.data_dir.as_deref().and_then(|data_dir| {
        let output_records: Vec<crate::jobs::OpsOutputRecord> = units
            .iter()
            .enumerate()
            .map(|(i, unit)| crate::jobs::OpsOutputRecord {
                file: format!("op_{i:02}.{}", format.extension()),
                sha256: unit.sha256.clone(),
            })
            .collect();
        let byte_refs: Vec<&[u8]> = units.iter().map(|unit| unit.bytes.as_slice()).collect();
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let ops_inputs = crate::jobs::OpsBundleInputs {
            op: op_label(&req.command).to_owned(),
            command_sha256: sha256_hex(&command_sha256),
            request_id: req.request_id.clone(),
            client: req.client.clone(),
            inputs: input_records,
            format,
        };
        match crate::jobs::write_ops_bundle(
            data_dir,
            &ops_inputs,
            &output_records,
            &byte_refs,
            elapsed_ms,
        ) {
            Ok(dir) => Some(
                dir.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            ),
            Err(err) => {
                tracing::warn!(%err, "image-ops bundle write failed; response still complete");
                warnings.push(
                    "job bundle could not be written; artifact inputs will not resolve".to_owned(),
                );
                None
            }
        }
    });
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    ImageOpSettled {
        response: Ok(ImageOpResponse {
            created,
            job_id: job.id().to_owned(),
            request_id: req.request_id.clone(),
            bundle,
            outputs: units
                .iter()
                .map(|unit| ImageOpOutput {
                    b64_json: base64::engine::general_purpose::STANDARD.encode(&unit.bytes),
                    sha256: unit.sha256.clone(),
                    format: format.name().to_owned(),
                })
                .collect(),
            warnings: (!warnings.is_empty()).then_some(warnings),
        }),
        job_id: Some(job.id().to_owned()),
    }
}
