//! Image-op job settlement: map blocking-pool outcomes onto units or named
//! failures, and execute one command end to end.

use std::time::Instant;

use axum::http::StatusCode;
use base64::Engine as _;
use honkbox_api_types::{ImageOpCommand, ImageOpOutput, ImageOpRequest, ImageOpResponse};

use super::AppConfig;
use super::chat_prep::admit_then_register;
use super::image_ops_dedup::{ImageOpSettled, op_label, sha256_hex};
use super::image_ops_exec::{
    AdmittedOpInput, ImageOpUnitOutput, ImageOpWork, ImageOpWorkError, admit_op_inputs,
    check_matting_inputs, image_op_pre_admission_join_failure, matting_sidecar_url,
    run_image_op_units, settled_without_job,
};

/// Bundle-write failure is one explicit, ADVISORY outcome: completed pixels
/// stay in the response, this warning is returned, and no stable handles are
/// advertised. Never a failed op.
const BUNDLE_WRITE_WARNING: &str =
    "job bundle could not be written; artifact inputs will not resolve";

/// Persist explicit stable references next to `ops.json`: input digests and
/// artifact ids, output content-addressed ids and digests. Both sides
/// resolve after a restart without any in-memory ledger.
fn write_artifact_records(
    dir: &std::path::Path,
    bundle: &str,
    inputs: &[crate::jobs::OpsInputRecord],
    outputs: &[crate::jobs::OpsOutputRecord],
) -> Result<(), String> {
    let input_ids = inputs
        .iter()
        .map(|record| {
            let artifact_id = record
                .artifact_id
                .clone()
                .unwrap_or_else(|| format!("sha256:{}", record.sha256));
            serde_json::json!({
                "source": &record.source,
                "artifact_id": artifact_id,
                "sha256": &record.sha256,
            })
        })
        .collect::<Vec<_>>();
    let output_ids = outputs
        .iter()
        .enumerate()
        .map(|(index, record)| {
            serde_json::json!({
                "index": index,
                "artifact_id": &record.artifact_id,
                "file": &record.file,
                "sha256": &record.sha256,
            })
        })
        .collect::<Vec<_>>();
    let sidecar = serde_json::json!({
        "bundle": bundle,
        "inputs": input_ids,
        "outputs": output_ids,
    });
    crate::jobs::write_atomic(
        dir,
        "artifacts.json",
        serde_json::to_string_pretty(&sidecar)
            .map_err(|err| format!("cannot encode artifacts.json: {err}"))?
            .as_bytes(),
    )
}

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

/// Run the `matting` op through the rembg-class HTTP sidecar: POST each
/// admitted PNG and expect the RGBA cutout PNG back. Fails fast on the
/// first bad input; a non-PNG sidecar reply is a 500 (the contract is
/// PNG-in/PNG-out, never coerced).
async fn run_matting_sidecar(
    url: &str,
    admitted: &[AdmittedOpInput],
    token: &tokio_util::sync::CancellationToken,
) -> Result<ImageOpWork, ImageOpWorkError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .map_err(|e| ImageOpWorkError::Failed(format!("matting sidecar client failed: {e}")))?;
    let mut outputs = Vec::with_capacity(admitted.len());
    let mut records = Vec::with_capacity(admitted.len());
    for (i, input) in admitted.iter().enumerate() {
        // Cooperative cancel honored BETWEEN inputs, like the CPU path.
        if token.is_cancelled() {
            return Err(ImageOpWorkError::Cancelled { completed: i });
        }
        let reply = client
            .post(url)
            .header("content-type", "image/png")
            .body(input.raw.clone())
            .send()
            .await
            .map_err(|e| {
                ImageOpWorkError::Failed(format!("matting sidecar request failed: {e}"))
            })?;
        if !reply.status().is_success() {
            return Err(ImageOpWorkError::Failed(format!(
                "matting sidecar answered {}",
                reply.status()
            )));
        }
        let bytes = reply
            .bytes()
            .await
            .map_err(|e| ImageOpWorkError::Failed(format!("matting sidecar read failed: {e}")))?
            .to_vec();
        // The cutout contract is PNG: sniff, never trust the reply blindly.
        match crate::formats::sniff(&bytes) {
            Some(row) if row.name == "png" => {}
            Some(row) => {
                return Err(ImageOpWorkError::Failed(format!(
                    "matting sidecar returned {} bytes, not PNG",
                    row.name
                )));
            }
            None => {
                return Err(ImageOpWorkError::Failed(
                    "matting sidecar returned unrecognizable bytes".to_owned(),
                ));
            }
        }
        records.push(crate::jobs::OpsInputRecord {
            source: input.source.clone(),
            sha256: input.sha256.clone(),
            artifact_id: input.artifact_id.clone(),
        });
        outputs.push(ImageOpUnitOutput {
            sha256: sha256_hex(&bytes),
            bytes,
        });
    }
    Ok((outputs, records))
}

/// Execute one image-op command: register the job, acquire CPU admission,
/// run the units on the blocking pool (cancel checked between inputs),
/// write the bundle (ADVISORY on failure: complete outputs still return
/// with a warning and no stable handles), and build the response.
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
    let registry = config.artifacts.clone();
    let admitted = match tokio::task::spawn_blocking(move || {
        admit_op_inputs(&inputs, data_dir.as_deref(), &registry)
    })
    .await
    {
        Ok(Ok(admitted)) => admitted,
        Ok(Err(err)) => return settled_without_job(err),
        Err(join) => {
            return image_op_pre_admission_join_failure(format!("image-op worker failed: {join}"));
        }
    };
    // Matting pre-admission: PNG-only input is a named 400 and an
    // unconfigured sidecar is an honest 501 — both with NO job, like the
    // other pre-admission rejections above.
    let matting_url = if matches!(req.command, ImageOpCommand::Matting) {
        if let Err(err) = check_matting_inputs(&admitted) {
            return settled_without_job(err);
        }
        match matting_sidecar_url() {
            Some(url) => Some(url),
            None => {
                return ImageOpSettled {
                    response: Err((
                        StatusCode::NOT_IMPLEMENTED,
                        "matting unavailable: no sidecar configured (set HONKBOX_MATTING_URL); see GET /v1/capabilities image_ops"
                            .to_owned(),
                        Some("matting_unavailable".to_owned()),
                    )),
                    job_id: None,
                };
            }
        }
    } else {
        None
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
    // Debug-only provenance: read admitted upload text before `admitted`
    // moves into the worker. These annotations never reach op execution.
    let source_annotations: Vec<crate::jobs::BundleAnnotation> = admitted
        .iter()
        .flat_map(|input| crate::jobs::image_metadata::png_source_annotations(&input.raw))
        .collect();
    let (units, input_records) = if let Some(url) = matting_url {
        // Matting bypasses the blocking pool: the sidecar is an async HTTP
        // call per input (cancel honored BETWEEN inputs, like the CPU path).
        drop(lease);
        match run_matting_sidecar(&url, &admitted, &token).await {
            Ok(pair) => pair,
            Err(ImageOpWorkError::Cancelled { completed }) => {
                return ImageOpSettled {
                    response: Err((
                        StatusCode::CONFLICT,
                        format!(
                            "job {} cancelled before completion ({completed} of {} input(s) processed)",
                            job.id(),
                            req.inputs.len()
                        ),
                        Some("job_cancelled".to_owned()),
                    )),
                    job_id: Some(job.id().to_owned()),
                };
            }
            Err(ImageOpWorkError::Bad { message, code }) => {
                return ImageOpSettled {
                    response: Err((StatusCode::BAD_REQUEST, message, Some(code.to_owned()))),
                    job_id: Some(job.id().to_owned()),
                };
            }
            Err(ImageOpWorkError::Failed(message)) => {
                return ImageOpSettled {
                    response: Err((StatusCode::INTERNAL_SERVER_ERROR, message, None)),
                    job_id: Some(job.id().to_owned()),
                };
            }
        }
    } else {
        let worker_job_id = job_id.clone();
        let work = tokio::task::spawn_blocking(move || {
            run_image_op_units(
                &command,
                &admitted,
                format,
                jpeg_quality,
                &token,
                &artifacts,
                &worker_job_id,
            )
        })
        .await;
        drop(lease);
        match settle_image_op_work(job.id(), req.inputs.len(), work) {
            Ok(pair) => pair,
            Err(settled) => return *settled,
        }
    };
    // Bundle (ADVISORY on failure): promote each output to a content-addressed
    // blob first, then write the bundle with stable ids recorded next to the
    // bytes. Promotion dedups by sha256, so retries never duplicate storage.
    let mut warnings: Vec<String> = Vec::new();
    let bundle = config.data_dir.as_deref().and_then(|data_dir| {
        let promoted = units
            .iter()
            .map(|unit| {
                config
                    .artifacts
                    .put_bytes(&unit.bytes, format.mime())
                    .map_err(|err| format!("artifact promotion failed: {err}"))
            })
            .collect::<Result<Vec<_>, _>>();
        let promoted = match promoted {
            Ok(promoted) => promoted,
            Err(err) => {
                tracing::warn!(%err, "image-ops artifact promotion failed; inline fallback");
                warnings.push(BUNDLE_WRITE_WARNING.to_owned());
                return None;
            }
        };
        let output_records: Vec<crate::jobs::OpsOutputRecord> = units
            .iter()
            .zip(promoted)
            .enumerate()
            .map(|(i, (_unit, info))| crate::jobs::OpsOutputRecord {
                file: format!("op_{i:02}.{}", format.extension()),
                sha256: info.sha256,
                artifact_id: info.id,
                bytes: info.size_bytes,
            })
            .collect();
        let byte_refs: Vec<&[u8]> = units.iter().map(|unit| unit.bytes.as_slice()).collect();
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let ops_inputs = crate::jobs::OpsBundleInputs {
            op: op_label(&req.command).to_owned(),
            command_sha256: sha256_hex(&command_sha256),
            request_id: req.request_id.clone(),
            job_id: job_id.clone(),
            client: req.client.clone(),
            inputs: input_records.clone(),
            format,
            annotations: source_annotations,
        };
        let written = crate::jobs::write_ops_bundle(
            data_dir,
            &ops_inputs,
            &output_records,
            &byte_refs,
            elapsed_ms,
        )
        .and_then(|dir| {
            let bundle = dir
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            write_artifact_records(&dir, &bundle, &input_records, &output_records).map(|()| bundle)
        });
        match written {
            Ok(bundle) => Some(bundle),
            Err(err) => {
                tracing::warn!(%err, "image-ops bundle write failed; response still complete");
                warnings.push(BUNDLE_WRITE_WARNING.to_owned());
                None
            }
        }
    });
    // No bundle means base64 is the only handle: say so instead of implying
    // a stable reference the client could later feed back as an artifact id.
    if bundle.is_none()
        && !warnings
            .iter()
            .any(|warning| warning == BUNDLE_WRITE_WARNING)
    {
        warnings.push("no job bundle: outputs are inline base64 only".to_owned());
    }
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    ImageOpSettled {
        response: Ok(ImageOpResponse {
            created,
            job_id: job.id().to_owned(),
            request_id: req.request_id.clone(),
            bundle: bundle.clone(),
            outputs: units
                .iter()
                .map(|unit| ImageOpOutput {
                    b64_json: base64::engine::general_purpose::STANDARD.encode(&unit.bytes),
                    sha256: unit.sha256.clone(),
                    format: format.name().to_owned(),
                    // The promoted blob id IS the sha256 of these bytes;
                    // returned only when a bundle was actually written.
                    artifact_id: bundle.as_ref().map(|_| unit.sha256.clone()),
                })
                .collect(),
            warnings: (!warnings.is_empty()).then_some(warnings),
        }),
        job_id: Some(job.id().to_owned()),
    }
}
