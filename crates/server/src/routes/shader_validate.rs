//! `POST /v1/shader-validate` — headless shader compile-check hook
//! (TASK-shader-validate-hook).
//!
//! The validator is an opt-in external binary; the handler probes it per
//! request (one short process) so the answer always describes the binary that
//! would run. An absent binary is a named 501 carrying the probe status; there
//! is no silent skip. Shader text is staged in a unique temp directory for the
//! call and removed afterwards — honkbox never ships or executes shaders.

use std::path::PathBuf;
use std::time::Duration;

use axum::{
    Json,
    extract::rejection::JsonRejection,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_external_tools::ToolId;
use honkbox_external_tools::shader_validate::{self, SHADER_VALIDATOR_ENV};
use serde::{Deserialize, Serialize};

use super::openai_error_response;

/// One bounded shader compile request. The shader is text, never a host path.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ShaderValidateRequest {
    /// `.gdshader` / GLSL source text.
    pub shader: String,
    /// Logical file name used in diagnostics (no path separators).
    pub filename: Option<String>,
}

/// One reported compile failure, named by the shader and (when known) line.
#[derive(Debug, Serialize)]
pub(crate) struct ShaderIssueDto {
    pub file: String,
    pub line: Option<u32>,
    pub message: String,
}

/// The validator's verdict plus the tool version that produced it.
#[derive(Debug, Serialize)]
pub(crate) struct ShaderValidateResult {
    pub valid: bool,
    pub tool: String,
    pub version: Option<String>,
    pub filename: String,
    pub issues: Vec<ShaderIssueDto>,
}

const VALIDATE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_SHADER_BYTES: usize = 1024 * 1024;
const DEFAULT_FILENAME: &str = "shader.gdshader";

fn binary() -> PathBuf {
    super::middleware::env_var(SHADER_VALIDATOR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(shader_validate::binary)
}

fn unavailable(reason: &str) -> Response {
    openai_error_response(
        StatusCode::NOT_IMPLEMENTED,
        format!("shader validator unavailable: {reason}"),
        Some("shader_validator_unavailable"),
    )
}

fn invalid(reason: String) -> Response {
    openai_error_response(
        StatusCode::BAD_REQUEST,
        reason,
        Some("invalid_shader_validate_request"),
    )
}

/// Validate one shader before it ships in a bundle. The tool status is
/// checked first; a missing tool gets a named 501 even for a bad request.
#[utoipa::path(
    post,
    path = "/v1/shader-validate",
    request_body = serde_json::Value,
    responses(
        (status = 200, description = "Compile verdict with named issues", body = serde_json::Value),
        (status = 400, description = "Invalid shader request", body = honkbox_api_types::OpenAiError),
        (status = 500, description = "Staging or validator failure", body = honkbox_api_types::OpenAiError),
        (status = 501, description = "Validator absent or broken", body = honkbox_api_types::OpenAiError)
    )
)]
pub(super) async fn post_shader_validate(
    body: Result<Json<ShaderValidateRequest>, JsonRejection>,
) -> Response {
    let bin = binary();
    let tool = shader_validate::status(&bin, VALIDATE_TIMEOUT);
    if let Some(reason) = &tool.unavailable {
        return unavailable(&reason.to_string());
    }
    let Json(req) = match body {
        Ok(body) => body,
        Err(rejection) => {
            return invalid(format!(
                "invalid shader validate request: {}",
                rejection.body_text()
            ));
        }
    };
    if req.shader.trim().is_empty() || req.shader.len() > MAX_SHADER_BYTES {
        return invalid(format!(
            "shader text must be non-empty and at most {MAX_SHADER_BYTES} bytes"
        ));
    }
    let filename = req
        .filename
        .as_deref()
        .unwrap_or(DEFAULT_FILENAME)
        .to_owned();
    if filename.len() > 128
        || filename.is_empty()
        || !filename
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        || filename == "."
        || filename == ".."
    {
        return invalid("filename must be a plain file name without path separators".to_owned());
    }

    let dir = std::env::temp_dir().join(format!(
        "honkbox-shader-validate-{}-{}",
        std::process::id(),
        ulid::Ulid::new()
    ));
    let shader_path = dir.join(&filename);
    let shader = req.shader;
    let staged = tokio::task::spawn_blocking(move || {
        let result = match std::fs::create_dir_all(&dir)
            .and_then(|()| std::fs::write(&shader_path, shader))
        {
            Ok(()) => Ok(shader_validate::validate(
                &bin,
                &shader_path,
                VALIDATE_TIMEOUT,
            )),
            Err(err) => Err(err.to_string()),
        };
        let _ = std::fs::remove_dir_all(&dir);
        result
    })
    .await;
    let report = match staged {
        Ok(Ok(report)) => report,
        Ok(Err(io)) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("cannot stage shader: {io}"),
                Some("shader_validate_stage_failed"),
            );
        }
        Err(join) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("shader validation task failed: {join}"),
                Some("shader_validate_failed"),
            );
        }
    };
    match report {
        Ok(report) => {
            let valid = report.is_valid();
            // The validator may echo the staged host path; publish only the
            // logical name the caller supplied.
            let issues = report
                .issues
                .into_iter()
                .map(|issue| ShaderIssueDto {
                    file: filename.clone(),
                    line: issue.line,
                    message: issue.message,
                })
                .collect();
            Json(ShaderValidateResult {
                valid,
                tool: ToolId::ShaderValidator.as_str().to_owned(),
                version: tool.version,
                filename,
                issues,
            })
            .into_response()
        }
        Err(reason) => unavailable(&reason.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    static ENV_LOCK: Mutex<()> = Mutex::new(());
    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    /// Restores the env override even when a test panics; the lock is held
    /// separately by the test for its whole body.
    struct EnvRestore(&'static str);

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            super::super::set_env_var(self.0, None);
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "honkbox-shader-route-{}-{}-{tag}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[cfg(unix)]
    fn fake_binary(tag: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = scratch(tag);
        let path = dir.join("validator");
        let staged = dir.join("validator.staged");
        std::fs::write(&staged, script).expect("fake validator");
        let mut perms = std::fs::metadata(&staged).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&staged, perms).expect("chmod");
        std::fs::rename(staged, &path).expect("rename");
        path
    }

    fn request(
        shader: &str,
        filename: Option<&str>,
    ) -> Result<Json<ShaderValidateRequest>, JsonRejection> {
        Ok(Json(ShaderValidateRequest {
            shader: shader.to_owned(),
            filename: filename.map(str::to_owned),
        }))
    }

    async fn body(response: Response) -> (StatusCode, serde_json::Value) {
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        (status, serde_json::from_slice(&bytes).unwrap_or_default())
    }

    #[tokio::test]
    async fn an_absent_validator_is_a_named_501() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _restore = EnvRestore(SHADER_VALIDATOR_ENV);
        let missing = scratch("missing").join("no-such-validator");
        super::super::set_env_var(SHADER_VALIDATOR_ENV, Some(&missing.to_string_lossy()));
        let (status, value) =
            body(post_shader_validate(request("shader_type spatial;", None)).await).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(value["error"]["code"], "shader_validator_unavailable");
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not installed")
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_diagnostic_from_the_validator_names_the_shader_and_line() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _restore = EnvRestore(SHADER_VALIDATOR_ENV);
        let bin = fake_binary(
            "diagnostic",
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'validator 2.0'; exit 0; fi\nprintf '%s\\n' \"$1:7: unexpected token\"\nexit 1\n",
        );
        let staged_dir = bin.parent().expect("parent").to_string_lossy().into_owned();
        super::super::set_env_var(SHADER_VALIDATOR_ENV, Some(&bin.to_string_lossy()));
        let (status, value) = body(
            post_shader_validate(request("shader_type spatial;", Some("hero.gdshader"))).await,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{value}");
        assert_eq!(value["valid"], false);
        assert_eq!(value["tool"], "shader_validator");
        assert_eq!(value["filename"], "hero.gdshader");
        assert_eq!(value["issues"][0]["file"], "hero.gdshader");
        assert_eq!(value["issues"][0]["line"], 7);
        assert_eq!(value["issues"][0]["message"], "unexpected token");
        assert!(!value.to_string().contains(&staged_dir), "{value}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_clean_shader_returns_a_named_result() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _restore = EnvRestore(SHADER_VALIDATOR_ENV);
        let bin = fake_binary(
            "clean",
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'validator 2.0'; exit 0; fi\necho OK\n",
        );
        super::super::set_env_var(SHADER_VALIDATOR_ENV, Some(&bin.to_string_lossy()));
        let (status, value) =
            body(post_shader_validate(request("shader_type spatial;", None)).await).await;
        assert_eq!(status, StatusCode::OK, "{value}");
        assert_eq!(value["valid"], true);
        assert_eq!(value["version"], "2.0");
        assert_eq!(value["issues"], serde_json::json!([]));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_path_like_filename_is_a_named_400() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _restore = EnvRestore(SHADER_VALIDATOR_ENV);
        let bin = fake_binary("bad-name", "#!/bin/sh\necho 'validator 2.0'\n");
        super::super::set_env_var(SHADER_VALIDATOR_ENV, Some(&bin.to_string_lossy()));
        let (status, value) = body(
            post_shader_validate(request("shader_type spatial;", Some("../evil.gdshader"))).await,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(value["error"]["code"], "invalid_shader_validate_request");
    }
}
