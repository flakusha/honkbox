//! Rerank endpoint (`FEAT-rerank-endpoint`).
//!
//! TEI callers send `texts[]`, Jina callers send `documents[]`; both map to
//! the same per-request sidecar rank call. Scores are sorted, `top_n`-cut and
//! returned in the shared `RerankResult` shape. There is no embedding-model
//! fallback and no shared ranker process: the seam owns a temp job dir and a
//! supervised child for every request.

use axum::{
    Json,
    body::Bytes,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{RerankItem, RerankRequest, RerankResult};
use serde::Deserialize;
use utoipa::ToSchema;

use super::runtime_config::{python_allowlist, rerank_supported};

use super::python_sidecar;

/// One Jina `documents[]` entry: a bare string or a `text`/`content` object.
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(untagged)]
enum RerankDocument {
    /// Plain text.
    Text(String),
    /// Object form (`{"text": ...}` or `{"content": ...}`).
    Object {
        /// The candidate text.
        #[serde(default)]
        text: Option<String>,
        /// Alternate text field some Jina clients use.
        #[serde(default)]
        content: Option<String>,
    },
}

/// Accepts both wire shapes; `RerankRequest` itself is the shared contract
/// handed to the sidecar seam. `RerankRequest` is serialize-only in
/// `api-types`, so this is the sole deserialization surface.
#[derive(Debug, Deserialize, ToSchema)]
pub(super) struct RerankBody {
    /// Query to score candidates against.
    query: String,
    /// TEI shape.
    #[serde(default)]
    texts: Option<Vec<String>>,
    /// Jina shape.
    #[serde(default)]
    documents: Option<Vec<RerankDocument>>,
    /// How many top-scoring candidates to return.
    #[serde(default)]
    top_n: Option<usize>,
    /// Rerank model slug.
    #[serde(default)]
    model: Option<String>,
    /// Whether each result carries its original text.
    #[serde(default)]
    return_text: Option<bool>,
}

impl RerankDocument {
    /// Flatten to text, keeping an invalid entry as an empty candidate so
    /// result indices still match the request order.
    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Object { text, content } => text.or(content).unwrap_or_default(),
        }
    }
}

/// Split the two accepted wire shapes into one ordered candidate list.
/// `texts` wins when both are present.
fn candidates(body: &RerankBody) -> Vec<String> {
    if let Some(texts) = &body.texts {
        return texts.clone();
    }
    body.documents
        .as_ref()
        .map(|docs| {
            docs.iter()
                .cloned()
                .map(RerankDocument::into_text)
                .collect()
        })
        .unwrap_or_default()
}

/// One bad request, as a plain text response (TEI/Jina flavor).
fn bad_request(message: &str) -> Response {
    (StatusCode::BAD_REQUEST, message.to_owned()).into_response()
}

/// Named 501 for an unconfigured ranker, shared by both guards below.
fn no_ranker() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        "rerank unavailable: no ranker model configured; see GET /v1/capabilities rerank"
            .to_owned(),
    )
        .into_response()
}

/// Shape sidecar scores into the wire result: stable order, `top_n` cut,
/// optional original text. Pure so stub scores can exercise the full shape.
fn scored_result(
    model: String,
    texts: &[String],
    scores: Vec<f64>,
    top_n: Option<usize>,
    return_text: bool,
) -> RerankResult {
    let mut results: Vec<RerankItem> = scores
        .into_iter()
        .enumerate()
        .map(|(index, score)| RerankItem {
            index,
            score,
            text: return_text.then(|| texts[index].clone()),
        })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.index.cmp(&b.index))
    });
    results.truncate(top_n.unwrap_or(results.len()).min(results.len()));
    RerankResult { results, model }
}

/// Shared implementation behind both route registrations.
async fn rerank(body: Bytes) -> Response {
    let body: RerankBody = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(err) => return bad_request(&format!("invalid rerank request: {err}")),
    };
    let texts = candidates(&body);
    if texts.is_empty() {
        return bad_request("rerank needs a non-empty texts[] or documents[]");
    }
    if !rerank_supported() {
        return no_ranker();
    }
    let model = body
        .model
        .clone()
        .or_else(|| python_allowlist().first().cloned());
    let Some(model) = model.filter(|slug| !slug.is_empty()) else {
        return no_ranker();
    };

    let request = RerankRequest {
        query: body.query.clone(),
        texts: Some(texts.clone()),
        documents: None,
        top_n: body.top_n,
        model: Some(model.clone()),
        return_text: body.return_text,
    };
    let scores = match python_sidecar::rank(&request.query, texts.as_slice(), &model).await {
        Ok(scores) => scores,
        Err(err) if err.starts_with(python_sidecar::UNAVAILABLE_PREFIX) => {
            return (StatusCode::NOT_IMPLEMENTED, err).into_response();
        }
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err).into_response(),
    };

    let result = scored_result(
        model,
        &texts,
        scores,
        request.top_n,
        request.return_text.unwrap_or(false),
    );
    Json(result).into_response()
}

/// `POST /v1/rerank` — TEI `texts[]` or Jina `documents[]` in, ranked items out.
#[utoipa::path(
    post,
    path = "/v1/rerank",
    request_body = RerankBody,
    responses(
        (status = 200, description = "Reranked candidates", body = RerankResult),
        (status = 400, description = "Bad request"),
        (status = 501, description = "No ranker available"),
        (status = 500, description = "Rerank failed")
    )
)]
pub(super) async fn post_rerank(body: Bytes) -> Response {
    rerank(body).await
}

/// Trailing-slash twin of [`post_rerank`]; the router registers both.
#[utoipa::path(
    post,
    path = "/v1/rerank/",
    request_body = RerankBody,
    responses(
        (status = 200, description = "Reranked candidates", body = RerankResult),
        (status = 400, description = "Bad request"),
        (status = 501, description = "No ranker available"),
        (status = 500, description = "Rerank failed")
    )
)]
pub(super) async fn post_rerank_slash(body: Bytes) -> Response {
    rerank(body).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stubbed_scores_produce_a_valid_ranked_result() {
        let texts = ["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()];
        let result = scored_result(
            "ranker:stub".to_owned(),
            &texts,
            vec![0.1, 0.9, 0.5],
            Some(2),
            true,
        );
        assert_eq!(result.model, "ranker:stub");
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.results[0].index, 1);
        assert_eq!(result.results[0].text.as_deref(), Some("beta"));
        assert_eq!(result.results[1].index, 2);
        assert_eq!(result.results[1].text.as_deref(), Some("gamma"));
    }

    #[tokio::test]
    async fn unconfigured_rerank_route_returns_named_501() {
        // The 501 must be produced before any slot, job dir or child
        // process is touched; do not assert on global FACTS here, which
        // other tests may have configured first.
        let response = rerank(Bytes::from_static(
            br#"{"query":"q","texts":["alpha","beta"]}"#,
        ))
        .await;
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        assert!(
            String::from_utf8_lossy(&body).contains("no ranker model configured"),
            "{body:?}"
        );
    }
}
