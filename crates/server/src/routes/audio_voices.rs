//! Kokoro voice-metadata helper and the `GET /v1/audio/voices` route,
//! extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only handlers and their helpers live
//! here.

use axum::{
    Json,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{Voice, VoiceList};

/// Map a Kokoro voice-id prefix to `(language, gender)` metadata.
/// Kokoro ids are `<lang><gender>_<name>`: `a`/`b` US/GB English, `e` es,
/// `f` fr, `h`/`z` zh, `i` it, `j` ja, `p` pt-BR; `f` female, `m` male.
pub(super) fn kokoro_voice_meta(id: &str) -> (&'static str, &'static str) {
    let mut chars = id.chars();
    let lang = chars.next();
    let gender = chars.next();
    let language = match lang {
        Some('a') => "en-US",
        Some('b') => "en-GB",
        Some('e') => "es-ES",
        Some('f') => "fr-FR",
        Some('h' | 'z') => "zh-CN",
        Some('i') => "it-IT",
        Some('j') => "ja-JP",
        Some('p') => "pt-BR",
        _ => "unknown",
    };
    let gender = match gender {
        Some('f') => "female",
        Some('m') => "male",
        _ => "unknown",
    };
    (language, gender)
}

/// `GET /v1/audio/voices`: the supported TTS voice set as an OpenAI-style
/// list (`{object: "list", data: [{object: "voice", …}]}`). The Kokoro
/// backend cannot enumerate voices at runtime, so this serves the
/// statically documented `KOKORO_VOICES` set — honest, no model loaded,
/// no state. `data[].id` values are exactly what `/v1/audio/speech`
/// accepts as `voice`.
#[utoipa::path(
    get,
    path = "/v1/audio/voices",
    responses((status = 200, description = "Supported TTS voices", body = VoiceList))
)]
pub(super) async fn get_audio_voices() -> Response {
    let data = honkbox_audio_worker::KOKORO_VOICES
        .iter()
        .map(|id| {
            let (language, gender) = kokoro_voice_meta(id);
            Voice {
                id: (*id).to_owned(),
                object: "voice".to_owned(),
                language: language.to_owned(),
                gender: gender.to_owned(),
            }
        })
        .collect();
    Json(VoiceList {
        object: "list".to_owned(),
        data,
    })
    .into_response()
}
