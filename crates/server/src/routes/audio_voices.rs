//! Kokoro voice-metadata helper and the `GET /v1/audio/voices` route,
//! extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only handlers and their helpers live
//! here.

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{Voice, VoiceList};

use super::AppConfig;

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

/// `GET /v1/audio/voices`: the voice set of the loaded TTS backend as an
/// OpenAI-style list (`{object: "list", data: [{object: "voice", …}]}`).
/// Kokoro (`kokoro-cpu`) cannot enumerate voices at runtime, so it serves the
/// statically documented `KOKORO_VOICES` set; every other backend returns
/// `data: []` rather than advertising another backend's ids. `object: "voice"`
/// and the list envelope stay stable across backends.
#[utoipa::path(
    get,
    path = "/v1/audio/voices",
    responses((status = 200, description = "Supported TTS voices", body = VoiceList))
)]
pub(crate) async fn get_audio_voices(State(config): State<AppConfig>) -> Response {
    let data = if config.tts.backend_id() == honkbox_audio_worker::KOKORO_BACKEND_ID {
        honkbox_audio_worker::KOKORO_VOICES
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
            .collect()
    } else {
        Vec::new()
    };
    Json(VoiceList {
        object: "list".to_owned(),
        data,
    })
    .into_response()
}
