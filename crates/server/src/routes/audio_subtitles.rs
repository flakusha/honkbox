//! Subtitle timestamp and rendering helpers, extracted from `routes.rs`.
//!
//! These back the `srt` / `vtt` `response_format` branches of the audio
//! transcription endpoints; the router registrations stay in `routes.rs`.

use honkbox_audio_worker::TranscriptSegment as WorkerTranscriptSegment;

/// SRT timestamp: `HH:MM:SS,mmm` (comma decimal separator, per the
/// `SubRip` spec).
pub(super) fn srt_timestamp(ms: u32) -> String {
    subtitle_timestamp(ms, ',')
}

/// `WebVTT` timestamp: `HH:MM:SS.mmm`.
pub(super) fn vtt_timestamp(ms: u32) -> String {
    subtitle_timestamp(ms, '.')
}

/// `HH:MM:SS{sep}mmm` from milliseconds. Hours grow past two digits
/// unclamped (both formats allow it; whisper segments never get there).
pub(super) fn subtitle_timestamp(ms: u32, sep: char) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{sep}{millis:03}")
}

/// Render worker segments as `SubRip` (`response_format = "srt"`):
/// `index\nstart --> end\ntext\n\n` per cue, 1-based. No segments = an
/// empty body (a no-speech input yields an empty subtitle track).
pub(super) fn render_srt(segments: &[WorkerTranscriptSegment]) -> String {
    let mut out = String::new();
    for (index, segment) in segments.iter().enumerate() {
        out.push_str(&(index + 1).to_string());
        out.push('\n');
        out.push_str(&srt_timestamp(segment.start_ms));
        out.push_str(" --> ");
        out.push_str(&srt_timestamp(segment.end_ms));
        out.push('\n');
        out.push_str(&segment.text);
        out.push_str("\n\n");
    }
    out
}

/// Render worker segments as `WebVTT` (`response_format = "vtt"`): the
/// `WEBVTT` header, then the same cue shape with dot-decimal timestamps.
pub(super) fn render_vtt(segments: &[WorkerTranscriptSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for (index, segment) in segments.iter().enumerate() {
        out.push_str(&(index + 1).to_string());
        out.push('\n');
        out.push_str(&vtt_timestamp(segment.start_ms));
        out.push_str(" --> ");
        out.push_str(&vtt_timestamp(segment.end_ms));
        out.push('\n');
        out.push_str(&segment.text);
        out.push_str("\n\n");
    }
    out
}
