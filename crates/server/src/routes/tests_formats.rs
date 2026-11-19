//! Format-registry and subtitle timestamp/rendering tests.

use super::*;
use honkbox_audio_worker::TranscriptSegment as WorkerTranscriptSegment;

/// The registry's write rows are EXACTLY the compiled encoders — set
/// and order on both directions. This is the lock that makes
/// `formats.write` (derived through the registry) byte-compat with the
/// pre-registry `OutputFormat::ALL` derivation, and it lives HERE (the
/// routes layer owns the wire-name → enum mapping, so a `#[path]`
/// -included formats module never links the native backends).
#[test]
fn format_registry_matches_the_compiled_encoders() {
    for format in honkbox_sd_core::OutputFormat::ALL {
        assert!(
            crate::formats::output_rows().any(|row| row.output && row.name == format.name()),
            "{} encoder missing from the registry write column",
            format.name()
        );
        assert!(
            parse_output_format(format.name()).is_ok(),
            "{} must parse",
            format.name()
        );
    }
    for row in crate::formats::output_rows() {
        assert!(
            honkbox_sd_core::OutputFormat::from_name(row.name).is_some(),
            "registry write row {:?} has no compiled encoder",
            row.name
        );
    }
    // Unknown names (and wrong-case spellings) stay rejected, naming
    // the registry-derived accepted values.
    for raw in ["gif", "JPEG", "Png", "", "svg", "ora"] {
        let (message, code) = parse_output_format(raw).expect_err(raw);
        assert_eq!(code, "unsupported_output_format");
        assert!(message.contains(raw), "{raw}: {message}");
        assert!(message.contains("\"png\""), "{raw}: {message}");
    }
}

/// SRT timestamps use the `SubRip` comma separator; hours pad to two
/// digits and roll over correctly at minute/second/hour boundaries.
#[test]
fn srt_timestamps_render_deterministically() {
    assert_eq!(srt_timestamp(0), "00:00:00,000");
    assert_eq!(srt_timestamp(100), "00:00:00,100");
    assert_eq!(srt_timestamp(3_661_001), "01:01:01,001");
    assert_eq!(srt_timestamp(u32::MAX), "1193:02:47,295");
}

/// VTT timestamps use the dot separator; everything else matches SRT.
#[test]
fn vtt_timestamps_render_deterministically() {
    assert_eq!(vtt_timestamp(0), "00:00:00.000");
    assert_eq!(vtt_timestamp(1_000), "00:00:01.000");
    assert_eq!(vtt_timestamp(3_661_001), "01:01:01.001");
}

/// SRT cue rendering: 1-based numbering, arrow line, blank-line tail.
#[test]
fn srt_renders_numbered_cues() {
    let segments = vec![
        WorkerTranscriptSegment {
            start_ms: 0,
            end_ms: 1_500,
            text: "hello".to_owned(),
        },
        WorkerTranscriptSegment {
            start_ms: 2_000,
            end_ms: 3_250,
            text: "world".to_owned(),
        },
    ];
    assert_eq!(
        render_srt(&segments),
        "1\n00:00:00,000 --> 00:00:01,500\nhello\n\n\
         2\n00:00:02,000 --> 00:00:03,250\nworld\n\n"
    );
    // No segments = an empty (valid) subtitle track.
    assert_eq!(render_srt(&[]), "");
}

/// VTT cue rendering: the WEBVTT header, then the same cue shape.
#[test]
fn vtt_renders_header_and_cues() {
    let segments = vec![WorkerTranscriptSegment {
        start_ms: 0,
        end_ms: 1_500,
        text: "hello".to_owned(),
    }];
    assert_eq!(
        render_vtt(&segments),
        "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.500\nhello\n\n"
    );
    // Header-only track for a no-speech input.
    assert_eq!(render_vtt(&[]), "WEBVTT\n\n");
}
