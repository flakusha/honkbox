//! Face detection backend (gated policy surface): SCRFD/RetinaFace ONNX
//! via the workspace-pinned `ort`, CPU only, behind the non-default `face`
//! feature. honkbox bundles no weights; `new` with missing or unloadable
//! weights yields an absent engine, so `serves(Face)` is false and the route
//! answers the named 501 — never a fake result. Input is the decoded
//! image-container bytes the route supplies; nothing is fetched, nothing is
//! retained past the call (local-only retention, policy gate 2).
//! Policy record: `.tmp/research/face-detection-policy-decision-2026-11-20.md`
//! (consent / local-only / default-off / capabilities / export — all five gates).
//!
//! Output layout contract: the classic 3-stride detector heads (`score_8` /
//! `score_16` / `score_32` for fg scores, `bbox_8` / … for box distances,
//! optional `landmark_8` / … or `kps_8` / … for 5-point landmarks), anchor-free
//! priors at grid centers, distances in stride units. Models that do not
//! expose the score/bbox pairs answer an honest engine error, never silence.

use std::path::Path;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

use ort::value::Tensor;

use super::{
    RecEngine, RecError, RecKind, RecPage, RecRegion, RecognitionPipeline, RecognizeOutput,
    RecognizeRequest,
};

/// Stable backend id (sd-core `SD_CPP_BACKEND_ID` pattern).
pub const FACE_BACKEND_ID: &str = "scrfd-ort";

/// The linked `ort` DEPENDENCY version this build compiles (workspace pin,
/// shared with the audio VAD — no new vendor class).
pub const FACE_ORT_VERSION: &str = "2.0.0-rc.15";

/// Detector input edge: SCRFD/RetinaFace standard.
const INPUT_EDGE: usize = 640;
/// Score threshold and NMS IoU, the upstream defaults.
const SCORE_THRESHOLD: f32 = 0.5;
const NMS_IOU: f32 = 0.4;

/// Gated face detector. Absent (no usable weights) until `new` loads an ONNX
/// session; presence is the honesty source for `serves(Face)`.
pub struct FaceDetection {
    session: Option<Mutex<ort::session::Session>>,
    cancelled: AtomicBool,
}

impl std::fmt::Debug for FaceDetection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaceDetection")
            .field("available", &self.session.is_some())
            .finish_non_exhaustive()
    }
}

impl FaceDetection {
    /// Build from an already-located SCRFD/RetinaFace ONNX file. `None` or an
    /// unloadable file yields an absent engine (honest `serves(Face) == false`),
    /// never a stub result.
    #[must_use]
    pub fn new(weights: Option<&Path>) -> Self {
        let session = weights.and_then(|path| {
            if !path.is_file() {
                return None;
            }
            let env = ort::init().with_name("honkbox-face").build().ok()?;
            let mut builder = ort::session::Session::builder(&env).ok()?;
            builder.commit_from_file(path).ok().map(Mutex::new)
        });
        Self {
            session,
            cancelled: AtomicBool::new(false),
        }
    }

    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

/// One raw detection in letterboxed 640×640 coordinates.
struct RawDet {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    kps: [f32; 10],
}

fn decode_stride(
    scores: &[f32],
    score_cols: usize,
    boxes: &[f32],
    kps: Option<&[f32]>,
    stride: usize,
    out: &mut Vec<RawDet>,
) {
    // Rows are (cell, anchor) pairs, 2 anchors per grid cell.
    let rows = boxes.len() / 4;
    if rows == 0 {
        return;
    }
    let gw = ((rows / 2) as f32).sqrt() as usize;
    if gw == 0 {
        return;
    }
    let stride = stride as f32;
    for i in 0..rows {
        let score = if score_cols >= 2 {
            // Two-column bg/fg logits: softmax, take fg.
            let b = scores[i * 2].exp();
            let f = scores[i * 2 + 1].exp();
            f / (b + f)
        } else {
            // Single logit: sigmoid.
            1.0 / (1.0 + (-scores[i]).exp())
        };
        if score < SCORE_THRESHOLD {
            continue;
        }
        let cell = i / 2;
        let cx = (cell % gw) as f32 * stride;
        let cy = (cell / gw) as f32 * stride;
        let d = &boxes[i * 4..i * 4 + 4];
        let k = kps.map_or([0.0; 10], |k| {
            let k = &k[i * 10..i * 10 + 10];
            [
                cx + k[0] * stride,
                cy + k[1] * stride,
                cx + k[2] * stride,
                cy + k[3] * stride,
                cx + k[4] * stride,
                cy + k[5] * stride,
                cx + k[6] * stride,
                cy + k[7] * stride,
                cx + k[8] * stride,
                cy + k[9] * stride,
            ]
        });
        out.push(RawDet {
            x1: cx - d[0] * stride,
            y1: cy - d[1] * stride,
            x2: cx + d[2] * stride,
            y2: cy + d[3] * stride,
            score,
            kps: k,
        });
    }
}

fn nms(mut dets: Vec<RawDet>) -> Vec<RawDet> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<RawDet> = Vec::with_capacity(dets.len());
    'outer: for det in dets {
        for prior in &kept {
            let ix1 = det.x1.max(prior.x1);
            let iy1 = det.y1.max(prior.y1);
            let ix2 = det.x2.min(prior.x2);
            let iy2 = det.y2.min(prior.y2);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
            let union = (det.x2 - det.x1).max(0.0) * (det.y2 - det.y1).max(0.0)
                + (prior.x2 - prior.x1).max(0.0) * (prior.y2 - prior.y1).max(0.0)
                - inter;
            if union > 0.0 && inter / union > NMS_IOU {
                continue 'outer;
            }
        }
        kept.push(det);
    }
    kept
}

/// Letterbox `rgb` into a 640×640 RGB CHW blob; returns the blob plus the
/// scale/pad needed to map detections back to source pixels.
fn letterbox(rgb: &image::RgbImage) -> (Vec<f32>, f32, f32, f32) {
    use image::imageops::FilterType;
    let (w, h) = (rgb.width() as f32, rgb.height() as f32);
    let scale = INPUT_EDGE as f32 / w.max(h);
    let nw = (w * scale).round() as u32;
    let nh = (h * scale).round() as u32;
    let resized = image::imageops::resize(rgb, nw, nh, FilterType::Triangle);
    let pad_x = (INPUT_EDGE as u32 - nw) / 2;
    let pad_y = (INPUT_EDGE as u32 - nh) / 2;
    let mut blob = vec![0.0f32; 3 * INPUT_EDGE * INPUT_EDGE];
    for y in 0..nh {
        for x in 0..nw {
            let p = resized.get_pixel(x, y).0;
            for c in 0..3 {
                // SCRFD normalization: RGB order, (x - 127.5) / 128.
                blob[c * INPUT_EDGE * INPUT_EDGE
                    + ((y + pad_y) as usize) * INPUT_EDGE
                    + (x + pad_x) as usize] = (f32::from(p[c]) - 127.5) / 128.0;
            }
        }
    }
    (blob, scale, pad_x as f32, pad_y as f32)
}

impl RecognitionPipeline for FaceDetection {
    fn backend_id(&self) -> &'static str {
        FACE_BACKEND_ID
    }

    fn kinds(&self) -> &'static [RecKind] {
        &[RecKind::Face]
    }

    /// Only a loaded session serves: missing weights stay an honest 501.
    fn serves(&self, kind: RecKind) -> bool {
        kind == RecKind::Face && self.session.is_some()
    }

    fn recognize(&self, req: &RecognizeRequest) -> Result<RecognizeOutput, RecError> {
        if req.kind != RecKind::Face {
            return Err(RecError::Unsupported {
                kind: req.kind,
                backend: self.backend_id(),
            });
        }
        if self.session.is_none() {
            return Err(RecError::Unsupported {
                kind: req.kind,
                backend: self.backend_id(),
            });
        }
        if req.image.is_empty() {
            return Err(RecError::Invalid("image input is empty".to_owned()));
        }
        if self.cancelled() {
            return Err(RecError::Engine("cancelled".to_owned()));
        }
        let rgb = image::load_from_memory(&req.image)
            .map_err(|e| RecError::Invalid(format!("undecodable image: {e}")))?
            .to_rgb8();
        let (src_w, src_h) = (rgb.width() as f32, rgb.height() as f32);
        let (blob, scale, pad_x, pad_y) = letterbox(&rgb);
        let input =
            Tensor::from_array(([1usize, 3, INPUT_EDGE, INPUT_EDGE], blob.into_boxed_slice()))
                .map_err(|e| RecError::Engine(format!("face input tensor: {e}")))?
                .into_dyn();
        let mut session = self
            .session
            .as_ref()
            .ok_or(RecError::Unsupported {
                kind: req.kind,
                backend: self.backend_id(),
            })?
            .lock()
            .map_err(|e| RecError::Engine(format!("face session poisoned: {e}")))?;
        let extracted: Vec<(usize, usize, Vec<f32>, Vec<f32>, Option<Vec<f32>>)> = {
            let outputs = session
                .run(ort::inputs!["input.1" => input])
                .map_err(|e| RecError::Engine(format!("face inference: {e}")))?;
            let mut feats = Vec::with_capacity(3);
            for stride in [8usize, 16, 32] {
                let s = stride.to_string();
                let score_key = format!("score_{s}");
                let bbox_key = format!("bbox_{s}");
                let Some(score) = outputs.get(&score_key) else {
                    return Err(RecError::Engine(format!(
                        "face model missing `{score_key}` (expected SCRFD score/bbox heads)"
                    )));
                };
                let Some(bbox) = outputs.get(&bbox_key) else {
                    return Err(RecError::Engine(format!(
                        "face model missing `{bbox_key}` (expected SCRFD score/bbox heads)"
                    )));
                };
                let (score_shape, score_data) = score
                    .try_extract_tensor::<f32>()
                    .map_err(|e| RecError::Engine(format!("face `{score_key}`: {e}")))?;
                let (_, box_data) = bbox
                    .try_extract_tensor::<f32>()
                    .map_err(|e| RecError::Engine(format!("face `{bbox_key}`: {e}")))?;
                let score_cols = usize::try_from(*score_shape.last().unwrap_or(&1)).unwrap_or(1);
                let kps_vec: Option<Vec<f32>> = ["landmark_", "kps_"]
                    .iter()
                    .find_map(|prefix| outputs.get(&format!("{prefix}{s}")))
                    .and_then(|v| v.try_extract_tensor::<f32>().ok())
                    .map(|(_, d)| d.to_vec());
                feats.push((
                    stride,
                    score_cols,
                    score_data.to_vec(),
                    box_data.to_vec(),
                    kps_vec,
                ));
            }
            feats
        };
        drop(session);
        if self.cancelled() {
            return Err(RecError::Engine("cancelled".to_owned()));
        }
        let mut dets = Vec::new();
        for (stride, score_cols, score_data, box_data, kps_vec) in &extracted {
            decode_stride(
                score_data,
                *score_cols,
                box_data,
                kps_vec.as_deref(),
                *stride,
                &mut dets,
            );
        }
        let regions = nms(dets)
            .into_iter()
            .map(|d| {
                let map = |v: f32, pad: f32| ((v - pad) / scale).clamp(0.0, f32::MAX);
                let x1 = map(d.x1, pad_x).min(src_w);
                let y1 = map(d.y1, pad_y).min(src_h);
                let x2 = map(d.x2, pad_x).min(src_w);
                let y2 = map(d.y2, pad_y).min(src_h);
                let kps: Vec<[f32; 2]> = d
                    .kps
                    .chunks_exact(2)
                    .map(|p| [map(p[0], pad_x).min(src_w), map(p[1], pad_y).min(src_h)])
                    .collect();
                RecRegion {
                    bbox: [x1, y1, x2, y2],
                    // Faces carry no text; the bbox + landmarks + score are
                    // the detection.
                    text: String::new(),
                    confidence: Some(d.score),
                    landmarks: Some(kps),
                }
            })
            .collect::<Vec<_>>();
        let pages = if regions.is_empty() {
            Vec::new()
        } else {
            vec![RecPage {
                text: String::new(),
                regions,
            }]
        };
        Ok(RecognizeOutput {
            kind: RecKind::Face,
            engine: RecEngine {
                id: self.backend_id(),
                version: Some(FACE_ORT_VERSION),
                bbox_support: true,
            },
            pages,
        })
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anchor-free decode is deterministic: one hot cell decodes to the
    /// exact grid-center box (the compiled-build fixture contract).
    #[test]
    fn stride_decode_is_deterministic() {
        // Stride 8 over a 16×16 grid: 16*16*2 = 512 rows, single logit.
        let rows = 16 * 16 * 2;
        let mut scores = vec![-10.0f32; rows];
        // Cell (gx=2, gy=2) is row 2*16+2 = 34, second anchor.
        let hot = 34 * 2 + 1;
        scores[hot] = 10.0;
        let mut boxes = vec![0.0f32; rows * 4];
        boxes[hot * 4..hot * 4 + 4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let mut dets = Vec::new();
        decode_stride(&scores, 1, &boxes, None, 8, &mut dets);
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert_eq!(
            (d.x1, d.y1, d.x2, d.y2),
            (16.0 - 8.0, 16.0 - 16.0, 16.0 + 24.0, 16.0 + 32.0)
        );
        assert!(d.score > 0.999);
    }

    /// NMS keeps the hotter of two overlapping boxes.
    #[test]
    fn nms_keeps_hottest_overlap() {
        let a = RawDet {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            score: 0.9,
            kps: [0.0; 10],
        };
        let b = RawDet {
            x1: 1.0,
            y1: 1.0,
            x2: 10.0,
            y2: 10.0,
            score: 0.7,
            kps: [0.0; 10],
        };
        let kept = nms(vec![a, b]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].score, 0.9);
    }

    /// Missing weights stay absent: honest 501, never a fake detection.
    #[test]
    fn absent_engine_refuses_face() {
        let backend = FaceDetection::new(None);
        assert!(!backend.serves(RecKind::Face));
        let req = RecognizeRequest {
            kind: RecKind::Face,
            image: vec![1, 2, 3],
            language: None,
            mode: None,
            engine: None,
        };
        assert!(matches!(
            backend.recognize(&req),
            Err(RecError::Unsupported { .. })
        ));
    }
}
