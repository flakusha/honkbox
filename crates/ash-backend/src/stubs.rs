//! Direct-Vulkan STT/TTS seams: real worker traits, no inference yet.
//!
//! Like [`crate::llm`], these are placeholders on the direct-Vulkan path,
//! NOT deterministic emulators: behind `experimental-ash-stt` /
//! `experimental-ash-tts` the types implement the `honkbox-audio-worker`
//! traits, and every verb returns a named `Model` error from the own
//! worker enum — no fake transcript text or audio is ever produced. The
//! module also builds under plain `experimental-ash-sd`, where the types
//! exist without the worker traits (reservation only, as before).

use crate::device::VulkanDevice;
use std::sync::Arc;

/// Named reason a direct-Vulkan STT/TTS seam cannot serve a request yet.
fn unimplemented(modality: &str) -> String {
    format!(
        "direct-Vulkan {modality} seam is not implemented yet: \
         the modality inference kernels land in later slices; no output is faked"
    )
}

/// Direct-Vulkan STT seam (backend id `ash-stt`).
///
/// Holds the selected execution device so later slices build kernels on it
/// without re-plumbing the server seam. With `experimental-ash-stt`, the
/// `SttPipeline` impl (feature `experimental-ash-stt`) fails every
/// transcription with `SttError::Model` naming the missing
/// capability — the pipeline has no model, so model-load failure is the
/// honest classification.
#[derive(Debug)]
pub struct AshSttPipeline {
    device: Arc<VulkanDevice>,
}

impl AshSttPipeline {
    /// Bind the execution device.
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        Self { device }
    }

    /// The device this seam will execute on.
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
}

#[cfg(feature = "experimental-ash-stt")]
impl honkbox_audio_worker::SttPipeline for AshSttPipeline {
    fn transcribe(
        &self,
        _pcm16: &[i16],
        _sample_rate: u32,
        _req: &honkbox_audio_worker::TranscribeRequest,
    ) -> Result<honkbox_audio_worker::Transcript, honkbox_audio_worker::SttError> {
        Err(honkbox_audio_worker::SttError::Model(unimplemented("stt")))
    }

    fn backend_id(&self) -> &'static str {
        "ash-stt"
    }
}

/// Direct-Vulkan TTS seam (backend id `ash-tts`).
///
/// Same posture as [`AshSttPipeline`]: with `experimental-ash-tts`, the
/// `TtsPipeline` impl fails every synthesis with `TtsError::Model` naming
/// the missing capability.
#[derive(Debug)]
pub struct AshTtsPipeline {
    device: Arc<VulkanDevice>,
}

impl AshTtsPipeline {
    /// Bind the execution device.
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        Self { device }
    }

    /// The device this seam will execute on.
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
}

#[cfg(feature = "experimental-ash-tts")]
impl honkbox_audio_worker::TtsPipeline for AshTtsPipeline {
    fn synthesize(
        &self,
        _req: &honkbox_audio_worker::SynthesizeRequest,
    ) -> Result<honkbox_audio_worker::SynthesizedAudio, honkbox_audio_worker::TtsError> {
        Err(honkbox_audio_worker::TtsError::Model(unimplemented("tts")))
    }

    fn backend_id(&self) -> &'static str {
        "ash-tts"
    }
}

// The seams sit behind `Send + Sync` pipeline traits; pin that contract on
// the stubs now so it cannot regress as fields grow.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AshSttPipeline>();
    assert_send_sync::<AshTtsPipeline>();
};

#[cfg(test)]
mod tests {
    use crate::VkError;
    use crate::device::VulkanDevice;
    use std::sync::Arc;

    /// One shared device: `None` on loader-less machines (tests skip), a
    /// panic on real API failures (those are bugs in this crate).
    fn test_device() -> Option<Arc<VulkanDevice>> {
        match VulkanDevice::headless() {
            Ok(dev) => Some(Arc::new(dev)),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                None
            }
            Err(e) => panic!("device creation failed: {e}"),
        }
    }

    #[cfg(feature = "experimental-ash-stt")]
    mod stt {
        use super::test_device;
        use crate::stubs::AshSttPipeline;
        use honkbox_audio_worker::{SttError, SttPipeline, TranscribeRequest};

        #[test]
        fn backend_id_is_ash_stt() {
            let Some(device) = test_device() else {
                return;
            };
            let pipeline = AshSttPipeline::new(device);
            assert_eq!(SttPipeline::backend_id(&pipeline), "ash-stt");
        }

        #[test]
        fn transcribe_returns_named_unimplemented_error() {
            let Some(device) = test_device() else {
                return;
            };
            let pipeline = AshSttPipeline::new(device);
            let err = pipeline
                .transcribe(
                    &[0, 1, 2, 3],
                    16_000,
                    &TranscribeRequest {
                        model: "test".to_string(),
                        language: None,
                        translate: false,
                    },
                )
                .expect_err("placeholder must not serve");
            assert!(matches!(err, SttError::Model(_)));
            let message = err.to_string();
            assert!(message.contains("not implemented"));
            assert!(message.contains("no output is faked"));
        }
    }

    #[cfg(feature = "experimental-ash-tts")]
    mod tts {
        use super::test_device;
        use crate::stubs::AshTtsPipeline;
        use honkbox_audio_worker::{SynthesizeRequest, TtsError, TtsPipeline};

        #[test]
        fn backend_id_is_ash_tts() {
            let Some(device) = test_device() else {
                return;
            };
            let pipeline = AshTtsPipeline::new(device);
            assert_eq!(TtsPipeline::backend_id(&pipeline), "ash-tts");
        }

        #[test]
        fn synthesize_returns_named_unimplemented_error() {
            let Some(device) = test_device() else {
                return;
            };
            let pipeline = AshTtsPipeline::new(device);
            let err = pipeline
                .synthesize(&SynthesizeRequest {
                    model: "test".to_string(),
                    text: "hi".to_string(),
                    voice: None,
                    sample_rate: None,
                })
                .expect_err("placeholder must not serve");
            assert!(matches!(err, TtsError::Model(_)));
            let message = err.to_string();
            assert!(message.contains("not implemented"));
            assert!(message.contains("no output is faked"));
        }
    }
}
