# TASK: Audio/LLM residency slot-contention policy

Status: proposed
Priority: P2
Epic: epic-audio-streaming
Owner: unassigned
Acceptance: STT/TTS requests queue behind an active LLM lease and vice versa under a documented policy, with queue order observable in metrics instead of silent latency.
Verification: `cargo test --locked -p honkbox-audio-worker`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/cuda-strata-inference-2026-11-25.md` §7.4, `TASK-llm-prompt-cache.md`, `crates/audio-worker/src/lib.rs`, `crates/residency/src/lib.rs`
Resources: `StubStt`/`StubTts` holding the lease; no audio hardware, no model downloads

## Attribution

Coexistence observation from the Qwen3.x weight-family overlap across `Niko1221/Strata`, `ValerioDolci/capy-flash`, and honkbox audio.cpp Qwen3-ASR (`AUDIOCPP_STT_FAMILIES`). Method: white-room. Scheduling policy only; no weight sharing (chat GGUF and ASR weights are different artifacts) and no code sharing (llama.cpp vs audio.cpp stay separate backends).

## Change

- Document the slot-contention policy: STT/TTS queue behind an active LLM lease and vice versa; queue order is observable via metrics.
- Contention-policy tests use stub backends holding the lease; no audio hardware needed.
- No sharing of weights, contexts, or caches between the LLM and audio backends.

## Acceptance criteria

- [ ] Contended STT/TTS/LLM ordering matches the documented policy in stub tests.
- [ ] Queue depth and wait are observable in bounded metrics; no silent latency path remains.
- [ ] No audio hardware, checkpoint, download, or new dependency in any test.
