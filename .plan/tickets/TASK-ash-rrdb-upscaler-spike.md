# TASK: Spike RRDB and PixelShuffle shapes on Ash

Status: proposed
Priority: P2
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: synthetic tests prove one RRDB block and PixelShuffle geometry; an ignored test uses a user-supplied safetensors checkpoint to prove key/shape rejection and same-input repeatability without claiming a full upscaler.
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-sd rrdb_spike`, `HONKBOX_ASH_UPSCALER_MODEL=<path> HONKBOX_ASH_UPSCALER_MODEL_SHA256=<sha256> cargo test --locked -p honkbox-ash-backend --features ash-sd --test upscaler_spike -- --ignored --nocapture`, `bun scripts/check-ash-provenance.ts`.
Links: `.tmp/research/ash-promotion-refresh-2026-11-25.md` §3.5; `crates/ash-backend/src/conv2d.rs`; `crates/ash-backend/src/compute.rs`; https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py; https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html.
Resources: tiny deterministic synthetic input; unique temp output; opt-in GPU test in a separate process on one exclusive device; user checkpoint mounted read-only with recorded SHA-256/license; no download, cache mutation or bundled weights.

## Verified 2026-11-25

- **[FACT]** Ash has `Conv2d`, `ElementwiseAdd`, checkpoint shape validation
  and F32/F16/BF16 decode. It has no RRDB, LeakyReLU or PixelShuffle op, and
  `build.rs` registers no upscaler shader.
- **[FACT]** BasicSR RRDB uses three residual-dense blocks per RRDB, five
  convolutions per dense block, LeakyReLU slope 0.2 and residual scale 0.2.
  Real-ESRGAN's x4 path uses two nearest-neighbor 2x resizes plus convolutions;
  it does **not** use PixelShuffle.
- **[FACT]** PixelShuffle is an independent shape/gather contract:
  `[N,C*r*r,H,W] -> [N,C,H*r,W*r]`, requiring channel divisibility by `r*r`.
  It must not be presented as a Real-ESRGAN checkpoint component.
- **[UNKNOWN]** The prior note recorded local Real-ESRGAN-family files, but
  this pass did not open weights. The operator must supply the exact
  safetensors artifact, SHA-256 and license for the opt-in test.

## Dependencies

- Shared `Conv2d`, `ElementwiseAdd`, `CheckpointMap`, `load_tensor` and
  admission/error conventions.
- A user-supplied original-layout Real-ESRGAN safetensors checkpoint for the
  ignored real-weight test. No dependency on SD2 or HTTP Ash promotion.

## Change

- Add the minimal LeakyReLU(0.2) primitive required by one RRDB block; register
  any new shader in `build.rs` and give it an original/attributed
  `// Provenance:` line. Do not substitute SiLU/GELU.
- Compose one synthetic RRDB block from the existing convolution/add ops and
  compare with a CPU reference. Cover `[1,64,H,W]` shape preservation, dense
  channel concatenation, residual scale 0.2, wrong shapes and bad keys.
- Add a test-only PixelShuffle shape/gather reference for factors 2 and 4,
  including batch preservation and rejection of non-divisible channels. This
  spike does not add a production PixelShuffle route/op.
- Add an ignored `upscaler_spike` integration test. With
  `HONKBOX_ASH_UPSCALER_MODEL` and `HONKBOX_ASH_UPSCALER_MODEL_SHA256`, load
  `body.0.rdb{1,2,3}.conv{1..5}.*`, run one block twice on the same input, and
  compare output hashes. Wrong keys/shapes fail before dispatch.

## Acceptance criteria

- [ ] CPU/synthetic tests prove RRDB dense-block composition and LeakyReLU
  slope 0.2 against an independent reference using existing shared ops.
- [ ] PixelShuffle tests prove factors 2/4, NCHW geometry, batch preservation,
  channel-divisibility errors and a deterministic gather reference.
- [ ] The ignored checkpoint test verifies the supplied SHA-256, reports the
  model path/license supplied by the operator, and rejects wrong keys/shapes
  with named errors before dispatch.
- [ ] Two runs with identical checkpoint/input produce byte-identical output
  hashes; no seed or performance assumption is introduced.
- [ ] Results explicitly say “one RRDB block + independent PixelShuffle shape
  spike”; no full RRDBNet, image API, quality, latency or VRAM claim is made.

## Verification

```sh
cargo test --locked -p honkbox-ash-backend --features ash-sd rrdb_spike
HONKBOX_ASH_UPSCALER_MODEL=/read-only/model.safetensors \
HONKBOX_ASH_UPSCALER_MODEL_SHA256=<recorded-sha256> \
cargo test --locked -p honkbox-ash-backend --features ash-sd \
  --test upscaler_spike -- --ignored --nocapture
bun scripts/check-ash-provenance.ts
```

Run the ignored test alone on an idle GPU host. No command here is executed by
the filing pass.

## Non-goals

Full RRDBNet or model-zoo dispatch; nearest-resize production implementation;
tiling, face restoration, training, `.pth`/pickle loading, image upscale API,
catalog registration, automatic downloads, weight redistribution, benchmark
or quality claims.
