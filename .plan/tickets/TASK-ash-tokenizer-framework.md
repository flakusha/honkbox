# TASK: Build the shared Ash tokenizer framework

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: one tokenizer interface serves CLIP and LLM profiles through shared BPE/Unigram/pre-tokenizer adapters, preserves declared special-token and chat-template policy, and proves exact IDs plus unknown/overlong behavior from synthetic and user-supplied assets.
Verification: `TMPDIR="$(mktemp -d "$PWD/.tmp/ash-tokenizer.XXXXXX")" cargo test --locked -p honkbox-ash-backend --features ash-compute tokenizer`; `TMPDIR="$(mktemp -d "$PWD/.tmp/ash-tokenizer-sd.XXXXXX")" cargo test --locked -p honkbox-ash-backend --features ash-sd tokenizer`; the opt-in golden-asset command below; `bun scripts/check-ash-provenance.ts`.
Links: `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S3; `crates/ash-backend/src/tokenizer.rs`; `crates/ash-backend/src/prompt.rs`; `crates/ash-backend/src/sd.rs`; `.tmp/research/ash-lfm-implementation-candidate-2026-11-25.md` §5; `.tmp/research/ash-gemma4-implementation-candidate-2026-11-25.md` §5; `.tmp/research/ash-qwen35plus-implementation-candidate-2026-11-25.md` §4.
Resources: in-memory synthetic vocabularies; unique gitignored `.tmp/ash-tokenizer.*` directories; read-only user tokenizer assets and golden manifest with recorded SHA-256/license; no GPU, checkpoint, download, bundled asset, port, database, or shared mutable state.

## Verified 2026-11-25

- **[FACT]** `tokenizer.rs` implements only caller-supplied CLIP BPE: lowercase and whitespace cleanup, contraction splitting, byte-to-unicode mapping, `</w>` merges, and fixed BOS/EOS IDs (`tokenizer.rs:1-11,49-184`).
- **[FACT]** `prompt::tokenizer_from` and `sd::generate` call `ClipTokenizer::encode` directly (`prompt.rs:17-29`; `sd.rs:366-372`). A shared interface must migrate these callers rather than create a parallel convention.
- **[FACT]** S3 requires one shared interface with BPE, Unigram, pre-tokenizer and chat-template policy; current Ash has only the CLIP path (`ash-family-implementation-index-2026-11-25.md:66`).
- **[FACT]** The LFM candidate requires GPT-2 byte BPE with the pinned llama3 Unicode split and `pre=lfm2`; Gemma requires GGUF vocabulary/model, pre-tokenizer and special IDs; Qwen3.5 requires BPE pre-splitting, special IDs and a chat template (candidate §§4–5).
- **[UNKNOWN]** Exact Gemma/Qwen model types, special IDs and template constructs remain checkpoint-specific until the user-supplied GGUF metadata is inspected. Unigram support is required by S3, but no candidate header has yet selected it.

## Dependencies

- `TASK-ash-native-families-index.md` owns S3 scope and ordering.
- `TASK-ash-gguf-reader.md` supplies bounded vocabulary arrays, tokenizer model/pre-tokenizer metadata, special IDs and embedded template text. This ticket adds no GGUF parser.
- Existing CLIP callers and tests are migration inputs, not a second tokenizer API.
- No dependency on decoder graphs, autoregressive state, sampling or family admission.

## Change

- Define one public tokenizer interface with `encode`, `decode` and `apply_chat_template`. Return IDs plus byte spans needed for lossless detokenization; use one typed error enum for malformed assets, unknown IDs, unsupported templates and length failures.
- Make `ClipTokenizer` one profile behind that interface and migrate `prompt`/`sd` callers. Do not retain a competing encode/decode convention.
- Share one byte-BPE core and one Unigram decoder behind explicit `lfm2`, `gemma` and `qwen35` profile adapters. Keep pre-tokenizers data-driven from validated metadata; never infer a family from a filename.
- Preserve CLIP behavior exactly. For LLM profiles, do not lowercase, normalize Unicode, collapse whitespace or reinterpret special-token-looking user text unless the selected profile explicitly requires it.
- Resolve special tokens from metadata, validate unique/contiguous IDs where required, and inject BOS/EOS only under explicit profile policy. Raw text and template-generated control tokens use separate paths.
- Define chat policy without a new dependency or arbitrary expression engine: accept validated built-in declarative templates for the selected profiles, render roles/content deterministically, and reject unsupported Jinja constructs, tools and media with named errors. Golden tests cover rendered text and resulting IDs.
- Define unknown policy per profile: byte-level fallback only for byte-BPE, declared `<unk>` for Unigram when present, otherwise a named error. Decode rejects out-of-range IDs and invalid byte sequences; it never emits replacement characters silently.
- Define length policy before allocation: caller supplies a token limit, over-limit input returns `TooLong { limit, actual }`, and truncation is never implicit. Empty input follows the profile's declared BOS/EOS behavior.
- Add tiny synthetic BPE/Unigram/pre-tokenizer fixtures for CI. Add one ignored integration test that consumes a user-supplied asset directory plus golden manifest, verifies every recorded SHA-256 and license field, and checks exact IDs/round trips for ASCII, Unicode, contractions, digits, CRLF, repeated spaces, specials, unknown text and the limit boundary.

## Acceptance criteria

- [ ] One interface owns encode/decode/template behavior; all current CLIP callers compile through it and existing CLIP IDs remain unchanged.
- [ ] Shared BPE and Unigram implementations pass deterministic synthetic tests, including rank/tie behavior and malformed vocabulary/model rejection.
- [ ] `lfm2`, `gemma` and `qwen35` adapters consume shared reader metadata and reject missing, duplicate, unsupported or mismatched tokenizer fields by name.
- [ ] Unknown and overlong inputs fail or map exactly as declared; no silent truncation, normalization, replacement character or byte fallback outside byte-BPE.
- [ ] The ignored golden-asset test verifies hashes/licenses and exact IDs from user-supplied assets; normal tests require no external asset.
- [ ] No family graph, persistent state, sampler, second tokenizer convention, dependency, bundled weight, availability claim or performance claim is introduced.

## Verification

```sh
mkdir -p .tmp
TMPDIR="$(mktemp -d "$PWD/.tmp/ash-tokenizer.XXXXXX")" \
  cargo test --locked -p honkbox-ash-backend --features ash-compute tokenizer
TMPDIR="$(mktemp -d "$PWD/.tmp/ash-tokenizer-sd.XXXXXX")" \
  cargo test --locked -p honkbox-ash-backend --features ash-sd tokenizer

HONKBOX_ASH_TOKENIZER_ASSET_DIR=/read-only/tokenizer-assets \
HONKBOX_ASH_TOKENIZER_GOLDENS=/read-only/tokenizer-goldens.json \
HONKBOX_ASH_TOKENIZER_MANIFEST_SHA256=<recorded-sha256> \
TMPDIR="$(mktemp -d "$PWD/.tmp/ash-tokenizer-goldens.XXXXXX")" \
  cargo test --locked -p honkbox-ash-backend --features ash-compute \
    --test tokenizer_goldens -- --ignored --nocapture --test-threads=1

bun scripts/check-ash-provenance.ts
```

The ignored test is CPU-only and uses read-only assets. A later real-checkpoint test, if any, must additionally acquire the exclusive single-device lease and record the checkpoint SHA-256/license before device work.

## Non-goals

Family decoder or embedding graphs; KV/recurrent state; token sampling; streaming; tools/grammar execution; multimodal tokenizers; model downloads or conversion; bundled vocabularies/weights; multi-GPU; HTTP/DTO changes; benchmark, quality, latency or availability claims.