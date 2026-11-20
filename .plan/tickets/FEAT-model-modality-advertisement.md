# FEAT: per-model modality advertisement

Status: done
Priority: P2
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: `/v1/models` entries carry input/output modalities (chat, vision, audio, decisions, rerank, image) and `/v1/capabilities` flags agree with the live catalog; a client can route without probing routes or parsing error codes.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run codegen:check`
Links: `FEAT-capabilities-api` (server flags); llama.cpp `/v1/models` `architecture{input_modalities, output_modalities}` incl. `decisions`/`audio` markers; llama-swap `capabilities{in, out, reranker}` + `disableAuto` discovery
Resources: per-test config + temp catalog dirs

## Change

- Extend the model entry DTO with `modalities{in: [...], out: [...]}` derived from catalog scan (file kind: checkpoint/GGUF capabilities, LoRA, TTS voice, decision template marker) — never README prose.
- `FeatureFlags` gains `decisions` and `rerank` (default false until FEAT-decision-systemone-endpoint / FEAT-rerank-endpoint land); `video` stays false per the existing contract.
- Refresh on `/v1/models/refresh` and download commit, same as existing catalog-derived counts.

## Acceptance detail

Contract tests pin one entry per model kind (chat, vision, decision, reranker, image) and the flags/catalog agreement; unknown kinds degrade to empty modality lists, not errors.

## Verification Notes (2026-11-20)

The legacy flat `modalities` list is unchanged; the additive nested field is
`modality: {input, output}` (content-kind vocabulary) because the flat list's
`vision` value is a different vocabulary used by existing loop-lore consumers.
`FeatureFlags.decisions` is present and false until the endpoint lands;
`rerank` already existed. One contract entry per model kind plus the
flat-list/flag agreement is asserted. Full `bun run test` and `bun run check`
pass, including regenerated TS/OpenAPI.
