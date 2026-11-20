# TASK: CC0 pack fetch (PolyHaven API + Kenney/KayKit)

Status: done (2026-11-20, local-mirror fetch + sha/license rows landed)
Priority: P1
Epic: epic-game-engine-export
Owner: unassigned
Acceptance: a fetch job records sha + license rows for CC0 packs that loop-lore's attribution and dedup consume directly.
Verification: fixture fetch against local mirror; `bun run test:server`
(`arbitrary_source_url_is_refused_before_any_request` covers the SSRF boundary).
Links: `.tmp/research/shaders-materials-uiux-asset-pipeline-2026-11-20.md` §2–§3, `FEAT-game-engine-export-bundles.md`
Resources: per-test tmp dirs; no redistribution of fetched packs from honkbox

## Change

- Fetch-by-URL + sha + license record for PolyHaven (free API incl.
  commercial), Kenney (CC0) and KayKit (CC0, GLTF/FBX/OBJ) packs.
- Rows feed loop-lore `assets-attribution.md` + BLAKE3-dedup join keys via
  the bundle manifest; Quaternius QAL packs are EXCLUDED here (see
  `TASK-qal-guard.md`) — dev-only, never re-bundled/served.
- AmbientCG CC0 + credit snippet follows the same shape.
- Fetch URLs are pinned to the curated publisher origin or the operator
  mirror (`HONKBOX_ASSET_PACK_MIRROR`); redirects are disabled, so a caller
  cannot turn a pack fetch into an internal HTTP request.

## Acceptance detail

- CC0 needs no attribution row for use, but the fetch still records one
  (dedup + audit). CC BY sources record mandatory attribution text.
