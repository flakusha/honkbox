# TASK: Shared model library semantics

Status: proposed
Priority: P3
Epic: epic-external-tools
Owner: unassigned
Acceptance: library resolve/admission semantics defined for method, routing, conflicts, and dangling targets; user files never moved
Verification: `cargo test --locked -p honkbox-registry --lib shared_library`
Links: `.tmp/research/api-provisioning-extended-fields-2026-11-25.md` §2; `crates/registry/src/lib.rs`; `crates/llm-runtime/src/memory.rs`
Resources: per-test tmp dirs; fixture JSON; port 0; no shared state

## Change

- Semantics only, no package-manager build: `method` is `symlink` (unix data-dir default) vs `config-pointer` (JSON path ref where symlinks need privilege); never copy or move user files silently.
- Model-type routing: `<data>/Models/<category>/` where category is the registry scan root (reuse `comfy:`/`stt:`/`tts:`/`ocr:` roots).
- Collision: same digest dedupes/refcounts; same slug with different digest returns `409 slug-conflict` naming both digests; never silent overwrite.
- Dangling target (moved file/broken link) resolves as `410 gone` plus catalog `missing` flag; admission refuses with `404 model-missing`; no auto re-download.
- Hash sidecars `<target>.sha256.json` carry `{sha256, bytes, mtime, generator}`; derived-weight fingerprint schema is [UNKNOWN].
- Library root is read-only to consumers; writes only via the provisioning route.

Neural-network weight/tokenizer license and inference-use applicability are the responsibility of the operator; honkbox records digest/source and never clears, grants, or assumes a license.

## Acceptance criteria

- [ ] Symlink-vs-config-pointer method selected per platform without moving user files.
- [ ] Same-slug/different-digest returns 409 naming both digests; dangling target returns 410 with `missing` flag.
- [ ] Sidecar fields recorded; no verbatim checkpoint/weight filenames in code or tickets.
