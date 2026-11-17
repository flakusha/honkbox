# TASK: Opt-in remote image URLs for vision

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: an opt-in `vision.remote_image_urls` config fetches allowlisted `http(s)` image URLs safely and advertises `vision.remote_urls` in `/v1/capabilities`; default stays data-URI-only.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedExpansion-RemoteVisionURLs.md` §Recommended design
Resources: local HTTP fixture server on an assigned port; no real external network

## Change

- Default off. Allowlist exact hosts or CIDRs; deny loopback, private,
  link-local, CGNAT, and cloud-metadata ranges, including after redirects.
- Resolve and pin the address; re-check on every redirect to avoid DNS rebinding.
- Cap body at 10 MB, timeout at 10 s, allow only `image/png|jpeg|webp`, and
  reject `Content-Encoding` chains that bypass the size check.
- Reuse the existing data-URI decoder; hash fetched bytes into the job sidecar.
- Named errors: `image_url_host_not_allowed`, `image_url_too_large`,
  `image_url_timeout`.

## Acceptance detail

Tests cover private host, oversized body, bad MIME, redirect to a denied host,
and one successful allowlisted fetch. No generic HTTP client surface or `file://`.
