# vendor_patches/ — patch stack for the fetched vendor/diffusion-rs tree

`vendor/` is gitignored scratch, never committed. `bun scripts/vendor-patches.ts sync`
converges it from the pins in `vendor.manifest.toml` (shallow clone at `rev`,
nested copies materialized inside, pruned dirs dropped), then applies these
patches in manifest order. A missing file or failed hunk is a loud error naming
the patch, never a silent skip. Re-running sync is safe: the applied prefix is
skipped. The escape hatch is `rm -rf vendor/` + sync — there is no `--force`.

## Add a patch

1. Edit the file under `vendor/`, then save just that change with ROOT-relative
   prefixes (`vendor/` itself is ignored, so plain `git diff` sees nothing):
   `git -C vendor/diffusion-rs diff --src-prefix=a/vendor/diffusion-rs/ --dst-prefix=b/vendor/diffusion-rs/ -- <path> > vendor_patches/NNNN-slug.patch`
   (a patch may also CREATE a file — keep its `/dev/null` pre-image — or
   DELETE one, like 0000 does. If the file is already touched by the stack,
   trim the diff by hand to just your new hunk.)
2. List it in `vendor.manifest.toml` (`patches = [..., "NNNN-slug.patch"]`).
3. `bun scripts/vendor-patches.ts sync` lands it — the worktree already
   contains your edit, so the patch reverse-checks and is skipped as applied.

## Move the base (bump a rev)

1. Put the new full SHA in `vendor.manifest.toml` (`rev` or a nested entry).
2. `rm -rf vendor/` + `bun scripts/vendor-patches.ts sync` (a rev move under
   a dirty tree is refused, so the from-scratch fetch doubles as the move flow).
3. If a patch's context drifted, regenerate it against the new tree and re-sync.
   0000 comes first: it adapts upstream sd.cpp to our C ABI.

## Dry run

`sync --dry-run` is network-free and mutation-free: it reports would-fetch /
would-move / would-apply / already-applied per entry and exits non-zero when
the tree is not converged.
