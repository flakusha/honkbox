# Configuration

## Local install

```sh
bun install
bun run install:local       # vendor sync + web build + release build, then install
honkbox serve               # via ~/.local/bin/honkbox
```

`install:local` places the binary at `$XDG_DATA_HOME/honkbox/bin/honkbox`
(default `~/.local/share/honkbox/bin/honkbox`) and creates the command link
`~/.local/bin/honkbox`. It refuses to replace a foreign command or symlink at
the binary path unless `--force` is passed. On Windows the binary is
`honkbox.exe`; the installer tries a symlink first, warns on failure, then
copies it into the bind directory.

Installer flags: `--prefix DIR`, `--bindir DIR` (absolute paths), `--dry-run`,
`--no-build`, `--force`. A marker in the install prefix identifies a fallback
copy. `bun run uninstall:local` removes only the installer-owned link or marked
copy; `--purge` also removes the install prefix, never the runtime state dir.

## Native build environment

The default Vulkan build needs `cmake`, a C++ compiler, `libclang` (bindgen),
Vulkan headers, `glslc`, and SPIR-V headers. On Debian/Ubuntu these come from
`cmake clang libvulkan-dev glslang-tools spirv-headers`.

Export one working `LIBCLANG_PATH` in every shell before running Cargo or Bun,
for example:

```sh
export LIBCLANG_PATH=/path/to/newest/installed/llvm/lib
```

Use the installed LLVM library directory on the host. Bindgen and native
rebuilds depend on this value; changing it can rebuild all of sd.cpp. If the
default LLVM mis-generates bindings (`_IO_FILE` size errors), point it at
another installed LLVM. Code generation and tests also build the native backend.
For multi-GPU Vulkan selection and the context-lost caveat, see
[media capabilities](media-capabilities.md#multi-gpu-vulkan).

## Runtime data dir

The whole on-disk state lives under ONE **data dir**, resolved as:

1. `--data-dir` global CLI flag
2. `HONKBOX_DATA_DIR` env
3. `[core] data_dir` in the config file
4. default: `$XDG_STATE_HOME/honkbox`, else `$HOME/.local/state/honkbox` (empty
   or relative XDG/env values are treated as unset)

Derived paths: `<data_dir>/honkbox.toml` (config), `<data_dir>/jobs/`,
`<data_dir>/logs/honkbox.log`, `<data_dir>/registry.sqlite3` (registry cache),
and starter model dirs `<data_dir>/models/{stt,tts}`. The startup log names the
active data dir. Model/discovery roots stay explicit (external weight trees);
temp PNGs stay in `$TMPDIR`.

To relocate an installed binary's runtime state without reinstalling it:

```sh
honkbox serve --data-dir /absolute/path/to/honkbox-state
```

The flag overrides `HONKBOX_DATA_DIR` and `[core] data_dir`. Because the config
file is located in one pass, `[core] data_dir` refines the derived state/model
roots but does not relocate the config file being read. Managed download roots
remain under the selected data dir as described below.

A ready-to-copy example lives in [`honkbox.example.toml`](../../honkbox.example.toml).
Copy it to `<data_dir>/honkbox.toml` and edit; every table is optional.

## Model roots / discovery

Each `[models]` root list accepts `llm_dirs`, `comfyui_dirs`, `stt_dirs`,
`tts_dirs`, and `ocr_dirs`. Legacy singular `[models] comfyui_dir` still works
and is merged before `comfyui_dirs`. `[discovery]` provides matching extra
root lists appended after the base roots. Precedence is built-in defaults →
config file → env overrides; path lists use the platform path separator, and
`[discovery]` extras are appended after base roots.

| Root | Base config | Discovery config | Base env | Discovery env |
| --- | --- | --- | --- | --- |
| LLM | `[models] llm_dirs` | `[discovery] llm_dirs` | `HONKBOX_LLM_DIRS` | `HONKBOX_DISCOVERY_LLM_DIRS` |
| ComfyUI | `[models] comfyui_dirs` | `[discovery] comfyui_dirs` | `HONKBOX_COMFYUI_DIRS` | `HONKBOX_DISCOVERY_COMFYUI_DIRS` |
| STT | `[models] stt_dirs` | `[discovery] stt_dirs` | `HONKBOX_STT_DIRS` | `HONKBOX_DISCOVERY_STT_DIRS` |
| TTS | `[models] tts_dirs` | `[discovery] tts_dirs` | `HONKBOX_TTS_DIRS` | `HONKBOX_DISCOVERY_TTS_DIRS` |
| OCR | `[models] ocr_dirs` | `[discovery] ocr_dirs` | `HONKBOX_OCR_DIRS` | `HONKBOX_DISCOVERY_OCR_DIRS` |

`HONKBOX_COMFYUI_DIR` is the legacy singular ComfyUI env override. Every
configured root is scanned and exact duplicates are dropped. Without an
explicit STT/TTS file setting, starter roots derive from the final data dir as
`<data_dir>/models/stt` and `<data_dir>/models/tts`; explicit file roots replace
those starters. The managed download roots are appended independently:
`<data_dir>/models/{llm,sd,stt,tts,ocr}` (SD downloads use
`models/sd/checkpoints`). They remain scannable even when other roots are
configured, so completed downloads appear in `GET /v1/models`; exact duplicates
are removed.

HuggingFace conventions: default LLM root `~/.cache/huggingface` (i.e.
`$HF_HOME`, or `$XDG_CACHE_HOME/huggingface`); a hub cache at
`$HF_HUB_CACHE`, `$HF_HOME/hub`, or `$XDG_CACHE_HOME/huggingface/hub` is
appended as an LLM root. The registry understands
`hub/models--<owner>--<name>/snapshots/<sha>/...` and reports category
`<owner>/<name>`; `hf download --local-dir` flat trees work under ordinary
roots.

## Curated-model VRAM budget

`[models] vram_budget_bytes` sets the curated catalog's VRAM budget ceiling;
`HONKBOX_VRAM_BUDGET` overrides it with a byte count. An absent setting leaves
the budget unset. Empty or invalid env values retain the file/default value.
The loader clamps a model's context/cache size to this ceiling.

## Queue bound

`HONKBOX_QUEUE_WAIT_SECS` overrides `[server] queue_wait_secs`.

## Attached `llama-server`

`--llm-backend attached` consumes the server at
`[tools] llama_server_url` (loopback only; `/health` and `/v1/models` are
proved at startup). Its process stays operator-owned. `LLAMA_API_KEY`, when
set, is sent only to that server; honkbox's own bearer is never forwarded.

## Auth (`[server] auth`, REQ-7)

With `auth = "bearer"`, every route except
`GET /health`, `GET /metrics`, and the static web UI requires
`Authorization: Bearer <key>`; failures return a 401 with an OpenAI-shaped
error body. The key VALUE is read at request time from the env var named by
`[auth] api_key_env` (rotation needs no restart) and is never logged.
Fail-closed startup: `auth = "none"` with a non-loopback `--host`
(`127.0.0.0/8`, `::1`, `localhost` are loopback) refuses to start unless
`honkbox serve --allow-unauthenticated-bind` is passed.

Signed URLs are file-only configuration: `[auth] signed_url_secret_env` names
the env var holding the HMAC secret (default `HONKBOX_SIGNED_URL_SECRET`), and
`signed_url_ttl_secs` bounds their lifetime (default 300 seconds). The config
stores the env-var name, never the secret value.

`[[auth.scopes]]` grants scopes to API keys. Each entry has `api_key_env` (the
NAME of the key env var) and a `scopes` list; duplicate `api_key_env` entries
are de-duplicated. Empty scope strings are removed and only names are stored.

The additive JWT bearer arm is enabled by either `[auth] jwt_secret_env` (the
env-var name holding an HS256 secret) or `[auth] jwt_jwks_url` (an HTTP JWKS
URL for RS256 with `kid` selection). `jwt_issuer` and `jwt_audience` pin the
expected `iss` and `aud` claims when set; `jwt_clock_skew_secs` allows `exp`/`nbf`
leeway (default 60 seconds; `0` requires exact timing). Signature, expiry,
issuer, and audience verification fail closed. HS256 secrets are re-read per
request; JWKS keys refresh through a TTL cache. With neither key source
configured, the arm is off and the existing env-key bearer path is unchanged.

## Logs

`RUST_LOG` (default `info,honkbox_registry=warn`); `--log-file <path>` >
`HONKBOX_LOG_FILE` (empty counts as unset; read once, in `logging::init`) >
`<data_dir>/logs/honkbox.log` (stderr human-readable, file JSONL).

## Registry cache

`HONKBOX_REGISTRY_DB` (path; `off`/empty disables) >
`<data_dir>/registry.sqlite3`.

## Busy slot (REQ-6)

When one job holds the GPU slot, the NEXT request
waits up to `[server] queue_wait_secs` for it to free (a waiting client
sees latency, not an error); any further concurrent request gets the
immediate 429 (`queue_full`) and a waiter past its bound gets 429
(`queue_timeout`), both with a `Retry-After` hint. `0` restores the old
fail-fast 429 for every request.
