# Configuration

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

A ready-to-copy example lives in [`honkbox.example.toml`](../../honkbox.example.toml).
Copy it to `<data_dir>/honkbox.toml` and edit; every table is optional.

## Model roots / discovery

Lists; every root is scanned, exact duplicates dropped:

- `[models] llm_dirs` and `[models] comfyui_dirs` are lists. Legacy singular
  `[models] comfyui_dir` still works and is merged before `comfyui_dirs`.
- `[discovery] llm_dirs` / `comfyui_dirs` append extra roots; the registry
  already scans every root.
- Env overrides (path lists are path-separated; env wins over file):
  `HONKBOX_LLM_DIRS`, `HONKBOX_COMFYUI_DIR` (legacy),
  `HONKBOX_COMFYUI_DIRS` (plural), `HONKBOX_DISCOVERY_LLM_DIRS`,
  `HONKBOX_DISCOVERY_COMFYUI_DIRS`.
- HuggingFace conventions: default LLM root `~/.cache/huggingface` (i.e.
  `$HF_HOME`, or `$XDG_CACHE_HOME/huggingface`); a hub cache at
  `$HF_HUB_CACHE`, `$HF_HOME/hub`, or `$XDG_CACHE_HOME/huggingface/hub` is
  appended as an LLM root. The registry understands
  `hub/models--<owner>--<name>/snapshots/<sha>/...` and reports category
  `<owner>/<name>`; `hf download --local-dir` flat trees work under ordinary
  roots.

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
