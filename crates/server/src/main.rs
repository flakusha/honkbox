//! `honkbox` binary: serve the HTTP API + UI, or print the `OpenAPI` document.

mod admission;
mod backends;
mod catalog;
mod config;
mod downloads;
mod formats;
mod jobs;
mod jobs_registry;
mod logging;
mod metrics;
mod routes;
mod swap;

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use backends::{LlmBackend, SdBackend, SttBackend, TtsBackend};
use clap::{Parser, Subcommand};
use routes::{ApiDoc, AppConfig, app};
use tracing::info;
use utoipa::OpenApi as _;

/// `gc` TTL default: 7 days in seconds.
const GC_TTL_SECS: u64 = 7 * 24 * 60 * 60;
/// `gc` size-cap default: 4 GiB in bytes.
const GC_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Local-first generative studio.
#[derive(Debug, Parser)]
#[command(name = "honkbox", version, about = "Local-first generative studio")]
struct Cli {
    /// What to run.
    #[command(subcommand)]
    command: Command,
    /// Config file (`[models]`); defaults to `<data-dir>/honkbox.toml`.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Data dir root for jobs, logs, registry cache, starter models, and the
    /// config file. Beats `HONKBOX_DATA_DIR`; default
    /// `$XDG_STATE_HOME/honkbox` (fallback `$HOME/.local/state/honkbox`).
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
}

/// Subcommands: the only entry points of the binary.
#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the HTTP API + UI.
    Serve(ServeArgs),
    /// Print the `OpenAPI` document to stdout.
    Openapi,
    /// Inspect discovered model files.
    Models {
        /// What to do with the model list.
        #[command(subcommand)]
        action: ModelsAction,
    },
    /// Render an image into `<data-dir>/jobs/{ulid}/` (sync; picks `--backend`).
    Generate(GenerateArgs),
    /// Chat once via the LLM pipeline; prints `ChatResponse` JSON.
    Chat(ChatArgs),
    /// Transcribe a wav file; prints `Transcript` JSON.
    Transcribe(TranscribeArgs),
    /// Synthesize text to wav; writes `--out`, prints a JSON summary.
    Synthesize(SynthesizeArgs),
    /// Run job-bundle GC once: expire old bundles, enforce the size cap,
    /// sweep `*.part` and stale `$TMPDIR/honkbox-*.png` leftovers.
    Gc {
        /// Age (seconds) past which job bundles and temp files are removed.
        #[arg(long, default_value_t = GC_TTL_SECS)]
        max_age_secs: u64,
        /// Total size cap across job bundles (bytes); oldest evicted first.
        #[arg(long, default_value_t = GC_MAX_BYTES)]
        max_bytes: u64,
    },
}

/// `serve` flags (one struct: the handler takes a single argument).
#[derive(Debug, clap::Args)]
struct ServeArgs {
    /// Port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Bind address: an IP (`127.0.0.1`, `::1`, `0.0.0.0`, ...) or
    /// `localhost` (binds the IPv4 loopback). Non-loopback binds with
    /// `[server] auth = "none"` refuse to start unless
    /// `--allow-unauthenticated-bind` is passed (REQ-7, fail-closed).
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Explicit acknowledgement that a non-loopback bind without auth is
    /// intended (REQ-7 escape hatch; the warning-worthy choice).
    #[arg(long)]
    allow_unauthenticated_bind: bool,
    /// Serve the web-dist directory from disk (Vite dev flow).
    #[arg(long)]
    dev: bool,
    /// Directory served in dev mode.
    #[arg(long, default_value = "web-dist")]
    web_dist_dir: PathBuf,
    /// Log file (JSONL); falls back to `HONKBOX_LOG_FILE`, then
    /// `<data-dir>/logs/honkbox.log`.
    #[arg(long, env = "HONKBOX_LOG_FILE")]
    log_file: Option<PathBuf>,
    /// Chat backend (`stub` needs no model/GPU).
    #[arg(long, value_enum, default_value = "stub")]
    llm_backend: LlmBackend,
    /// Transcription backend (`stub` needs no model/GPU).
    #[arg(long, value_enum, default_value = "stub")]
    stt_backend: SttBackend,
    /// Synthesis backend (`stub` needs no model/GPU).
    #[arg(long, value_enum, default_value = "stub")]
    tts_backend: TtsBackend,
    /// Registry slug for the chat model (file backends only).
    #[arg(long)]
    llm_model: Option<String>,
    /// Registry slug for the transcription model (file backends only).
    #[arg(long)]
    stt_model: Option<String>,
    /// Registry slug for the synthesis model (file backends only).
    #[arg(long)]
    tts_model: Option<String>,
    /// Image backend (`stub` needs no model/GPU) serving
    /// `POST /v1/images/generations` (REQ-10).
    #[arg(long, value_enum, default_value = "stub")]
    sd_backend: SdBackend,
    /// Registry slug for the image checkpoint (file backends only).
    #[arg(long)]
    sd_model: Option<String>,
}

/// `generate` flags (one struct: the handler takes a single argument).
#[derive(Debug, clap::Args)]
struct GenerateArgs {
    /// Model slug from `models show`.
    #[arg(long)]
    model: String,
    /// Text prompt.
    #[arg(long)]
    prompt: String,
    /// RNG seed; same seed = byte-identical output on either backend (both deterministic).
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Diffusion steps.
    #[arg(long, default_value_t = 20)]
    steps: u32,
    /// Image size as `WxH`; longest edge capped at 2048.
    #[arg(long, default_value = "1024x1024")]
    size: String,
    /// Image container (`final_00` extension + sidecar `format`).
    #[arg(long, value_enum, default_value = "png")]
    format: Format,
    /// Which pipeline samples the image (`sd-cpp` decodes the real checkpoint).
    #[arg(long, value_enum, default_value = "sd-cpp")]
    backend: Backend,
    /// Text-encoder slug from `models show` (new archs: Qwen-VL-class file).
    #[arg(long)]
    clip: Option<String>,
    /// Standalone VAE slug from `models show`.
    #[arg(long)]
    vae: Option<String>,
}

/// `generate` output containers (clap-level; mapped to sd-core).
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum Format {
    Png,
    Jpeg,
    Webp,
}

impl From<Format> for honkbox_sd_core::OutputFormat {
    fn from(format: Format) -> Self {
        match format {
            Format::Png => Self::Png,
            Format::Jpeg => Self::Jpeg,
            Format::Webp => Self::Webp,
        }
    }
}

/// `generate` backends: instant stub pixels or the real sd.cpp pipeline.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum Backend {
    Stub,
    SdCpp,
}

/// `chat` flags: one prompt through the chat pipeline.
#[derive(Debug, clap::Args)]
struct ChatArgs {
    /// Model slug from `models show` (file backends only; stub ignores it).
    #[arg(long)]
    model: String,
    /// User prompt.
    #[arg(long)]
    prompt: String,
    /// Max completion tokens (default 512).
    #[arg(long)]
    max_tokens: Option<u32>,
    /// Sampling temperature (default 0.7).
    #[arg(long)]
    temperature: Option<f32>,
    /// Which pipeline answers (`stub` needs no model/GPU).
    #[arg(long, value_enum, default_value = "stub")]
    llm_backend: LlmBackend,
}

/// `transcribe` flags: wav file in, transcript JSON out.
#[derive(Debug, clap::Args)]
struct TranscribeArgs {
    /// Model slug from `models show` (file backends only; stub ignores it).
    #[arg(long)]
    model: String,
    /// Input `.wav` file (PCM16).
    #[arg(long)]
    input: PathBuf,
    /// Language hint (e.g. `en`).
    #[arg(long)]
    language: Option<String>,
    /// Which pipeline transcribes (`stub` needs no model/GPU).
    #[arg(long, value_enum, default_value = "stub")]
    stt_backend: SttBackend,
    /// Also write the transcript JSON to this file.
    #[arg(long)]
    out: Option<PathBuf>,
}

/// `synthesize` flags: text in, wav file + JSON summary out.
#[derive(Debug, clap::Args)]
struct SynthesizeArgs {
    /// Model slug from `models show` (file backends only; stub ignores it).
    #[arg(long)]
    model: String,
    /// Text to speak.
    #[arg(long)]
    text: String,
    /// Voice hint (backend-specific).
    #[arg(long)]
    voice: Option<String>,
    /// WAV output path.
    #[arg(long, default_value = "synthesize.wav")]
    out: PathBuf,
    /// Which pipeline synthesizes (`stub` needs no model/GPU).
    #[arg(long, value_enum, default_value = "stub")]
    tts_backend: TtsBackend,
    /// Output sample rate hint (backend may keep its native rate).
    #[arg(long)]
    sample_rate: Option<u32>,
}

/// Model-list actions.
#[derive(Debug, Subcommand)]
enum ModelsAction {
    /// Show discovered models as a table, or `--json` for the catalog.
    Show {
        /// Emit the `GET /v1/models` catalog JSON (`{object, data}`) instead of the human table.
        #[arg(long)]
        json: bool,
    },
}

/// P1 swap manager for the chat modality: load-on-demand + TTL/LRU
/// eviction, active only with the real llama backend (a stub server keeps
/// its single zero-cost stub pipeline; SD/audio stay startup-loaded — their
/// constructors and eviction costs differ). The eager startup load is
/// retained as a pre-warmed cache entry (the llama-swap `preload` analog):
/// the configured slug answers instantly, and idle TTL eviction frees it
/// like any other entry.
fn build_swap_manager(
    swap_cfg: &config::SwapConfig,
    llm_backend: backends::LlmBackend,
    catalog: &Arc<RwLock<Arc<honkbox_api_types::ModelCatalog>>>,
    llm: &Arc<dyn honkbox_llm_worker::LlmPipeline>,
    llm_model: Option<&str>,
) -> Option<Arc<swap::SwapManager>> {
    if !(swap_cfg.enabled && matches!(llm_backend, backends::LlmBackend::LlamaCpp)) {
        return None;
    }
    let manager = Arc::new(swap::SwapManager::new(
        swap::SwapSettings {
            llm_ttl: swap_cfg.llm_ttl,
            llm_max_loaded: swap_cfg.llm_max_loaded,
            max_ttl: swap_cfg.max_ttl,
            pinned: swap_cfg.pin.iter().cloned().collect(),
        },
        swap::catalog_llm_loader(Arc::clone(catalog)),
    ));
    if let Some(slug) = llm_model {
        manager.seed(slug, Arc::clone(llm));
    }
    // The reaper runs detached for the server's lifetime; the handle is
    // dropped on purpose (a missed idle sweep only delays a free, and the
    // process's exit tears the task down anyway).
    drop(swap::spawn_reaper(Arc::clone(&manager)));
    Some(manager)
}

#[tokio::main]
async fn main() {
    let Cli {
        command,
        config,
        data_dir,
    } = Cli::parse();
    match command {
        Command::Serve(args) => serve(args, config, data_dir).await,
        Command::Openapi => {
            let doc = ApiDoc::openapi();
            println!(
                "{}",
                serde_json::to_string_pretty(&doc).expect("serialize openapi")
            );
        }
        Command::Generate(args) => generate(args, data_dir),
        Command::Chat(args) => chat(args, data_dir),
        Command::Transcribe(args) => transcribe(args, data_dir),
        Command::Synthesize(args) => synthesize(args, data_dir),
        Command::Gc {
            max_age_secs,
            max_bytes,
        } => {
            let loaded = or_exit(config::load(config, data_dir));
            let data_dir = loaded.data_dir;
            logging::init(None, data_dir.as_deref());
            if let Some(dir) = data_dir.as_deref() {
                info!("honkbox data dir: {}", dir.display());
            }
            info!("honkbox gc max_age={max_age_secs}s max_bytes={max_bytes}");
            let jobs_root = data_dir.as_deref().map(|base| base.join("jobs"));
            jobs::gc_jobs(
                jobs_root.as_deref(),
                Duration::from_secs(max_age_secs),
                max_bytes,
            );
            println!("gc done");
        }
        Command::Models { action } => match action {
            ModelsAction::Show { json } => models_show(config, data_dir, json),
        },
    }
}

/// `serve`: load config (loud exit-1 on error), resolve the data dir, then
/// install logging, scan, and serve the HTTP API + UI.
/// Resolve the serve bind address + auth (REQ-7): parse `--host` into an
/// `IpAddr`, then fail-closed — `auth = "none"` on a non-loopback host is a
/// refused start unless the operator passed the explicit acknowledgement
/// flag. Bearer mode logs its env-var NAME (never any value).
fn resolve_bind(
    host: &str,
    port: u16,
    auth: config::AuthConfig,
    allow_unauthenticated_bind: bool,
) -> Result<(SocketAddr, config::AuthConfig), String> {
    let bind_ip: IpAddr = config::parse_bind_host(host)?;
    config::check_bind_auth(auth.mode, host, allow_unauthenticated_bind)?;
    if auth.mode == config::AuthMode::Bearer {
        info!(
            "bearer auth enabled (key read per request from {})",
            auth.api_key_env
        );
    }
    Ok((SocketAddr::new(bind_ip, port), auth))
}

/// Startup GC of job bundles + stale temp files, off the async runtime;
/// removals log at `debug!`.
fn spawn_startup_gc(jobs: &config::JobsConfig, data_dir: Option<&std::path::Path>) {
    let gc_ttl = jobs.gc_ttl;
    let gc_max_bytes = jobs.gc_max_bytes;
    let jobs_root = data_dir.map(|base| base.join("jobs"));
    tokio::task::spawn_blocking(move || {
        jobs::gc_jobs(jobs_root.as_deref(), gc_ttl, gc_max_bytes);
    });
}

/// Install the REQ-6 Option A admission queue from the resolved config and
/// log the operator-visible state (disabled = the pre-REQ-6 immediate 429).
fn install_admission_queue(queue_wait: Duration) {
    crate::admission::configure(queue_wait);
    if queue_wait.is_zero() {
        info!("GPU admission queue: disabled (busy requests 429 immediately)");
    } else {
        info!(
            "GPU admission queue: one waiter, {}s bound",
            queue_wait.as_secs_f64()
        );
    }
}

/// Install the resolved `[ledger]` settings onto the process-wide resource
/// ledger (TASK-resource-ledger-bounded-queue) and log the operator-visible
/// state. Called at serve startup, before the router accepts traffic. The
/// queue TTL is the SAME `[server] queue_wait_secs` knob the admission 429
/// names, so the ledger eviction clock and the "gave up waiting" message
/// agree; the queue caps come from `[ledger] queue_cap` (+ per-kind
/// `[ledger.queue_caps]`).
fn install_resource_ledger(ledger: &config::LedgerSettings, queue_wait: Duration) {
    honkbox_residency::global().configure(config::residency_config(ledger, queue_wait));
    if ledger.vram_budget_bytes == 0 {
        info!(
            "resource ledger: {} GPU slot(s), budgets unmanaged, queue cap {} per kind",
            ledger.gpu_slots, ledger.queue_cap
        );
    } else {
        info!(
            "resource ledger: {} GPU slot(s), VRAM budget {} B (headroom {} B), \
             queue cap {} per kind",
            ledger.gpu_slots, ledger.vram_budget_bytes, ledger.oom_headroom_bytes, ledger.queue_cap
        );
    }
}

#[allow(clippy::too_many_arguments)]
// Startup wiring reads long but stays linear: args → config → pipelines → router.
#[allow(clippy::too_many_lines)]
async fn serve(args: ServeArgs, config: Option<PathBuf>, data_dir_flag: Option<PathBuf>) {
    let ServeArgs {
        port,
        host,
        allow_unauthenticated_bind,
        dev,
        web_dist_dir,
        log_file,
        llm_backend,
        stt_backend,
        tts_backend,
        llm_model,
        stt_model,
        tts_model,
        sd_backend,
        sd_model,
    } = args;
    // Config loads BEFORE logging: the file-refined data dir relocates
    // the log too, and config-parse errors go to stderr via `or_exit`.
    let loaded = or_exit(config::load(config, data_dir_flag));
    let data_dir = loaded.data_dir;
    logging::init(log_file, data_dir.as_deref());
    if let Some(dir) = data_dir.as_deref() {
        info!("honkbox data dir: {}", dir.display());
    } else {
        info!("honkbox data dir: <unresolved>");
    }
    let cfg = loaded.models;
    // REQ-6 Option A: install the resolved admission-queue bound before any
    // route can acquire the GPU slot.
    install_admission_queue(loaded.server.queue_wait);
    // TASK-resource-ledger-bounded-queue: install the resolved ledger
    // capacities/bounds before any route can reserve capacity.
    install_resource_ledger(&loaded.ledger, loaded.server.queue_wait);
    spawn_startup_gc(&cfg.jobs, data_dir.as_deref());
    let (addr, auth) = or_exit(resolve_bind(
        &host,
        port,
        loaded.auth,
        allow_unauthenticated_bind,
    ));
    info!("honkbox serving on http://{addr} (dev: {dev})");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind address");
    // Clone the swap tunables out before `cfg` is partially moved into the
    // registry roots below.
    let swap_cfg = cfg.swap.clone();
    // VRAM budget for the curated catalog's fit pills (`Copy`).
    let vram_budget = cfg.vram_budget_bytes;
    // One registry scan feeds both slug resolution for file backends (stubs
    // never call `resolve`) and the `/v1/models` catalog. The roots stay in
    // `AppConfig` so `POST /v1/models/refresh` can re-scan them on demand.
    let roots = honkbox_registry::ModelRoots {
        llm_dirs: cfg.llm_dirs,
        comfyui_dirs: cfg.comfyui_dirs,
        stt_dirs: cfg.stt_dirs,
        tts_dirs: cfg.tts_dirs,
    };
    let found = honkbox_registry::scan_roots(&roots);
    info!("discovered {} models across configured roots", found.len());
    // Scan → cache → catalog (REQ-2): operator bindings are validated hard
    // (an alias conflict is a startup error), the SQLite cache stays
    // best-effort, and an empty scan serves the cached fallback catalog.
    // Same pipeline as `POST /v1/models/refresh`.
    let model_entries = cfg.entries.clone();
    let catalog = Arc::new(RwLock::new(Arc::new(or_exit(routes::refresh_catalog(
        &found,
        data_dir.as_deref(),
        &model_entries,
    )))));
    let resolve = |slug: &str| resolve_model(&found, slug);
    // Pipelines are built from the configured slugs before the strings move
    // into `AppConfig` (they are accepted as request `model` values alongside
    // the catalog).
    let llm = backends::make_llm(llm_backend, llm_model.as_deref(), &resolve);
    let stt = backends::make_stt(stt_backend, stt_model.as_deref(), &resolve);
    let tts = backends::make_tts(tts_backend, tts_model.as_deref(), &resolve);
    let sd = backends::make_sd(sd_backend, sd_model.as_deref(), &resolve);
    let swap = build_swap_manager(&swap_cfg, llm_backend, &catalog, &llm, llm_model.as_deref());
    // POST /v1/embeddings (REQ-4): load-once cache keyed by slug; the idle
    // reuse window reuses `[swap] max_ttl` (see `EmbedCache::catalog`).
    let embed = Arc::new(crate::swap::EmbedCache::catalog(swap_cfg.max_ttl));
    // Managed download supervisor: fetches into <data_dir>/models and fires
    // a rescan on every commit (stale *.part files are kept for resume; the
    // count is logged so leftovers stay visible).
    let models_root = data_dir.as_deref().map(|base| base.join("models"));
    let downloads = std::sync::Arc::new(
        crate::downloads::DownloadManager::new(models_root.clone()).expect("download manager"),
    );
    info!(
        stale = crate::downloads::count_stale_parts(models_root.as_deref()),
        "stale download *.part files (kept for resume)"
    );
    let app_config = AppConfig {
        dev,
        web_dist_dir,
        catalog,
        roots,
        llm_model,
        stt_model,
        tts_model,
        sd_model,
        llm,
        stt,
        tts,
        sd,
        swap,
        embed,
        swap_max_ttl: swap_cfg.max_ttl,
        data_dir,
        vram_budget,
        model_entries,
        downloads,
        auth,
    };
    let rescan_signal = app_config.downloads.rescan_signal();
    tokio::spawn(routes::rescan_on_downloads(
        app_config.clone(),
        rescan_signal,
    ));
    axum::serve(listener, app(app_config).into_make_service())
        .await
        .expect("serve app");
}

/// `models show`: load config (loud exit-1 on error), scan, print table/`--json`.
fn models_show(config: Option<PathBuf>, data_dir: Option<PathBuf>, json: bool) {
    let loaded = or_exit(config::load(config, data_dir));
    let cfg = loaded.models;
    let mut found = honkbox_registry::scan_roots(&honkbox_registry::ModelRoots {
        llm_dirs: cfg.llm_dirs,
        comfyui_dirs: cfg.comfyui_dirs,
        stt_dirs: cfg.stt_dirs,
        tts_dirs: cfg.tts_dirs,
    });
    found.sort_by(|a, b| a.slug.cmp(&b.slug));
    if json {
        // Same shape as `GET /v1/models` (P0 exit of the parity plan) with
        // the operator bindings applied (REQ-2); a binding conflict is the
        // same hard error the server exits on.
        let bindings = or_exit(routes::bind_entries(&cfg.entries, &found));
        println!(
            "{}",
            serde_json::to_string_pretty(&routes::build_catalog(&found, &bindings))
                .expect("serialize models")
        );
    } else {
        println!("{:<64} {:<6} {:>10}  PATH", "SLUG", "KIND", "SIZE");
        for m in &found {
            let kind = match m.root {
                honkbox_registry::RootKind::Llm => "llm",
                honkbox_registry::RootKind::Comfy => "comfy",
                honkbox_registry::RootKind::Stt => "stt",
                honkbox_registry::RootKind::Tts => "tts",
            };
            println!(
                "{:<64} {:<6} {:>10}  {}",
                m.slug,
                kind,
                human_size(m.bytes),
                m.path.display()
            );
        }
    }
}

/// `bytes` as B/KB/MB/GB with one decimal (bare bytes stay integral; integer math, no float).
fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    let (div, unit) = if bytes < KB {
        return format!("{bytes}B");
    } else if bytes < MB {
        (KB, "KB")
    } else if bytes < GB {
        (MB, "MB")
    } else {
        (GB, "GB")
    };
    format!("{}.{}{}", bytes / div, (bytes % div) * 10 / div, unit)
}
/// Registry slug → model path; unknown slug is a loud exit-1.
fn resolve_model(found: &[honkbox_registry::FoundModel], slug: &str) -> PathBuf {
    if let Some(m) = found.iter().find(|m| m.slug == slug) {
        m.path.clone()
    } else {
        eprintln!("error: unknown model {slug:?}; see `honkbox models show`");
        std::process::exit(1);
    }
}

/// Unwrap a fallible CLI step; `Err` is a loud exit-1 naming the cause.
fn or_exit<T, E: std::fmt::Display>(result: Result<T, E>) -> T {
    result.unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    })
}

/// `generate`: slug check, single-slot reserve, sample, bundle write.
///
/// Every failure is a loud exit-1 naming the cause; success prints the job dir.
fn generate(args: GenerateArgs, data_dir_flag: Option<PathBuf>) {
    let GenerateArgs {
        model,
        prompt,
        seed,
        steps,
        size,
        format,
        backend,
        clip,
        vae,
    } = args;
    let loaded = or_exit(config::load(None, data_dir_flag));
    let data_dir = loaded.data_dir;
    let cfg = loaded.models;
    logging::init(None, data_dir.as_deref());
    if let Some(dir) = data_dir.as_deref() {
        info!("honkbox data dir: {}", dir.display());
    }
    let started = std::time::Instant::now();
    info!(
        "honkbox generate model={model} seed={seed} steps={steps} size={size} backend={backend:?}"
    );
    #[cfg(feature = "vulkan")]
    if matches!(backend, Backend::SdCpp) && std::env::var_os("GGML_VK_VISIBLE_DEVICES").is_none() {
        eprintln!(
            "hint: GGML_VK_VISIBLE_DEVICES unset — sd.cpp uses Vulkan device 0 (on multi-GPU boxes this may be the iGPU: slow, and 1024² aborts with context-lost; export the dGPU index, e.g. 1)"
        );
    }
    let (width, height) = or_exit(jobs::parse_size(&size));
    let found = honkbox_registry::scan_roots(&honkbox_registry::ModelRoots {
        llm_dirs: cfg.llm_dirs,
        comfyui_dirs: cfg.comfyui_dirs,
        stt_dirs: cfg.stt_dirs,
        tts_dirs: cfg.tts_dirs,
    });
    let model_path = resolve_model(&found, &model);
    // Optional split-file inputs: same registry, unknown slug is a loud exit-1.
    let clip = clip.as_deref().map(|s| resolve_model(&found, s));
    let vae = vae.as_deref().map(|s| resolve_model(&found, s));
    let _lease = or_exit(honkbox_residency::global().try_reserve());
    let pipeline: Box<dyn honkbox_sd_core::SdPipeline> = match backend {
        Backend::Stub => Box::new(honkbox_sd_core::StubPipeline),
        Backend::SdCpp => Box::new(honkbox_sd_core::SdCppPipeline::new(model_path)),
    };
    let img = or_exit(pipeline.generate(&honkbox_sd_core::Txt2ImgRequest {
        prompt: prompt.clone(),
        seed,
        steps,
        width,
        height,
        model_slug: model.clone(),
        clip,
        vae,
        // CLI keeps the fixed recipe (no recipe flags on `generate`).
        sampler: None,
        scheduler: None,
        cfg_scale: None,
        negative_prompt: None,
        loras: None,
        init_image: None,
        mask_image: None,
        strength: None,
        hires: None,
        control_net: None,
        control_image: None,
        controlnet_strength: None,
    }));
    let format = honkbox_sd_core::OutputFormat::from(format);
    let bytes = or_exit(honkbox_sd_core::encode_image(img, format));
    let base = data_dir.unwrap_or_else(|| {
        eprintln!(
            "error: cannot resolve data directory (set --data-dir, $HONKBOX_DATA_DIR, or $HOME/$XDG_STATE_HOME)"
        );
        std::process::exit(1);
    });
    let dir = or_exit(jobs::write_bundle(
        &base,
        &jobs::BundleInputs {
            model_slug: model,
            prompt,
            seed,
            steps,
            width,
            height,
            format,
            backend_id: pipeline.backend_id(),
            sampler: honkbox_sd_core::SAMPLER.to_string(),
            scheduler: honkbox_sd_core::SCHEDULER.to_string(),
            cfg_scale: 7.0,
            negative_prompt: None,
            loras: Vec::new(),
            strength: None,
            recipe: None,
            hires: None,
        },
        &bytes,
        u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    ));
    println!("{}", dir.display());
}

/// Registry scan for file backends (stubs never reach here: no model/GPU).
fn scan_models(cfg: config::ModelConfig) -> Vec<honkbox_registry::FoundModel> {
    honkbox_registry::scan_roots(&honkbox_registry::ModelRoots {
        llm_dirs: cfg.llm_dirs,
        comfyui_dirs: cfg.comfyui_dirs,
        stt_dirs: cfg.stt_dirs,
        tts_dirs: cfg.tts_dirs,
    })
}

/// `chat`: one prompt through the chat pipeline; prints `ChatResponse` JSON.
///
/// Every failure is a loud exit-1 naming the cause.
fn chat(args: ChatArgs, data_dir_flag: Option<PathBuf>) {
    let loaded = or_exit(config::load(None, data_dir_flag));
    let cfg = loaded.models;
    logging::init(None, loaded.data_dir.as_deref());
    let ChatArgs {
        model,
        prompt,
        max_tokens,
        temperature,
        llm_backend,
    } = args;
    info!("honkbox chat model={model} backend={llm_backend:?}");
    let found = match llm_backend {
        LlmBackend::Stub => Vec::new(),
        LlmBackend::LlamaCpp => scan_models(cfg),
    };
    let resolve = |slug: &str| resolve_model(&found, slug);
    let pipe = backends::make_llm(llm_backend, Some(model.as_str()), &resolve);
    // Reserve BEFORE the model loads; the chat CLI is an `llm` job in the
    // ledger's fixed kind vocabulary.
    let _lease =
        or_exit(honkbox_residency::global().try_acquire(honkbox_residency::ResourceKind::Llm));
    let reply = or_exit(pipe.complete(&honkbox_llm_worker::ChatRequest {
        model,
        messages: vec![honkbox_llm_worker::ChatMessage {
            role: honkbox_llm_worker::Role::User,
            content: prompt,
            ..honkbox_llm_worker::ChatMessage::default()
        }],
        max_tokens,
        temperature,
        ..Default::default()
    }));
    println!(
        "{}",
        serde_json::to_string_pretty(&honkbox_api_types::ChatResponse {
            backend_id: reply.backend_id,
            text: reply.text,
            tokens: reply.tokens,
            millis: reply.millis,
        })
        .expect("serialize chat response")
    );
}

/// `transcribe`: decode the `<input>` wav, run the STT pipeline, print
/// `Transcript` JSON (`--out` also writes it to a file).
///
/// Every failure is a loud exit-1 naming the cause.
fn transcribe(args: TranscribeArgs, data_dir_flag: Option<PathBuf>) {
    let loaded = or_exit(config::load(None, data_dir_flag));
    let cfg = loaded.models;
    logging::init(None, loaded.data_dir.as_deref());
    let TranscribeArgs {
        model,
        input,
        language,
        stt_backend,
        out,
    } = args;
    info!("honkbox transcribe model={model} backend={stt_backend:?}");
    let bytes =
        or_exit(std::fs::read(&input).map_err(|err| format!("read {}: {err}", input.display())));
    let (sample_rate, pcm) = or_exit(honkbox_audio_worker::decode_wav(&bytes));
    let found = match stt_backend {
        SttBackend::Stub => Vec::new(),
        SttBackend::Whisper => scan_models(cfg),
    };
    let resolve = |slug: &str| resolve_model(&found, slug);
    let pipe = backends::make_stt(stt_backend, Some(model.as_str()), &resolve);
    // Reserve BEFORE the model loads. Audio pipelines share the `llm`
    // admission bucket (the fixed ledger vocabulary has no audio kind; see
    // `run_transcription` in routes.rs). The CLI takes the fast path only:
    // a busy slot exits loudly instead of queueing, as it always has.
    let _lease =
        or_exit(honkbox_residency::global().try_acquire(honkbox_residency::ResourceKind::Llm));
    let out_tx = or_exit(pipe.transcribe(
        &pcm,
        sample_rate,
        &honkbox_audio_worker::TranscribeRequest {
            model,
            language,
            translate: false, // the CLI transcribes; translations are HTTP-only
        },
    ));
    let transcript = honkbox_api_types::Transcript {
        backend_id: out_tx.backend_id,
        text: out_tx.text,
        segments: out_tx
            .segments
            .into_iter()
            .map(|segment| honkbox_api_types::TranscriptSegment {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text,
            })
            .collect(),
        millis: out_tx.millis,
    };
    let json = serde_json::to_string_pretty(&transcript).expect("serialize transcript");
    if let Some(path) = out {
        or_exit(
            std::fs::write(&path, &json).map_err(|err| format!("write {}: {err}", path.display())),
        );
    }
    println!("{json}");
}

/// `synthesize`: run the TTS pipeline, write WAV to `--out`, print a JSON summary.
///
/// Every failure is a loud exit-1 naming the cause.
fn synthesize(args: SynthesizeArgs, data_dir_flag: Option<PathBuf>) {
    let loaded = or_exit(config::load(None, data_dir_flag));
    let cfg = loaded.models;
    logging::init(None, loaded.data_dir.as_deref());
    let SynthesizeArgs {
        model,
        text,
        voice,
        out,
        tts_backend,
        sample_rate,
    } = args;
    info!("honkbox synthesize model={model} backend={tts_backend:?}");
    let found = match tts_backend {
        TtsBackend::Stub => Vec::new(),
        TtsBackend::Kokoro => scan_models(cfg),
    };
    let resolve = |slug: &str| resolve_model(&found, slug);
    let pipe = backends::make_tts(tts_backend, Some(model.as_str()), &resolve);
    let started = std::time::Instant::now();
    // Audio pipelines share the `llm` admission bucket (see the transcribe
    // CLI comment above); fast path only, a busy slot exits loudly.
    let _lease =
        or_exit(honkbox_residency::global().try_acquire(honkbox_residency::ResourceKind::Llm));
    let audio = or_exit(pipe.synthesize(&honkbox_audio_worker::SynthesizeRequest {
        model,
        text,
        voice,
        sample_rate,
    }));
    let millis = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let wav = honkbox_audio_worker::encode_wav(audio.sample_rate, &audio.pcm16);
    or_exit(std::fs::write(&out, &wav).map_err(|err| format!("write {}: {err}", out.display())));
    println!(
        "{}",
        serde_json::json!({
            "backend_id": pipe.backend_id(),
            "sample_rate": audio.sample_rate,
            "samples": audio.pcm16.len(),
            "millis": millis,
            "out": out,
        })
    );
}
