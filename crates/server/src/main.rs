//! `sd-rip` binary: serve the HTTP API + UI, or print the `OpenAPI` document.

mod config;
mod jobs;
mod logging;
mod routes;

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use routes::{ApiDoc, AppConfig, app};
use tracing::info;
use utoipa::OpenApi as _;

/// Local-first generative studio.
#[derive(Debug, Parser)]
#[command(name = "sd-rip", version, about = "Local-first generative studio")]
struct Cli {
    /// What to run.
    #[command(subcommand)]
    command: Command,
    /// Config file (`[models]`); defaults to `~/.sd-rip/sd-rip.toml`.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
}

/// Subcommands: the only entry points of the binary.
#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the HTTP API + UI.
    Serve {
        /// Port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Serve the web-dist directory from disk (Vite dev flow).
        #[arg(long)]
        dev: bool,
        /// Directory served in dev mode.
        #[arg(long, default_value = "web-dist")]
        web_dist_dir: PathBuf,
        /// Log file (JSONL); falls back to `SD_RIP_LOG_FILE`, then `~/.sd-rip/logs/sd-rip.log`.
        #[arg(long, env = "SD_RIP_LOG_FILE")]
        log_file: Option<PathBuf>,
    },
    /// Print the `OpenAPI` document to stdout.
    Openapi,
    /// Inspect discovered model files.
    Models {
        /// What to do with the model list.
        #[command(subcommand)]
        action: ModelsAction,
    },
    /// Render an image into `~/.sd-rip/jobs/{ulid}/` (sync; stub pixels).
    Generate {
        /// Model slug from `models show`.
        #[arg(long)]
        model: String,
        /// Text prompt.
        #[arg(long)]
        prompt: String,
        /// RNG seed; same seed = byte-identical PNG from the stub backend.
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
    },
}

/// `generate` output containers (clap-level; mapped to sd-core).
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum Format {
    Png,
    Jpeg,
    Webp,
}

/// Model-list actions.
#[derive(Debug, Subcommand)]
enum ModelsAction {
    /// Show discovered models as a table, or `--json` for an array.
    Show {
        /// Emit `{slug, kind, bytes, path}` JSON instead of the human table.
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() {
    let Cli { command, config } = Cli::parse();
    match command {
        Command::Serve {
            port,
            dev,
            web_dist_dir,
            log_file,
        } => {
            logging::init(log_file);
            let addr = SocketAddr::from(([127, 0, 0, 1], port));
            info!("sd-rip serving on http://{addr} (dev: {dev})");
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .expect("bind address");
            axum::serve(
                listener,
                app(AppConfig { dev, web_dist_dir }).into_make_service(),
            )
            .await
            .expect("serve app");
        }
        Command::Openapi => {
            let doc = ApiDoc::openapi();
            println!(
                "{}",
                serde_json::to_string_pretty(&doc).expect("serialize openapi")
            );
        }
        Command::Generate {
            model,
            prompt,
            seed,
            steps,
            size,
            format,
        } => generate(model, prompt, seed, steps, &size, format),
        Command::Models { action } => match action {
            ModelsAction::Show { json } => models_show(config, json),
        },
    }
}

/// `models show`: load config (loud exit-1 on error), scan, print table/`--json`.
fn models_show(config: Option<PathBuf>, json: bool) {
    let cfg = match config::load(config) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let mut found = sd_rip_registry::scan_roots(&sd_rip_registry::ModelRoots {
        llm_dirs: cfg.llm_dirs,
        comfyui_dir: cfg.comfyui_dir,
    });
    found.sort_by(|a, b| a.slug.cmp(&b.slug));
    if json {
        let out: Vec<serde_json::Value> = found
            .iter()
            .map(|m| {
                serde_json::json!({
                    "slug": m.slug,
                    "kind": m.root,
                    "bytes": m.bytes,
                    "path": m.path,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&out).expect("serialize models")
        );
    } else {
        println!("{:<64} {:<6} {:>10}  PATH", "SLUG", "KIND", "SIZE");
        for m in &found {
            let kind = match m.root {
                sd_rip_registry::RootKind::Llm => "llm",
                sd_rip_registry::RootKind::Comfy => "comfy",
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
/// `generate`: slug check, single-slot reserve, stub sample, bundle write.
///
/// Every failure is a loud exit-1 naming the cause; success prints the job dir.
fn generate(model: String, prompt: String, seed: u64, steps: u32, size: &str, format: Format) {
    use sd_rip_sd_core::SdPipeline as _;

    logging::init(None);
    let started = std::time::Instant::now();
    info!("sd-rip generate model={model} seed={seed} steps={steps} size={size}");
    let cfg = match config::load(None) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let (width, height) = match jobs::parse_size(size) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let found = sd_rip_registry::scan_roots(&sd_rip_registry::ModelRoots {
        llm_dirs: cfg.llm_dirs,
        comfyui_dir: cfg.comfyui_dir,
    });
    if !found.iter().any(|m| m.slug == model) {
        eprintln!("error: unknown model {model:?}; see `sd-rip models show`");
        std::process::exit(1);
    }
    let _lease = match sd_rip_residency::global().try_reserve() {
        Ok(lease) => lease,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let pipeline = sd_rip_sd_core::StubPipeline;
    let img = match pipeline.generate(&sd_rip_sd_core::Txt2ImgRequest {
        prompt: prompt.clone(),
        seed,
        steps,
        width,
        height,
        model_slug: model.clone(),
    }) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let format = match format {
        Format::Png => sd_rip_sd_core::OutputFormat::Png,
        Format::Jpeg => sd_rip_sd_core::OutputFormat::Jpeg,
        Format::Webp => sd_rip_sd_core::OutputFormat::Webp,
    };
    let bytes = match sd_rip_sd_core::encode_image(img, format) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    match jobs::write_bundle(
        &jobs::BundleInputs {
            model_slug: model,
            prompt,
            seed,
            steps,
            width,
            height,
            format,
            backend_id: pipeline.backend_id(),
        },
        &bytes,
        u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    ) {
        Ok(dir) => println!("{}", dir.display()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
