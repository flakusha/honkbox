//! `sd-rip` binary: serve the HTTP API + UI, or print the `OpenAPI` document.

mod config;
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
