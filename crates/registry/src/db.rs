//! SQLite-backed cache of the file-walk catalog (`db`).
//!
//! The file walk (`crate::catalog`) remains the **source of truth**; this
//! module is an incremental cache for offline reads only. After a successful
//! scan the caller upserts the fresh entries and prunes rows whose files no
//! longer exist. When roots are unavailable the server may fall back to
//! `load_entries` so `GET /v1/models` still reports the last-known catalog.
//!
//! Operator model bindings (REQ-2) round-trip the same way so a restart with
//! empty roots still serves cached aliases + capability overrides:
//! `sync_bindings` replace-all-writes the `aliases` (alias→slug) and
//! `entry_overrides` (per-slug kind + capabilities JSON) tables from the
//! server's operator config after a successful scan; `load_bindings` reads
//! them back for the offline fallback. The tables are caches of the operator
//! config — never derived from filenames. Orphan alias rows (slug pruned by
//! `prune_missing_roots`) are skipped by the caller when the slug has no
//! cached entry.
//!
//! DB path resolution: `HONKBOX_REGISTRY_DB` (absolute or relative path);
//! the values `"off"` and `""` disable persistence entirely (callers then
//! keep the exact pre-cache behavior). Unset falls back to a caller-supplied
//! default (the server passes `<data_dir>/registry.sqlite3`), so this crate
//! stays std-only and platform-neutral (no raw `HOME` read, which is unset
//! on Windows).
//!
//! Cached rows store the catalog identity columns only — `metadata`
//! (`GGUF`/safetensors headers) is re-probed from live files on the next real
//! scan and is `None` for cached entries. All failures are best-effort: the
//! cache never panics and never fails a scan; a corrupted DB file is deleted
//! and recreated on the next open (logged, not fatal).
//!
//! Follow-up (out of scope here): hf-hub integration for remote catalog
//! sources (network, user-agent, rate limits).

use std::path::{Path, PathBuf};

use rusqlite::Connection;

/// Current schema version of the `entries`/`aliases`/`entry_overrides` layout.
pub const SCHEMA_VERSION: u32 = 2;

/// Env var selecting the cache DB path; `"off"` or `""` disables persistence.
pub const DB_ENV: &str = "HONKBOX_REGISTRY_DB";

/// Cache errors. Best-effort by contract: callers log and continue.
#[derive(Debug)]
pub enum DbError {
    /// SQLite failure (open, schema setup, or statement execution).
    Sqlite(rusqlite::Error),
    /// Filesystem failure (creating the DB parent dir, removing a corrupted
    /// file).
    Io(std::io::Error),
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DbError::Sqlite(e) => write!(f, "registry db: {e}"),
            DbError::Io(e) => write!(f, "registry db io: {e}"),
        }
    }
}

impl std::error::Error for DbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DbError::Sqlite(e) => Some(e),
            DbError::Io(e) => Some(e),
        }
    }
}

impl From<rusqlite::Error> for DbError {
    fn from(e: rusqlite::Error) -> Self {
        DbError::Sqlite(e)
    }
}

impl From<std::io::Error> for DbError {
    fn from(e: std::io::Error) -> Self {
        DbError::Io(e)
    }
}

/// Cache result alias.
pub type DbResult<T> = Result<T, DbError>;

/// Test/embedder override for the cache DB path.
enum Override {
    /// Normal resolution: env var, else the caller-supplied default.
    Auto,
    /// Forced resolution: `Set(None)` disables persistence outright (no
    /// env, no default read); `Set(Some(path))` forces the path.
    Set(Option<PathBuf>),
}

static DB_OVERRIDE: std::sync::RwLock<Override> = std::sync::RwLock::new(Override::Auto);

/// Override the cache DB resolution (see [`Override`] docs). Idempotent;
/// thread-safe.
pub fn set_db_override(path: Option<PathBuf>) {
    if let Ok(mut guard) = DB_OVERRIDE.write() {
        *guard = Override::Set(path);
    }
}

/// Pure core of [`env_db_path`] (testable without env mutation).
/// An unset `HONKBOX_REGISTRY_DB` falls back to `default`; a relative or
/// empty default is ignored (never a relative path in an arbitrary cwd).
fn db_path_from(value: Option<std::ffi::OsString>, default: Option<PathBuf>) -> Option<PathBuf> {
    match value {
        Some(v) if v == "off" || v.is_empty() => None,
        Some(v) => Some(PathBuf::from(v)),
        None => default.filter(|p| p.is_absolute()),
    }
}

/// Resolve the cache DB path from the environment.
///
/// A [`set_db_override`] value wins. Returns `None` when persistence is
/// disabled (`HONKBOX_REGISTRY_DB` is `"off"` or empty, or no usable
/// default was supplied); otherwise the configured path or `default`.
#[must_use]
pub fn env_db_path(default: Option<PathBuf>) -> Option<PathBuf> {
    if let Ok(guard) = DB_OVERRIDE.read()
        && let Override::Set(path) = &*guard
    {
        return path.clone();
    }
    db_path_from(std::env::var_os(DB_ENV), default)
}

/// Handle to the registry cache DB (WAL mode, schema-validated on open).
pub struct Db {
    conn: Connection,
}

/// DDL shared by bootstrap and version migration.
const SCHEMA_SQL: &str = "CREATE TABLE IF NOT EXISTS meta (
     key TEXT PRIMARY KEY,
     value TEXT NOT NULL
 ) WITHOUT ROWID;
 CREATE TABLE IF NOT EXISTS entries (
     slug TEXT PRIMARY KEY,
     kind TEXT NOT NULL,
     category TEXT NOT NULL,
     path TEXT NOT NULL,
     bytes INTEGER NOT NULL,
     size_ts INTEGER NOT NULL
 );
 CREATE TABLE IF NOT EXISTS aliases (
     alias TEXT PRIMARY KEY,
     slug TEXT NOT NULL
 );
 CREATE TABLE IF NOT EXISTS entry_overrides (
     slug TEXT PRIMARY KEY,
     capabilities TEXT NOT NULL
 );";

/// One cached alias binding: `alias` resolves to the entry with `slug`
/// (operator-declared; never filename-derived). Rows are stored per slug in
/// declaration order so the first row of a slug is the served `id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasRow {
    /// The operator-declared request alias.
    pub alias: String,
    /// Registry slug owning the alias.
    pub slug: String,
}

/// One cached per-entry override: the operator capability declaration as a
/// JSON object (`{"chat":false,…}`; only the declared keys are present).
/// Parsed by the caller — the cache stays schema-agnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryOverride {
    /// Registry slug the override applies to.
    pub slug: String,
    /// Serialized [`crate`] caller capability declaration.
    pub capabilities_json: String,
}

impl Db {
    /// Open (creating if missing) the cache DB at `path`.
    ///
    /// Parent dirs are created as needed. A corrupted DB file is removed
    /// (plus its `-wal`/`-shm` siblings) and recreated with a warning —
    /// opening never fails hard on a bad cache file. A foreign or older
    /// schema version is dropped and recreated.
    pub fn open(path: &Path) -> DbResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        match Self::open_valid(path) {
            Ok(db) => Ok(db),
            Err(e) => {
                tracing::warn!(%e, path = %path.display(), "registry cache db unusable; recreating");
                for suffix in ["", "-wal", "-shm"] {
                    let mut p = path.as_os_str().to_owned();
                    p.push(suffix);
                    match std::fs::remove_file(Path::new(&p)) {
                        Ok(()) | Err(_) => {}
                    }
                }
                Self::open_valid(path)
            }
        }
    }

    /// Open the DB from [`env_db_path`]; `Ok(None)` when persistence is
    /// disabled (callers keep the exact no-cache behavior). `default` is
    /// the caller's `<data_dir>/registry.sqlite3` (absolute; see the module
    /// docs).
    pub fn open_from_env(default: Option<&Path>) -> DbResult<Option<Self>> {
        env_db_path(default.map(Path::to_path_buf)).map_or(Ok(None), |p| Self::open(&p).map(Some))
    }

    /// Raw open + schema bootstrap; errors distinguish corruption for [`Db::open`].
    fn open_valid(path: &Path) -> DbResult<Self> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(SCHEMA_SQL)?;
        let version = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |r| r.get::<_, String>(0),
            )
            .map(Some)
            .or_else(|e| {
                if e == rusqlite::Error::QueryReturnedNoRows {
                    Ok(None)
                } else {
                    Err(e)
                }
            })
            .map_or(None, |v| v.and_then(|s| s.parse::<u32>().ok()));
        if version != Some(SCHEMA_VERSION) {
            // From-empty bootstraps here (no version row); older/foreign
            // schemas are dropped wholesale — the cache is rebuildable.
            conn.execute_batch(&format!(
                "DROP TABLE IF EXISTS entries; DROP TABLE IF EXISTS aliases; \
                 DROP TABLE IF EXISTS entry_overrides; DROP TABLE IF EXISTS meta; {SCHEMA_SQL}\
                 INSERT OR REPLACE INTO meta VALUES ('schema_version', '{SCHEMA_VERSION}');"
            ))?;
        }
        Ok(Self { conn })
    }

    /// Upsert catalog entries keyed by slug, stamping `now_ts` as the size
    /// observation time. The whole batch is one transaction.
    pub fn upsert_entries(&self, entries: &[crate::CatalogEntry], now_ts: u64) -> DbResult<()> {
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO entries (slug, kind, category, path, bytes, size_ts)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(slug) DO UPDATE SET
                     kind = excluded.kind,
                     category = excluded.category,
                     path = excluded.path,
                     bytes = excluded.bytes,
                     size_ts = excluded.size_ts",
            )?;
            for e in entries {
                stmt.execute((
                    e.slug.as_str(),
                    kind_str(e.kind),
                    e.category.as_str(),
                    e.path.to_string_lossy().as_ref(),
                    i64::try_from(e.bytes).unwrap_or(i64::MAX),
                    i64::try_from(now_ts).unwrap_or(i64::MAX),
                ))?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Load all cached entries (slug-sorted; `metadata` is `None` — see the
    /// module docs). An empty cache yields an empty vector.
    pub fn load_entries(&self) -> DbResult<Vec<crate::CatalogEntry>> {
        let mut stmt = self
            .conn
            .prepare("SELECT slug, kind, category, path, bytes FROM entries ORDER BY slug")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (slug, kind, category, path, bytes) = row?;
            out.push(crate::CatalogEntry {
                slug,
                kind: parse_kind(&kind),
                category,
                path: PathBuf::from(path),
                bytes: u64::try_from(bytes.max(0)).unwrap_or(0),
                metadata: None,
            });
        }
        Ok(out)
    }

    /// Replace-all write of the operator bindings: the `aliases` and
    /// `entry_overrides` tables mirror the caller's operator config after a
    /// successful scan, so the last-known bindings survive a restart with
    /// empty roots. One transaction: the pair is always consistent.
    pub fn sync_bindings(&self, aliases: &[AliasRow], overrides: &[EntryOverride]) -> DbResult<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute_batch("DELETE FROM aliases; DELETE FROM entry_overrides;")?;
        {
            let mut stmt = tx.prepare("INSERT INTO aliases (alias, slug) VALUES (?1, ?2)")?;
            for row in aliases {
                stmt.execute((row.alias.as_str(), row.slug.as_str()))?;
            }
        }
        {
            let mut stmt =
                tx.prepare("INSERT INTO entry_overrides (slug, capabilities) VALUES (?1, ?2)")?;
            for o in overrides {
                stmt.execute((o.slug.as_str(), o.capabilities_json.as_str()))?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Load the cached operator bindings (row-id sorted for determinism).
    /// Orphan alias rows (their slug was pruned) are dropped here so callers
    /// never see bindings without a cached entry.
    pub fn load_bindings(&self) -> DbResult<(Vec<AliasRow>, Vec<EntryOverride>)> {
        let live: std::collections::HashSet<String> = {
            let mut stmt = self.conn.prepare("SELECT slug FROM entries")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.collect::<Result<_, _>>()?
        };
        let mut aliases: Vec<AliasRow> = {
            let mut stmt = self
                .conn
                .prepare("SELECT alias, slug FROM aliases ORDER BY rowid")?;
            let rows = stmt.query_map([], |r| {
                Ok(AliasRow {
                    alias: r.get(0)?,
                    slug: r.get(1)?,
                })
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        aliases.retain(|row| live.contains(&row.slug));
        let mut overrides: Vec<EntryOverride> = {
            let mut stmt = self
                .conn
                .prepare("SELECT slug, capabilities FROM entry_overrides ORDER BY rowid")?;
            let rows = stmt.query_map([], |r| {
                Ok(EntryOverride {
                    slug: r.get(0)?,
                    capabilities_json: r.get(1)?,
                })
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        overrides.retain(|o| live.contains(&o.slug));
        Ok((aliases, overrides))
    }

    /// Remove rows whose file no longer exists on disk; returns the number
    /// pruned.
    pub fn prune_missing_roots(&self) -> DbResult<usize> {
        let paths: Vec<(String, String)> = {
            let mut stmt = self.conn.prepare("SELECT slug, path FROM entries")?;
            let rows =
                stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        let mut pruned = 0;
        for (slug, path) in paths {
            if !Path::new(&path).exists() {
                self.conn
                    .execute("DELETE FROM entries WHERE slug = ?1", [&slug])?;
                pruned += 1;
            }
        }
        Ok(pruned)
    }

    /// Row count (tests and diagnostics).
    #[must_use]
    pub fn count(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM entries", [], |r| r.get::<_, i64>(0))
            .map_or(0, |n| usize::try_from(n).unwrap_or(0))
    }
}

/// Canonical string for a [`crate::ModelKind`] (mirrors its `Serialize` impl).
fn kind_str(kind: crate::ModelKind) -> &'static str {
    match kind {
        crate::ModelKind::Llm => "llm",
        crate::ModelKind::Sd => "sd",
        crate::ModelKind::ControlNet => "controlnet",
        crate::ModelKind::Lora => "lora",
        crate::ModelKind::Embedding => "embedding",
        crate::ModelKind::Vae => "vae",
        crate::ModelKind::Stt => "stt",
        crate::ModelKind::Tts => "tts",
        crate::ModelKind::Ocr => "ocr",
    }
}

/// Inverse of [`kind_str`]; unknown strings degrade to [`crate::ModelKind::Sd`]
/// (the cache is advisory and the next scan corrects it).
fn parse_kind(s: &str) -> crate::ModelKind {
    match s {
        "llm" => crate::ModelKind::Llm,
        "controlnet" => crate::ModelKind::ControlNet,
        "lora" => crate::ModelKind::Lora,
        "embedding" => crate::ModelKind::Embedding,
        "vae" => crate::ModelKind::Vae,
        "stt" => crate::ModelKind::Stt,
        "tts" => crate::ModelKind::Tts,
        "ocr" => crate::ModelKind::Ocr,
        _ => crate::ModelKind::Sd,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AliasRow, Db, EntryOverride, SCHEMA_VERSION, db_path_from, env_db_path, set_db_override,
    };
    use crate::{CatalogEntry, ModelKind};
    use std::path::PathBuf;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("honkbox-db-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create tempdir");
        dir
    }

    fn temp_db(tag: &str) -> PathBuf {
        temp_dir(tag).join("registry.sqlite3")
    }

    fn entry(slug: &str, kind: ModelKind, path: &str, bytes: u64) -> CatalogEntry {
        CatalogEntry {
            slug: slug.into(),
            kind,
            category: "checkpoints".into(),
            path: path.into(),
            bytes,
            metadata: None,
        }
    }

    fn real_file(tag: &str, name: &str) -> String {
        let p = temp_dir(tag).join(name);
        std::fs::write(&p, b"x").expect("write");
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn roundtrip_upsert_load_prune() {
        let tag = "roundtrip";
        let db = Db::open(&temp_db(tag)).expect("open");
        assert_eq!(db.count(), 0);
        let live = real_file(tag, "a.safetensors");
        let gone = format!("{live}.missing");
        let entries = vec![
            entry("comfy:a.safetensors", ModelKind::Sd, &live, 111),
            entry("comfy:b.safetensors", ModelKind::Lora, &gone, 222),
            entry("llm:sub/c.gguf", ModelKind::Llm, "/nowhere/c.gguf", 333),
        ];
        db.upsert_entries(&entries, 1_700_000).expect("upsert");
        assert_eq!(db.count(), 3);
        // Upsert is keyed by slug: re-inserting updates in place.
        db.upsert_entries(
            &[entry("comfy:a.safetensors", ModelKind::Sd, &live, 999)],
            1_700_001,
        )
        .expect("re-upsert");
        assert_eq!(db.count(), 3);
        let loaded = db.load_entries().expect("load");
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].slug, "comfy:a.safetensors"); // slug-sorted
        assert_eq!(loaded[0].bytes, 999);
        assert_eq!(loaded[0].kind, ModelKind::Sd);
        assert!(loaded.iter().all(|e| e.metadata.is_none()));
        let pruned = db.prune_missing_roots().expect("prune");
        assert_eq!(pruned, 2); // nonexistent paths only
        std::fs::remove_file(&live).expect("rm");
        assert_eq!(db.prune_missing_roots().expect("prune2"), 1);
        assert_eq!(db.count(), 0);
    }

    #[test]
    fn schema_version_from_empty() {
        let path = temp_db("fromempty");
        let db = Db::open(&path).expect("open");
        let v: String = db
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .expect("version row");
        assert_eq!(v, SCHEMA_VERSION.to_string());
        // Older schema is dropped and recreated at the current version.
        db.conn
            .execute_batch(
                "DROP TABLE entries; INSERT OR REPLACE INTO meta VALUES ('schema_version', '0');",
            )
            .expect("downgrade");
        drop(db);
        let db = Db::open(&path).expect("reopen");
        assert_eq!(db.count(), 0);
        db.upsert_entries(&[entry("llm:x.gguf", ModelKind::Llm, "/x", 1)], 5)
            .expect("upsert");
        assert_eq!(db.count(), 1);
    }

    #[test]
    fn disabled_mode_is_no_op() {
        use std::ffi::OsString;
        assert_eq!(db_path_from(Some("off".into()), None), None);
        assert_eq!(db_path_from(Some(OsString::new()), None), None);
        assert_eq!(
            db_path_from(
                Some("/tmp/honkbox-db-test-envpath.sqlite3".into()),
                Some(PathBuf::from("/default/ignored"))
            ),
            Some(PathBuf::from("/tmp/honkbox-db-test-envpath.sqlite3"))
        );
        // Unset env: the injected default wins when absolute; a relative or
        // missing default disables rather than landing in an arbitrary cwd.
        assert_eq!(
            db_path_from(None, Some(PathBuf::from("/home/u/registry.sqlite3"))),
            Some(PathBuf::from("/home/u/registry.sqlite3"))
        );
        assert_eq!(db_path_from(None, None), None);
        assert_eq!(
            db_path_from(None, Some(PathBuf::from("rel/db.sqlite3"))),
            None
        );
        // Override wins over env/default; `set_db_override(None)` disables.
        set_db_override(Some(PathBuf::from("/tmp/honkbox-db-override.sqlite3")));
        assert_eq!(
            env_db_path(Some(PathBuf::from("/default/registry.sqlite3"))),
            Some(PathBuf::from("/tmp/honkbox-db-override.sqlite3"))
        );
        set_db_override(None);
        assert_eq!(env_db_path(None), None);
        set_db_override(None::<PathBuf>); // restore env-driven resolution
    }

    #[test]
    fn bindings_roundtrip_and_orphan_prune() {
        let tag = "bindings";
        let db = Db::open(&temp_db(tag)).expect("open");
        let live = real_file(tag, "m.gguf");
        db.upsert_entries(&[entry("llm:m.gguf", ModelKind::Llm, &live, 10)], 1)
            .expect("upsert");
        db.sync_bindings(
            &[AliasRow {
                alias: "qwen3-embed".into(),
                slug: "llm:m.gguf".into(),
            }],
            &[EntryOverride {
                slug: "llm:m.gguf".into(),
                capabilities_json: r#"{"chat":false,"embeddings":true}"#.into(),
            }],
        )
        .expect("sync");
        let (aliases, overrides) = db.load_bindings().expect("load");
        assert_eq!(
            aliases,
            vec![AliasRow {
                alias: "qwen3-embed".into(),
                slug: "llm:m.gguf".into(),
            }]
        );
        assert_eq!(overrides.len(), 1);
        assert_eq!(overrides[0].slug, "llm:m.gguf");
        assert_eq!(
            overrides[0].capabilities_json,
            r#"{"chat":false,"embeddings":true}"#
        );

        // Replace-all: an emptied config clears the tables.
        db.sync_bindings(&[], &[]).expect("clear");
        let (aliases, overrides) = db.load_bindings().expect("load empty");
        assert!(aliases.is_empty() && overrides.is_empty());

        // Bindings whose slug has no cached entry are not served.
        db.sync_bindings(
            &[AliasRow {
                alias: "orphan".into(),
                slug: "llm:gone.gguf".into(),
            }],
            &[EntryOverride {
                slug: "llm:gone.gguf".into(),
                capabilities_json: "{}".into(),
            }],
        )
        .expect("sync orphan");
        let (aliases, overrides) = db.load_bindings().expect("load orphans");
        assert!(aliases.is_empty() && overrides.is_empty());
    }

    #[test]
    fn corrupted_db_file_is_recreated() {
        let path = temp_db("corrupt");
        std::fs::write(&path, b"definitely not a sqlite db").expect("corrupt");
        let db = Db::open(&path).expect("recreate");
        assert_eq!(db.count(), 0);
        db.upsert_entries(&[entry("llm:y.gguf", ModelKind::Llm, "/y", 2)], 7)
            .expect("upsert");
        assert_eq!(db.count(), 1);
    }
}
