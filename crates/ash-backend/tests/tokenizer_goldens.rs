//! Opt-in golden-asset test for the shared Ash tokenizer interface.
//!
//! All assets are user-supplied and read-only. Nothing is downloaded, bundled
//! or generated. The manifest records a SHA-256 for itself and for every
//! referenced asset, plus an SPDX license per asset.
//!
//! Manifest shape:
//! ```json
//! {
//!   "manifest_sha256": "<sha256 of this file>",
//!   "assets": [{"path": "vocab.json", "sha256": "...", "license": "MIT"}],
//!   "cases": [{
//!     "profile": "clip|lfm2|gemma|qwen35",
//!     "vocab": "vocab.json", "merges": "merges.txt",
//!     "tokens": "unigram.json", "metadata": "metadata.json",
//!     "vectors": [{"text": "...", "ids": [1,2], "decoded": "..."}]
//!   }]
//! }
//! ```
//! `decoded` is optional; without it the vector's `text` is the expected
//! round trip. `limit` is optional and defaults to no caller limit.
#![cfg(feature = "ash-compute")]

use honkbox_ash_backend::tokenizer::{ClipTokenizer, Tokenizer, TokenizerMetadata, UnigramAsset};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Deserialize)]
struct AssetRecord {
    path: String,
    sha256: String,
    license: String,
}

#[derive(Debug, Deserialize)]
struct Vector {
    text: String,
    ids: Vec<u32>,
    decoded: Option<String>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Case {
    profile: String,
    vocab: Option<String>,
    merges: Option<String>,
    tokens: Option<String>,
    metadata: Option<String>,
    vectors: Vec<Vector>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    manifest_sha256: String,
    assets: Vec<AssetRecord>,
    cases: Vec<Case>,
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        panic!(
            "{name} must be set to run the tokenizer golden test; \
             supply a read-only asset directory, goldens manifest and manifest SHA-256"
        )
    })
}

fn digest(command: &str, args: &[&str], path: &Path) -> Option<String> {
    let output = Command::new(command).args(args).arg(path).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .map(|hex| hex.to_ascii_lowercase())
}

fn sha256(path: &Path) -> String {
    digest("sha256sum", &[], path)
        .or_else(|| digest("shasum", &["-a", "256"], path))
        .unwrap_or_else(|| panic!("need sha256sum or shasum to verify {}", path.display()))
}

fn read_asset(dir: &Path, name: &str) -> String {
    let path = dir.join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn metadata_for(dir: &Path, case: &Case) -> TokenizerMetadata {
    match &case.metadata {
        Some(name) => TokenizerMetadata::from_json(&read_asset(dir, name))
            .expect("golden metadata must be valid"),
        None => TokenizerMetadata::default(),
    }
}

fn tokenizer_for(dir: &Path, case: &Case) -> Tokenizer {
    let metadata = metadata_for(dir, case);
    match case.profile.as_str() {
        "clip" | "lfm2" => {
            let vocab = read_asset(dir, case.vocab.as_ref().expect("golden vocab asset"));
            let merges = read_asset(dir, case.merges.as_ref().expect("golden merges asset"));
            if case.profile == "clip" {
                let clip = ClipTokenizer::from_str(&vocab, &merges).expect("golden CLIP assets");
                Tokenizer::clip(clip)
            } else {
                Tokenizer::lfm2(&vocab, &merges, metadata)
            }
        }
        "gemma" | "qwen35" => {
            let tokens = UnigramAsset::from_json(&read_asset(
                dir,
                case.tokens.as_ref().expect("golden unigram asset"),
            ))
            .expect("golden unigram asset");
            if case.profile == "gemma" {
                Tokenizer::gemma(tokens, metadata)
            } else {
                Tokenizer::qwen35(tokens, metadata)
            }
        }
        other => panic!("golden manifest names unsupported profile: {other}"),
    }
    .expect("golden tokenizer must be valid")
}

#[test]
#[ignore = "requires HONKBOX_ASH_TOKENIZER_ASSET_DIR, _GOLDENS and _MANIFEST_SHA256"]
fn golden_tokenizer_ids_and_round_trips() {
    let dir = PathBuf::from(required_env("HONKBOX_ASH_TOKENIZER_ASSET_DIR"));
    let goldens = PathBuf::from(required_env("HONKBOX_ASH_TOKENIZER_GOLDENS"));
    let expected = required_env("HONKBOX_ASH_TOKENIZER_MANIFEST_SHA256").to_ascii_lowercase();

    assert_eq!(
        sha256(&goldens),
        expected,
        "HONKBOX_ASH_TOKENIZER_MANIFEST_SHA256 does not match the goldens file"
    );
    let manifest_text = fs::read_to_string(&goldens)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", goldens.display()));
    let manifest: Manifest = serde_json::from_str(&manifest_text)
        .expect("goldens must be JSON with manifest_sha256, assets and cases");
    assert_eq!(
        manifest.manifest_sha256.to_ascii_lowercase(),
        expected,
        "manifest field disagrees with the recorded SHA-256"
    );

    let mut recorded = HashSet::new();
    for asset in &manifest.assets {
        let path = dir.join(&asset.path);
        assert_eq!(
            sha256(&path),
            asset.sha256.to_ascii_lowercase(),
            "SHA-256 mismatch for {}",
            asset.path
        );
        let license = asset.license.trim();
        assert!(
            !license.is_empty()
                && !matches!(
                    license.to_ascii_uppercase().as_str(),
                    "UNKNOWN" | "NOASSERTION" | "NONE"
                ),
            "missing license for {}",
            asset.path
        );
        recorded.insert(asset.path.as_str());
    }

    for case in &manifest.cases {
        for name in [&case.vocab, &case.merges, &case.tokens, &case.metadata]
            .into_iter()
            .flatten()
        {
            assert!(
                recorded.contains(name.as_str()),
                "golden case references unrecorded asset: {name}"
            );
        }
        let tokenizer = tokenizer_for(&dir, case);
        for vector in &case.vectors {
            let encoded = tokenizer
                .encode(&vector.text, vector.limit.unwrap_or(usize::MAX))
                .expect("golden text must encode");
            assert_eq!(encoded.ids, vector.ids, "golden IDs for {:?}", vector.text);
            assert_eq!(encoded.spans.len(), encoded.ids.len());
            let expected_text = vector.decoded.as_deref().unwrap_or(&vector.text);
            assert_eq!(
                tokenizer.decode(&vector.ids).expect("golden decode"),
                expected_text,
                "golden round trip for {:?}",
                vector.text
            );
        }
    }
}
