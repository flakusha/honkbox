//! Prompt tokenizer assets and CLIP sequence padding for the direct path.
//!
//! ponytail: assets stay caller-supplied (directory with `vocab.json` and
//! `merges.txt`); bundling a 50k-token vocab would add license surface and
//! bytes this engine does not own. Add a packaged asset when distribution
//! requirements change.

use crate::tokenizer::{ClipTokenizer, Tokenizer, TokenizerError};
use std::env;
use std::path::{Path, PathBuf};

/// Environment variable naming the directory with CLIP tokenizer assets.
pub const VOCAB_DIR_ENV: &str = "HONKBOX_ASH_CLIP_VOCAB_DIR";

/// Resolve tokenizer files from `dir`, then from the env var, then fail loudly.
pub fn tokenizer_from(dir: Option<&Path>) -> Result<Tokenizer, TokenizerError> {
    let root = match dir {
        Some(path) => PathBuf::from(path),
        None => env::var_os(VOCAB_DIR_ENV)
            .map(PathBuf::from)
            .ok_or_else(|| {
                TokenizerError::MissingAsset(format!(
                    "{VOCAB_DIR_ENV} is unset; pass --clip-vocab-dir"
                ))
            })?,
    };
    let clip = ClipTokenizer::from_files(&root.join("vocab.json"), &root.join("merges.txt"))
        .map_err(|e| TokenizerError::MissingAsset(e.to_string()))?;
    Tokenizer::clip(clip)
}

/// Truncate to `tokens` and right-pad with EOS so CFG can compare sequences.
pub fn pad_ids(mut ids: Vec<u32>, tokens: usize, eos: u32) -> Vec<u32> {
    ids.truncate(tokens);
    ids.resize(tokens, eos);
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_and_truncates_eos() {
        assert_eq!(pad_ids(vec![1, 2, 3], 4, 9), vec![1, 2, 3, 9]);
        assert_eq!(pad_ids(vec![1, 2, 3], 2, 9), vec![1, 2]);
    }
}
