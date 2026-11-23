//! CLIP BPE tokenizer over caller-supplied `vocab.json` / `merges.txt`.
//!
//! This repo does not bundle vocab or merge files: upstream CLIP assets are
//! MIT-licensed but still need a REUSE entry if vendored. Callers point at
//! their own files, so licensing stays with the asset owner. The byte-to-
//! unicode map follows the published GPT-2/CLIP algorithm (MIT); no upstream
//! source is copied verbatim.
//!
//! ponytail: ASCII/Unicode-letter segmentation and lowercase + whitespace
//! cleanup, not ftfy/HTML unescaping or diacritic folding. Add those when a
//! prompt corpus shows the difference.

use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::path::Path;

const END_OF_WORD: &str = "</w>";
const START_TOKEN: &str = "<|startoftext|>";
const END_TOKEN: &str = "<|endoftext|>";
const CONTRACTIONS: [&str; 6] = ["'s", "'t", "'re", "'ve", "'ll", "'d"];

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

/// GPT-2/CLIP byte→unicode table, derived from the published algorithm.
fn byte_to_unicode(byte: u8) -> char {
    let visible = (b'!'..=b'~')
        .chain(b'\xa1'..=b'\xac')
        .chain(b'\xae'..=b'\xff')
        .any(|b| b == byte);
    if visible {
        char::from(byte)
    } else {
        let extra = (0u32..256)
            .filter(|b| {
                let b = u8::try_from(*b).expect("0..256");
                !(b'!'..=b'~').contains(&b)
                    && !(b'\xa1'..=b'\xac').contains(&b)
                    && !(b'\xae'..=b'\xff').contains(&b)
            })
            .position(|b| u8::try_from(b).expect("0..256") == byte)
            .expect("non-visible byte is in the table");
        let extra = u32::try_from(extra).expect("byte table index fits u32");
        char::from_u32(256 + extra).expect("byte token codepoint")
    }
}

/// Merges and token IDs from CLIP asset files.
pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
}

impl ClipTokenizer {
    /// Parse `vocab.json` and `merges.txt` from caller-supplied paths.
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> io::Result<Self> {
        let vocab_text = std::fs::read_to_string(vocab_path)?;
        let merges_text = std::fs::read_to_string(merges_path)?;
        Self::from_str(&vocab_text, &merges_text)
    }

    /// Parse asset contents directly (also used by tests).
    pub fn from_str(vocab_json: &str, merges_text: &str) -> io::Result<Self> {
        let vocab: HashMap<String, u32> =
            serde_json::from_str(vocab_json).map_err(|e| invalid(format!("clip vocab: {e}")))?;
        let mut ranks = HashMap::new();
        for (index, line) in merges_text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let left = parts
                .next()
                .ok_or_else(|| invalid("clip merges: empty line"))?;
            let right = parts
                .next()
                .ok_or_else(|| invalid("clip merges: pair needs two tokens"))?;
            if parts.next().is_some() {
                return Err(invalid("clip merges: pair has extra tokens"));
            }
            let rank = u32::try_from(index).map_err(|_| invalid("clip merges: too many pairs"))?;
            ranks.insert((left.to_owned(), right.to_owned()), rank);
        }
        Ok(Self { vocab, ranks })
    }

    /// Tokenize a prompt into CLIP token IDs, including BOS and EOS.
    pub fn encode(&self, prompt: &str) -> io::Result<Vec<u32>> {
        let mut ids = vec![self.id(START_TOKEN)?];
        let normalized = prompt
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        for piece in Self::pieces(&normalized) {
            if piece == START_TOKEN || piece == END_TOKEN {
                ids.push(self.id(&piece)?);
                continue;
            }
            for token in self.bpe(&piece) {
                ids.push(self.id(&token)?);
            }
        }
        ids.push(self.id(END_TOKEN)?);
        Ok(ids)
    }

    fn id(&self, token: &str) -> io::Result<u32> {
        self.vocab
            .get(token)
            .copied()
            .ok_or_else(|| invalid(format!("clip vocab: unknown token '{token}'")))
    }

    /// Split lowercase text like CLIP's regex: contractions first, then runs
    /// of letters vs runs of non-letter, non-space characters.
    fn pieces(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            if rest.starts_with(START_TOKEN) {
                out.push(START_TOKEN.to_owned());
                rest = &rest[START_TOKEN.len()..];
                continue;
            }
            if rest.starts_with(END_TOKEN) {
                out.push(END_TOKEN.to_owned());
                rest = &rest[END_TOKEN.len()..];
                continue;
            }
            if let Some(contraction) = CONTRACTIONS.iter().find(|c| rest.starts_with(**c)) {
                out.push((*contraction).to_owned());
                rest = &rest[contraction.len()..];
                continue;
            }
            let first = rest.chars().next().expect("non-empty rest");
            if first.is_whitespace() {
                rest = rest.trim_start();
                continue;
            }
            let is_letter = first.is_alphabetic();
            let end = rest
                .char_indices()
                .find(|(_, c)| c.is_whitespace() || c.is_alphabetic() != is_letter)
                .map_or(rest.len(), |(index, _)| index);
            out.push(rest[..end].to_owned());
            rest = &rest[end..];
        }
        out
    }

    /// Apply ranked BPE merges to one pre-split text piece.
    fn bpe(&self, piece: &str) -> Vec<String> {
        let mut word: Vec<String> = piece
            .as_bytes()
            .iter()
            .map(|byte| byte_to_unicode(*byte).to_string())
            .collect();
        if word.is_empty() {
            return word;
        }
        let last = word.len() - 1;
        word[last].push_str(END_OF_WORD);

        while word.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for index in 0..word.len() - 1 {
                if let Some(rank) = self
                    .ranks
                    .get(&(word[index].clone(), word[index + 1].clone()))
                    && best.is_none_or(|(current, _)| *rank < current)
                {
                    best = Some((*rank, index));
                }
            }
            let Some((_, index)) = best else { break };
            let merged = format!("{}{}", word[index], word[index + 1]);
            word.splice(index..index + 2, [merged]);
        }
        word
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> ClipTokenizer {
        let vocab = serde_json::json!({
            START_TOKEN: 49406u32,
            END_TOKEN: 49407u32,
            "h": 1, "e": 2, "l": 3, "o</w>": 4,
            "he": 5, "hel": 6, "hell": 7, "hello</w>": 8,
            "w": 9, "o": 10, "wo": 11, "wor": 12, "worl": 13,
            "d</w>": 14, "world</w>": 15,
            "i": 16, "t</w>": 17, "it</w>": 18,
            "'": 19, "s</w>": 20, "'s</w>": 21
        })
        .to_string();
        let merges = concat!(
            "#version: 0.2\n",
            "h e\nhe l\nhel l\nhell o</w>\n",
            "w o\nwo r\nwor l\nworl d</w>\n",
            "i t</w>\n' s</w>\n"
        );
        ClipTokenizer::from_str(&vocab, merges).expect("tokenizer")
    }

    #[test]
    fn merges_pairs_into_vocab_tokens() {
        let tokenizer = tokenizer();
        assert_eq!(
            tokenizer.encode("hello world").expect("encode"),
            vec![49406, 8, 15, 49407]
        );
    }

    #[test]
    fn splits_contractions_and_reports_unknown_tokens() {
        let tokenizer = tokenizer();
        assert_eq!(
            tokenizer.encode("it's").expect("encode"),
            vec![49406, 18, 21, 49407]
        );
        assert!(tokenizer.encode("unknown@@").is_err());
    }

    #[test]
    fn rejects_malformed_assets() {
        assert!(ClipTokenizer::from_str("[]", "").is_err());
        assert!(ClipTokenizer::from_str("{}", "only_one").is_err());
    }
}
