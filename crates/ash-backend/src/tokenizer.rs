// size-allow: 1020
//! Shared tokenizer interface over caller-supplied assets.
//!
//! One [`Tokenizer`] serves CLIP and LLM profiles through `encode`, `decode`
//! and `apply_chat_template`. Byte-BPE is shared by CLIP and `lfm2`; one
//! Unigram model is shared by `gemma` and `qwen35`. Metadata is caller-supplied
//! (typed or JSON); no family is inferred from a filename, no GGUF is parsed,
//! and no vocabulary, template or dependency is bundled.
//!
//! ponytail: deterministic built-in declarative templates, not a Jinja engine.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{self, ErrorKind};
use std::ops::Range;
use std::path::Path;

const END_OF_WORD: &str = "</w>";
const START_TOKEN: &str = "<|startoftext|>";
const END_TOKEN: &str = "<|endoftext|>";
const CLIP_BOS_ID: u32 = 49406;
const CLIP_EOS_ID: u32 = 49407;
const CONTRACTIONS: [&str; 6] = ["'s", "'t", "'re", "'ve", "'ll", "'d"];
const UNK_PENALTY: f64 = -1.0e9;

/// Named failures from asset parsing, encoding, decoding and templates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// Caller-supplied file could not be read.
    MissingAsset(String),
    /// Asset bytes, JSON or declared values are malformed.
    MalformedAsset(String),
    /// Two fields claim the same name or token.
    DuplicateField(String),
    /// Field name is not supported by this profile.
    UnsupportedField(String),
    /// Field value disagrees with the vocabulary or another field.
    MismatchedField(String),
    /// Required field was not supplied.
    MissingField(String),
    /// Text cannot be represented by this vocabulary.
    UnknownToken(String),
    /// ID lies outside the vocabulary.
    UnknownId(u32),
    /// Decoded bytes are not valid UTF-8; no replacement character is emitted.
    InvalidUtf8(Vec<u8>),
    /// Profile has no built-in declarative template.
    UnsupportedTemplate(String),
    /// Jinja, tools or media were requested by name.
    UnsupportedFeature(String),
    /// Encoded length exceeds the caller-supplied limit; never truncated.
    TooLong {
        /// Caller-supplied maximum token count.
        limit: usize,
        /// Exact token count produced before the limit check.
        actual: usize,
    },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingAsset(message) => write!(f, "tokenizer asset: {message}"),
            Self::MalformedAsset(message) => write!(f, "tokenizer malformed asset: {message}"),
            Self::DuplicateField(name) => write!(f, "tokenizer duplicate field: {name}"),
            Self::UnsupportedField(name) => write!(f, "tokenizer unsupported field: {name}"),
            Self::MismatchedField(name) => write!(f, "tokenizer mismatched field: {name}"),
            Self::MissingField(name) => write!(f, "tokenizer missing field: {name}"),
            Self::UnknownToken(token) => write!(f, "tokenizer unknown token: {token}"),
            Self::UnknownId(id) => write!(f, "tokenizer unknown id: {id}"),
            Self::InvalidUtf8(bytes) => write!(f, "tokenizer invalid UTF-8 bytes: {bytes:?}"),
            Self::UnsupportedTemplate(profile) => {
                write!(f, "tokenizer unsupported template: {profile}")
            }
            Self::UnsupportedFeature(feature) => {
                write!(f, "tokenizer unsupported feature: {feature}")
            }
            Self::TooLong { limit, actual } => {
                write!(f, "tokenizer too long: limit {limit}, actual {actual}")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

/// Token IDs plus byte spans into the text that produced them.
///
/// Control tokens carry a span over their declared token text. Byte-BPE spans
/// cover source bytes; Unigram spans cover the raw string.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Encoded {
    /// Token IDs in sequence order.
    pub ids: Vec<u32>,
    /// One span per ID, in the same order.
    pub spans: Vec<Range<usize>>,
}

/// Supported tokenizer profiles. Each has an explicit injection/pre-tokenizer policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerProfile {
    /// SD 1.x CLIP BPE: lowercase, whitespace cleanup, fixed BOS/EOS.
    Clip,
    /// LFM2 GPT-2 byte BPE: no lowercasing, byte fallback, BOS only.
    Lfm2,
    /// Gemma Unigram: raw text, BOS only, declared `<unk>`.
    Gemma,
    /// Qwen3.5 Unigram adapter: raw text, no injected controls.
    Qwen35,
}

impl TokenizerProfile {
    /// Stable name used in named errors.
    pub fn label(self) -> &'static str {
        match self {
            Self::Clip => "clip",
            Self::Lfm2 => "lfm2",
            Self::Gemma => "gemma",
            Self::Qwen35 => "qwen35",
        }
    }
}

/// Data-driven pre-tokenizer selected by validated caller metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreTokenizer {
    /// Lowercase + collapse whitespace + CLIP letter/contraction split.
    Clip,
    /// CLIP-style split without lowercase or whitespace cleanup.
    #[default]
    Lfm2,
    /// Raw string matched at UTF-8 character boundaries.
    Unicode,
}

/// Special token declaration resolved against the caller's vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SpecialTokenMeta {
    /// Declared name (`bos`, `eos`, `unk` or a chat role marker).
    pub name: String,
    /// ID in the caller-supplied vocabulary.
    pub id: u32,
}

/// Caller-supplied tokenizer metadata. All fields are optional and validated.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct TokenizerMetadata {
    /// Special tokens by name.
    #[serde(default)]
    pub specials: Vec<SpecialTokenMeta>,
    /// Pre-tokenizer override; must match the selected profile.
    #[serde(default)]
    pub pre_tokenizer: Option<PreTokenizer>,
    /// Caller-supplied template text. Only built-in templates are accepted.
    #[serde(default)]
    pub chat_template: Option<String>,
    /// Tool use; rejected by name.
    #[serde(default)]
    pub tools: bool,
    /// Media tokens; rejected by name.
    #[serde(default)]
    pub media: bool,
}

impl TokenizerMetadata {
    /// Parse caller-supplied JSON metadata.
    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        serde_json::from_str(json)
            .map_err(|e| TokenizerError::MalformedAsset(format!("metadata: {e}")))
    }
}

/// Chat role for [`Tokenizer::apply_chat_template`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    /// System instruction.
    System,
    /// User turn.
    User,
    /// Assistant turn.
    Assistant,
}

/// One chat message: plain text content only, no tools or media.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// Message role.
    pub role: ChatRole,
    /// Raw text content.
    pub content: String,
}

/// One token and its Unigram score, in caller-supplied asset order.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct UnigramToken {
    /// Token text (valid UTF-8).
    pub token: String,
    /// Log-score used by the best-path search.
    pub score: f64,
}

/// Caller-supplied Unigram asset. IDs are positional indices.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct UnigramAsset {
    /// Declared tokens and scores.
    pub tokens: Vec<UnigramToken>,
}

impl UnigramAsset {
    /// Parse caller-supplied JSON; validation happens in [`Tokenizer::gemma`]
    /// and [`Tokenizer::qwen35`].
    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        serde_json::from_str(json)
            .map_err(|e| TokenizerError::MalformedAsset(format!("unigram asset: {e}")))
    }
}

fn io_error(error: TokenizerError) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, error.to_string())
}

fn malformed(message: impl Into<String>) -> TokenizerError {
    TokenizerError::MalformedAsset(message.into())
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

/// Byte→unicode inverse, built once from the same table.
fn byte_alphabet() -> HashMap<char, u8> {
    (0u8..=u8::MAX)
        .map(|byte| (byte_to_unicode(byte), byte))
        .collect()
}

/// Shared byte-BPE core: parse, rank-merge, encode and decode.
#[derive(Debug, Clone)]
struct ByteBpe {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    reverse: HashMap<u32, String>,
    byte_tokens: HashMap<char, u32>,
    alphabet: HashMap<char, u8>,
    end_of_word: Option<&'static str>,
}

impl ByteBpe {
    fn from_str(
        vocab_json: &str,
        merges_text: &str,
        end_of_word: Option<&'static str>,
    ) -> Result<Self, TokenizerError> {
        let vocab: HashMap<String, u32> =
            serde_json::from_str(vocab_json).map_err(|e| malformed(format!("vocab: {e}")))?;
        if vocab.is_empty() {
            return Err(malformed("vocab is empty"));
        }
        let mut reverse = HashMap::with_capacity(vocab.len());
        for (token, id) in &vocab {
            if reverse.insert(*id, token.clone()).is_some() {
                return Err(TokenizerError::MismatchedField(format!(
                    "duplicate vocab id {id}"
                )));
            }
        }
        let mut ranks = HashMap::new();
        for (index, line) in merges_text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let left = parts.next().ok_or_else(|| malformed("empty merge line"))?;
            let right = parts
                .next()
                .ok_or_else(|| malformed("merge pair needs two tokens"))?;
            if parts.next().is_some() {
                return Err(malformed("merge pair has extra tokens"));
            }
            let rank = u32::try_from(index).map_err(|_| malformed("too many merge pairs"))?;
            ranks.insert((left.to_owned(), right.to_owned()), rank);
        }
        let alphabet = byte_alphabet();
        let byte_tokens = alphabet
            .keys()
            .filter_map(|character| {
                vocab
                    .get(&character.to_string())
                    .map(|id| (*character, *id))
            })
            .collect();
        Ok(Self {
            vocab,
            ranks,
            reverse,
            byte_tokens,
            alphabet,
            end_of_word,
        })
    }

    fn id(&self, token: &str) -> Result<u32, TokenizerError> {
        self.vocab
            .get(token)
            .copied()
            .ok_or_else(|| TokenizerError::UnknownToken(token.to_owned()))
    }

    fn contains_id(&self, id: u32) -> bool {
        self.reverse.contains_key(&id)
    }

    fn token_text(&self, id: u32) -> Option<String> {
        self.reverse.get(&id).cloned()
    }

    /// Split text like the CLIP regex: contractions first, then runs of
    /// letters vs runs of non-letter characters; whitespace runs are kept
    /// only for raw LLM profiles.
    fn pieces(text: &str, keep_whitespace: bool) -> Vec<String> {
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
                if keep_whitespace {
                    let end = rest
                        .char_indices()
                        .find(|(_, c)| !c.is_whitespace())
                        .map_or(rest.len(), |(index, _)| index);
                    out.push(rest[..end].to_owned());
                    rest = &rest[end..];
                } else {
                    rest = rest.trim_start();
                }
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

    /// Apply ranked BPE merges to one pre-split piece.
    /// Each token carries its source-byte length for lossless spans.
    fn bpe(&self, piece: &str) -> Vec<(String, usize)> {
        let mut word: Vec<String> = piece
            .as_bytes()
            .iter()
            .map(|byte| byte_to_unicode(*byte).to_string())
            .collect();
        let mut lengths = vec![1usize; word.len()];
        if word.is_empty() {
            return Vec::new();
        }
        if let Some(eow) = self.end_of_word {
            let last = word.len() - 1;
            word[last].push_str(eow);
        }

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
            let length = lengths[index] + lengths[index + 1];
            word.splice(index..index + 2, [merged]);
            lengths.splice(index..index + 2, [length]);
        }
        word.into_iter().zip(lengths).collect()
    }

    fn encode_pieces(
        &self,
        pieces: &[String],
        byte_fallback: bool,
        literal_specials: bool,
    ) -> Result<(Vec<u32>, Vec<Range<usize>>), TokenizerError> {
        let mut ids = Vec::new();
        let mut spans = Vec::new();
        let mut cursor = 0usize;
        for piece in pieces {
            if literal_specials && (piece == START_TOKEN || piece == END_TOKEN) {
                ids.push(self.id(piece)?);
                spans.push(cursor..cursor);
                continue;
            }
            for (token, length) in self.bpe(piece) {
                let start = cursor;
                cursor += length;
                match self.vocab.get(&token) {
                    Some(id) => {
                        ids.push(*id);
                        spans.push(start..cursor);
                    }
                    None if byte_fallback => {
                        let mut offset = start;
                        for character in token.chars() {
                            let id = *self
                                .byte_tokens
                                .get(&character)
                                .ok_or_else(|| TokenizerError::UnknownToken(token.clone()))?;
                            ids.push(id);
                            spans.push(offset..offset + 1);
                            offset += 1;
                        }
                    }
                    None => return Err(TokenizerError::UnknownToken(token)),
                }
            }
        }
        Ok((ids, spans))
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for id in ids {
            let token = self.reverse.get(id).ok_or(TokenizerError::UnknownId(*id))?;
            let token = self
                .end_of_word
                .and_then(|eow| token.strip_suffix(eow))
                .unwrap_or(token);
            for character in token.chars() {
                let byte = *self
                    .alphabet
                    .get(&character)
                    .ok_or_else(|| malformed(format!("token outside byte alphabet: {token}")))?;
                bytes.push(byte);
            }
        }
        String::from_utf8(bytes).map_err(|e| TokenizerError::InvalidUtf8(e.into_bytes()))
    }
}

/// CLIP BPE asset holder. Its merge algorithm lives in the shared [`ByteBpe`] core.
#[derive(Debug, Clone)]
pub struct ClipTokenizer {
    core: ByteBpe,
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
        ByteBpe::from_str(vocab_json, merges_text, Some(END_OF_WORD))
            .map(|core| Self { core })
            .map_err(io_error)
    }

    fn core(self) -> ByteBpe {
        self.core
    }
}

/// Shared Unigram best-path encoder and strict detokenizer.
#[derive(Debug, Clone)]
struct Unigram {
    tokens: Vec<String>,
    scores: Vec<f64>,
}

impl Unigram {
    fn from_asset(asset: UnigramAsset) -> Result<Self, TokenizerError> {
        if asset.tokens.is_empty() {
            return Err(malformed("unigram tokens are empty"));
        }
        let mut seen = HashSet::with_capacity(asset.tokens.len());
        let mut tokens = Vec::with_capacity(asset.tokens.len());
        let mut scores = Vec::with_capacity(asset.tokens.len());
        for entry in asset.tokens {
            if entry.token.is_empty() {
                return Err(malformed("unigram token is empty"));
            }
            if !entry.score.is_finite() {
                return Err(malformed(format!(
                    "unigram score not finite for '{}'",
                    entry.token
                )));
            }
            if !seen.insert(entry.token.clone()) {
                return Err(TokenizerError::DuplicateField(entry.token));
            }
            tokens.push(entry.token);
            scores.push(entry.score);
        }
        Ok(Self { tokens, scores })
    }

    fn contains_id(&self, id: u32) -> bool {
        usize::try_from(id).is_ok_and(|index| index < self.tokens.len())
    }

    fn token_text(&self, id: u32) -> Option<String> {
        usize::try_from(id)
            .ok()
            .and_then(|index| self.tokens.get(index))
            .cloned()
    }

    fn encode_pieces(
        &self,
        pieces: &[String],
        unk: Option<u32>,
    ) -> Result<(Vec<u32>, Vec<Range<usize>>), TokenizerError> {
        let mut ids = Vec::new();
        let mut spans = Vec::new();
        let mut offset = 0usize;
        for piece in pieces {
            let boundaries: Vec<usize> = piece
                .char_indices()
                .map(|(index, _)| index)
                .chain(std::iter::once(piece.len()))
                .collect();
            let mut best: Vec<Option<(f64, usize, usize)>> = vec![None; boundaries.len()];
            best[0] = Some((0.0, 0, 0));
            let unk_index = usize::MAX;
            for start in 0..boundaries.len() - 1 {
                let Some((score, _, _)) = best[start] else {
                    continue;
                };
                let remaining = &piece[boundaries[start]..];
                for (token_index, token) in self.tokens.iter().enumerate() {
                    if !remaining.starts_with(token) {
                        continue;
                    }
                    let end_byte = boundaries[start] + token.len();
                    let Some(end) = boundaries.iter().position(|b| *b == end_byte) else {
                        continue;
                    };
                    let candidate = score + self.scores[token_index];
                    if best[end].is_none_or(|(current, _, _)| candidate > current) {
                        best[end] = Some((candidate, start, token_index));
                    }
                }
                if unk.is_some() && best[start + 1].is_none() {
                    best[start + 1] = Some((score + UNK_PENALTY, start, unk_index));
                }
            }
            let mut end = best.len() - 1;
            if best[end].is_none() {
                return Err(TokenizerError::UnknownToken(piece.clone()));
            }
            let mut segments = Vec::new();
            while end > 0 {
                let (_, start, token_index) = best[end].expect("best path exists");
                segments.push((start, end, token_index));
                end = start;
            }
            segments.reverse();
            for (start, end, token_index) in segments {
                let id = if token_index == unk_index {
                    unk.ok_or_else(|| TokenizerError::UnknownToken(piece.clone()))?
                } else {
                    u32::try_from(token_index)
                        .map_err(|_| malformed("unigram has too many tokens"))?
                };
                ids.push(id);
                spans.push(offset + boundaries[start]..offset + boundaries[end]);
            }
            offset += piece.len();
        }
        Ok((ids, spans))
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut out = String::new();
        for id in ids {
            let index = usize::try_from(*id).map_err(|_| TokenizerError::UnknownId(*id))?;
            let token = self
                .tokens
                .get(index)
                .ok_or(TokenizerError::UnknownId(*id))?;
            out.push_str(token);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
enum Model {
    ByteBpe(ByteBpe),
    Unigram(Unigram),
}

impl Model {
    fn contains_id(&self, id: u32) -> bool {
        match self {
            Self::ByteBpe(model) => model.contains_id(id),
            Self::Unigram(model) => model.contains_id(id),
        }
    }

    fn token_text(&self, id: u32) -> Option<String> {
        match self {
            Self::ByteBpe(model) => model.token_text(id),
            Self::Unigram(model) => model.token_text(id),
        }
    }
}

fn default_pre_tokenizer(profile: TokenizerProfile) -> PreTokenizer {
    match profile {
        TokenizerProfile::Clip => PreTokenizer::Clip,
        TokenizerProfile::Lfm2 => PreTokenizer::Lfm2,
        TokenizerProfile::Gemma | TokenizerProfile::Qwen35 => PreTokenizer::Unicode,
    }
}

fn injection(profile: TokenizerProfile) -> (Option<&'static str>, Option<&'static str>) {
    match profile {
        TokenizerProfile::Clip => (Some("bos"), Some("eos")),
        TokenizerProfile::Lfm2 | TokenizerProfile::Gemma => (Some("bos"), None),
        TokenizerProfile::Qwen35 => (None, None),
    }
}

fn is_special_name(name: &str) -> bool {
    matches!(
        name,
        "bos"
            | "eos"
            | "unk"
            | "system_start"
            | "system_end"
            | "user_start"
            | "user_end"
            | "assistant_start"
            | "assistant_end"
    )
}

/// Declarative built-in chat template: control names around raw text parts.
struct ChatSpec {
    prefix: &'static [&'static str],
    open: [&'static str; 3],
    close: [&'static str; 3],
    suffix: &'static [&'static str],
}

impl ChatSpec {
    fn for_profile(profile: TokenizerProfile) -> Option<Self> {
        let open = ["system_start", "user_start", "assistant_start"];
        let close = ["system_end", "user_end", "assistant_end"];
        match profile {
            TokenizerProfile::Lfm2 | TokenizerProfile::Gemma => Some(Self {
                prefix: &["bos"],
                open,
                close,
                suffix: &["eos"],
            }),
            TokenizerProfile::Qwen35 => Some(Self {
                prefix: &[],
                open,
                close,
                suffix: &["eos"],
            }),
            TokenizerProfile::Clip => None,
        }
    }
}

fn role_index(role: ChatRole) -> usize {
    match role {
        ChatRole::System => 0,
        ChatRole::User => 1,
        ChatRole::Assistant => 2,
    }
}

enum Part<'a> {
    Control(&'static str),
    Text(&'a str),
}

/// One tokenizer interface for all profiles.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    profile: TokenizerProfile,
    model: Model,
    specials: HashMap<String, u32>,
    pre_tokenizer: PreTokenizer,
    literal_specials: bool,
}

impl Tokenizer {
    /// Build the CLIP profile from caller-supplied assets.
    pub fn clip(clip: ClipTokenizer) -> Result<Self, TokenizerError> {
        let metadata = TokenizerMetadata {
            specials: vec![
                SpecialTokenMeta {
                    name: "bos".to_owned(),
                    id: CLIP_BOS_ID,
                },
                SpecialTokenMeta {
                    name: "eos".to_owned(),
                    id: CLIP_EOS_ID,
                },
            ],
            pre_tokenizer: Some(PreTokenizer::Clip),
            ..TokenizerMetadata::default()
        };
        Self::build(
            TokenizerProfile::Clip,
            Model::ByteBpe(clip.core()),
            metadata,
        )
    }

    /// Build the `lfm2` byte-BPE profile from caller-supplied assets/metadata.
    pub fn lfm2(
        vocab_json: &str,
        merges_text: &str,
        metadata: TokenizerMetadata,
    ) -> Result<Self, TokenizerError> {
        let core = ByteBpe::from_str(vocab_json, merges_text, None)?;
        Self::build(TokenizerProfile::Lfm2, Model::ByteBpe(core), metadata)
    }

    /// Build the `gemma` Unigram profile from caller-supplied assets/metadata.
    pub fn gemma(asset: UnigramAsset, metadata: TokenizerMetadata) -> Result<Self, TokenizerError> {
        let model = Unigram::from_asset(asset)?;
        Self::build(TokenizerProfile::Gemma, Model::Unigram(model), metadata)
    }

    /// Build the `qwen35` Unigram adapter from caller-supplied assets/metadata.
    pub fn qwen35(
        asset: UnigramAsset,
        metadata: TokenizerMetadata,
    ) -> Result<Self, TokenizerError> {
        let model = Unigram::from_asset(asset)?;
        Self::build(TokenizerProfile::Qwen35, Model::Unigram(model), metadata)
    }

    fn build(
        profile: TokenizerProfile,
        model: Model,
        metadata: TokenizerMetadata,
    ) -> Result<Self, TokenizerError> {
        if metadata.chat_template.is_some() {
            return Err(TokenizerError::UnsupportedFeature(
                "jinja chat_template".to_owned(),
            ));
        }
        if metadata.tools {
            return Err(TokenizerError::UnsupportedFeature("tools".to_owned()));
        }
        if metadata.media {
            return Err(TokenizerError::UnsupportedFeature("media".to_owned()));
        }
        let pre_tokenizer = metadata
            .pre_tokenizer
            .unwrap_or_else(|| default_pre_tokenizer(profile));
        if pre_tokenizer != default_pre_tokenizer(profile) {
            return Err(TokenizerError::UnsupportedField("pre_tokenizer".to_owned()));
        }
        let mut specials = HashMap::new();
        let mut ids_seen = HashMap::new();
        let mut texts_seen = HashMap::new();
        for meta in &metadata.specials {
            if !is_special_name(&meta.name) {
                return Err(TokenizerError::UnsupportedField(meta.name.clone()));
            }
            if specials.insert(meta.name.clone(), meta.id).is_some() {
                return Err(TokenizerError::DuplicateField(meta.name.clone()));
            }
            if let Some(previous) = ids_seen.insert(meta.id, meta.name.clone())
                && previous != meta.name
            {
                return Err(TokenizerError::MismatchedField(meta.name.clone()));
            }
            if !model.contains_id(meta.id) {
                return Err(TokenizerError::MismatchedField(meta.name.clone()));
            }
            let text = model
                .token_text(meta.id)
                .ok_or_else(|| TokenizerError::MismatchedField(meta.name.clone()))?;
            if let Some(previous) = texts_seen.insert(text, meta.name.clone())
                && previous != meta.name
            {
                return Err(TokenizerError::MismatchedField(meta.name.clone()));
            }
        }
        let (bos, eos) = injection(profile);
        for name in [bos, eos].into_iter().flatten() {
            if !specials.contains_key(name) {
                return Err(TokenizerError::MissingField(name.to_owned()));
            }
        }
        if let (Some(bos), Some(eos)) = (bos, eos)
            && specials[eos] != specials[bos] + 1
        {
            return Err(TokenizerError::MismatchedField("clip bos/eos".to_owned()));
        }
        Ok(Self {
            profile,
            model,
            specials,
            pre_tokenizer,
            literal_specials: profile == TokenizerProfile::Clip,
        })
    }

    /// Selected profile.
    pub fn profile(&self) -> TokenizerProfile {
        self.profile
    }

    /// Encode raw text; BOS/EOS are injected only under profile policy.
    /// Over-limit input returns [`TokenizerError::TooLong`], never truncation.
    pub fn encode(&self, text: &str, limit: usize) -> Result<Encoded, TokenizerError> {
        let pieces = self.split(text);
        let (mut ids, mut spans) = self.encode_raw(&pieces)?;
        let (bos, eos) = injection(self.profile);
        if let Some(name) = bos {
            let id = self.special(name)?;
            ids.insert(0, id);
            spans.insert(0, 0..0);
        }
        if let Some(name) = eos {
            let id = self.special(name)?;
            ids.push(id);
            spans.push(text.len()..text.len());
        }
        self.finish(ids, spans, limit)
    }

    /// Decode IDs strictly: out-of-range IDs and invalid UTF-8 are named failures.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        match &self.model {
            Model::ByteBpe(model) => model.decode(ids),
            Model::Unigram(model) => model.decode(ids),
        }
    }

    /// Render a built-in declarative chat template, then encode its parts.
    /// Raw text and generated control tokens take separate paths.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        limit: usize,
    ) -> Result<Encoded, TokenizerError> {
        let spec = ChatSpec::for_profile(self.profile)
            .ok_or_else(|| TokenizerError::UnsupportedTemplate(self.profile.label().to_owned()))?;
        let mut parts: Vec<Part<'_>> = Vec::new();
        parts.extend(spec.prefix.iter().copied().map(Part::Control));
        for message in messages {
            let index = role_index(message.role);
            parts.push(Part::Control(spec.open[index]));
            parts.push(Part::Text(&message.content));
            parts.push(Part::Control(spec.close[index]));
        }
        parts.extend(spec.suffix.iter().copied().map(Part::Control));

        let mut ids = Vec::new();
        let mut spans = Vec::new();
        let mut cursor = 0usize;
        for part in parts {
            match part {
                Part::Control(name) => {
                    let id = self.special(name)?;
                    let width = self
                        .model
                        .token_text(id)
                        .ok_or_else(|| TokenizerError::MismatchedField(name.to_owned()))?
                        .len();
                    ids.push(id);
                    spans.push(cursor..cursor + width);
                    cursor += width;
                }
                Part::Text(content) => {
                    let pieces = self.split(content);
                    let (part_ids, part_spans) = self.encode_raw(&pieces)?;
                    for (id, span) in part_ids.into_iter().zip(part_spans) {
                        ids.push(id);
                        spans.push(span.start + cursor..span.end + cursor);
                    }
                    cursor += content.len();
                }
            }
        }
        self.finish(ids, spans, limit)
    }

    fn split(&self, text: &str) -> Vec<String> {
        match self.pre_tokenizer {
            PreTokenizer::Clip => {
                let normalized = text
                    .to_lowercase()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                ByteBpe::pieces(&normalized, false)
            }
            PreTokenizer::Lfm2 => ByteBpe::pieces(text, true),
            PreTokenizer::Unicode => vec![text.to_owned()],
        }
    }

    fn encode_raw(
        &self,
        pieces: &[String],
    ) -> Result<(Vec<u32>, Vec<Range<usize>>), TokenizerError> {
        match &self.model {
            Model::ByteBpe(model) => model.encode_pieces(
                pieces,
                self.profile == TokenizerProfile::Lfm2,
                self.literal_specials,
            ),
            Model::Unigram(model) => model.encode_pieces(pieces, self.specials.get("unk").copied()),
        }
    }

    fn special(&self, name: &str) -> Result<u32, TokenizerError> {
        self.specials
            .get(name)
            .copied()
            .ok_or_else(|| TokenizerError::MissingField(name.to_owned()))
    }

    fn finish(
        &self,
        ids: Vec<u32>,
        spans: Vec<Range<usize>>,
        limit: usize,
    ) -> Result<Encoded, TokenizerError> {
        if ids.len() > limit {
            return Err(TokenizerError::TooLong {
                limit,
                actual: ids.len(),
            });
        }
        Ok(Encoded { ids, spans })
    }
}
