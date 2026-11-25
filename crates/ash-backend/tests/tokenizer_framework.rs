//! Deterministic synthetic tests for the shared Ash tokenizer interface.
//!
//! No assets are bundled or downloaded: vocabularies and metadata are built
//! in memory by every test.

use honkbox_ash_backend::tokenizer::{
    ChatMessage, ChatRole, ClipTokenizer, PreTokenizer, SpecialTokenMeta, Tokenizer,
    TokenizerError, TokenizerMetadata, UnigramAsset, UnigramToken,
};
use std::collections::BTreeMap;

fn meta(name: &str, id: u32) -> SpecialTokenMeta {
    SpecialTokenMeta {
        name: name.to_owned(),
        id,
    }
}

fn token(token: &str, score: f64) -> UnigramToken {
    UnigramToken {
        token: token.to_owned(),
        score,
    }
}

fn clip_tokenizer() -> Tokenizer {
    let vocab = serde_json::json!({
        "<|startoftext|>": 49406u32,
        "<|endoftext|>": 49407u32,
        "h": 1, "e": 2, "l": 3, "o</w>": 4,
        "he": 5, "hel": 6, "hell": 7, "hello</w>": 8,
        "w": 9, "o": 10, "wo": 11, "wor": 12, "worl": 13,
        "d</w>": 14, "world</w>": 15,
        "i": 16, "t</w>": 17, "it</w>": 18,
        "'": 19, "s</w>": 20, "'s</w>": 21,
        "i</w>": 22
    })
    .to_string();
    let merges = concat!(
        "#version: 0.2\n",
        "h e\nhe l\nhel l\nhell o</w>\n",
        "w o\nwo r\nwor l\nworl d</w>\n",
        "i t</w>\n' s</w>\n"
    );
    let clip = ClipTokenizer::from_str(&vocab, merges).expect("clip assets");
    Tokenizer::clip(clip).expect("clip profile")
}

fn byte_vocab() -> String {
    let entries: [(&str, u32); 22] = [
        ("<bos>", 1),
        ("<eos>", 2),
        ("<user_start>", 3),
        ("<user_end>", 4),
        ("<assistant_start>", 5),
        ("<assistant_end>", 6),
        ("<system_start>", 7),
        ("<system_end>", 8),
        ("h", 10),
        ("e", 11),
        ("x", 12),
        ("i", 13),
        ("Ġ", 18),
        ("s", 19),
        ("y", 20),
        ("t", 21),
        ("m", 22),
        ("H", 23),
        ("E", 24),
        ("X", 25),
        ("b", 26),
        ("o", 27),
    ];
    let mut vocab = BTreeMap::new();
    for (text, id) in entries {
        vocab.insert(text.to_owned(), id);
    }
    vocab.insert("<".to_owned(), 14);
    vocab.insert(">".to_owned(), 17);
    vocab.insert("ÿ".to_owned(), 99);
    vocab.insert("he".to_owned(), 30);
    serde_json::to_string(&vocab).expect("vocab json")
}

fn byte_metadata() -> TokenizerMetadata {
    TokenizerMetadata {
        specials: vec![
            meta("bos", 1),
            meta("eos", 2),
            meta("user_start", 3),
            meta("user_end", 4),
            meta("assistant_start", 5),
            meta("assistant_end", 6),
            meta("system_start", 7),
            meta("system_end", 8),
        ],
        ..TokenizerMetadata::default()
    }
}

fn byte_tokenizer() -> Tokenizer {
    let vocab = byte_vocab();
    Tokenizer::lfm2(&vocab, "h e\nhe x\n", byte_metadata()).expect("lfm2 profile")
}

fn unigram_asset() -> UnigramAsset {
    UnigramAsset {
        tokens: vec![
            token("<bos>", 0.0),
            token("<eos>", 0.0),
            token("<unk>", 0.0),
            token("<system_start>", 0.0),
            token("<system_end>", 0.0),
            token("<user_start>", 0.0),
            token("<user_end>", 0.0),
            token("<assistant_start>", 0.0),
            token("<assistant_end>", 0.0),
            token("a", -1.0),
            token("b", -1.0),
            token("ab", -1.5),
            token("hi", -0.5),
            token("!", -1.0),
        ],
    }
}

fn unigram_metadata(with_unk: bool) -> TokenizerMetadata {
    let mut specials = vec![
        meta("bos", 0),
        meta("eos", 1),
        meta("system_start", 3),
        meta("system_end", 4),
        meta("user_start", 5),
        meta("user_end", 6),
        meta("assistant_start", 7),
        meta("assistant_end", 8),
    ];
    if with_unk {
        specials.push(meta("unk", 2));
    }
    TokenizerMetadata {
        specials,
        ..TokenizerMetadata::default()
    }
}

fn user_message(text: &str) -> [ChatMessage; 1] {
    [ChatMessage {
        role: ChatRole::User,
        content: text.to_owned(),
    }]
}

#[test]
fn clip_exact_ids_and_round_trip_unchanged() {
    let tokenizer = clip_tokenizer();
    let hello = tokenizer.encode("hello world", 16).expect("encode");
    assert_eq!(hello.ids, vec![49406, 8, 15, 49407]);
    assert_eq!(hello.spans.len(), hello.ids.len());
    assert_eq!(
        tokenizer.decode(&hello.ids[1..3]).expect("decode"),
        "helloworld"
    );
    assert_eq!(
        tokenizer.encode("it's", 16).expect("encode").ids,
        vec![49406, 18, 21, 49407]
    );
    assert_eq!(
        tokenizer.encode("unknown@@", 16),
        Err(TokenizerError::UnknownToken("u".to_owned()))
    );
}

#[test]
fn byte_bpe_falls_back_per_byte_without_lowercasing() {
    let tokenizer = byte_tokenizer();
    let encoded = tokenizer.encode("hex", 16).expect("fallback");
    assert_eq!(encoded.ids, vec![1, 10, 11, 12]);
    assert_eq!(encoded.spans, vec![0..0, 0..1, 1..2, 2..3]);
    assert_eq!(tokenizer.decode(&encoded.ids[1..]).expect("decode"), "hex");
    assert_eq!(
        tokenizer.encode("HEX", 16).expect("upper").ids,
        vec![1, 23, 24, 25]
    );

    let special_looking = tokenizer.encode("<bos>", 16).expect("raw text");
    assert_eq!(special_looking.ids, vec![1, 14, 26, 27, 19, 17]);
    assert!(!special_looking.ids[1..].contains(&1));

    let spaced = tokenizer.encode(" h i ", 16).expect("spaces kept");
    assert_eq!(spaced.ids, vec![1, 18, 10, 18, 13, 18]);
    assert_eq!(tokenizer.decode(&spaced.ids[1..]).expect("decode"), " h i ");
}

#[test]
fn unigram_best_path_and_deterministic_tie() {
    let tokenizer = Tokenizer::gemma(unigram_asset(), unigram_metadata(true)).expect("gemma");
    assert_eq!(tokenizer.encode("ab", 16).expect("best").ids, vec![0, 11]);
    assert_eq!(tokenizer.decode(&[11]).expect("decode"), "ab");

    let tie = UnigramAsset {
        tokens: vec![
            token("<bos>", 0.0),
            token("ab", -2.0),
            token("a", -1.0),
            token("b", -1.0),
        ],
    };
    let metadata = TokenizerMetadata {
        specials: vec![meta("bos", 0)],
        ..TokenizerMetadata::default()
    };
    let tie = Tokenizer::gemma(tie, metadata).expect("tie asset");
    assert_eq!(tie.encode("ab", 16).expect("tie").ids, vec![0, 1]);
}

#[test]
fn unigram_unknown_policy_and_malformed_assets() {
    let tokenizer = Tokenizer::qwen35(unigram_asset(), unigram_metadata(true)).expect("qwen35");
    assert_eq!(tokenizer.encode("©", 16).expect("unk").ids, vec![2]);

    let strict = Tokenizer::gemma(unigram_asset(), unigram_metadata(false)).expect("strict");
    assert_eq!(
        strict.encode("©", 16),
        Err(TokenizerError::UnknownToken("©".to_owned()))
    );

    let metadata = unigram_metadata(true);
    assert!(matches!(
        Tokenizer::gemma(UnigramAsset { tokens: vec![] }, metadata.clone()),
        Err(TokenizerError::MalformedAsset(_))
    ));
    assert!(matches!(
        Tokenizer::gemma(
            UnigramAsset {
                tokens: vec![token("a", f64::NAN)]
            },
            metadata.clone()
        ),
        Err(TokenizerError::MalformedAsset(_))
    ));
    assert!(matches!(
        Tokenizer::gemma(
            UnigramAsset {
                tokens: vec![token("a", -1.0), token("a", -2.0)]
            },
            metadata
        ),
        Err(TokenizerError::DuplicateField(_))
    ));
}

#[test]
fn declarative_chat_templates_are_profile_specific() {
    let messages = user_message("hi");
    let lfm = byte_tokenizer();
    assert_eq!(
        lfm.apply_chat_template(&messages, 16).expect("lfm2").ids,
        vec![1, 3, 10, 13, 4, 2]
    );

    let gemma = Tokenizer::gemma(unigram_asset(), unigram_metadata(true)).expect("gemma");
    assert_eq!(
        gemma.apply_chat_template(&messages, 16).expect("gemma").ids,
        vec![0, 5, 12, 6, 1]
    );

    let qwen = Tokenizer::qwen35(unigram_asset(), unigram_metadata(true)).expect("qwen35");
    assert_eq!(
        qwen.apply_chat_template(&messages, 16).expect("qwen35").ids,
        vec![5, 12, 6, 1]
    );

    assert_eq!(
        clip_tokenizer().apply_chat_template(&messages, 16),
        Err(TokenizerError::UnsupportedTemplate("clip".to_owned()))
    );
}

#[test]
fn caller_limit_is_exact_and_never_truncates() {
    let tokenizer = clip_tokenizer();
    assert_eq!(
        tokenizer.encode("hi", 3),
        Err(TokenizerError::TooLong {
            limit: 3,
            actual: 4
        })
    );
    assert_eq!(tokenizer.encode("hi", 4).expect("boundary").ids.len(), 4);

    let lfm = byte_tokenizer();
    assert_eq!(
        lfm.apply_chat_template(&user_message("hi"), 5),
        Err(TokenizerError::TooLong {
            limit: 5,
            actual: 6
        })
    );
}

#[test]
fn decode_rejects_unknown_ids_and_invalid_utf8() {
    let tokenizer = byte_tokenizer();
    assert_eq!(
        tokenizer.decode(&[9999]),
        Err(TokenizerError::UnknownId(9999))
    );
    assert!(matches!(
        tokenizer.decode(&[99]),
        Err(TokenizerError::InvalidUtf8(_))
    ));
}

#[test]
fn metadata_fields_are_named_and_validated() {
    let mut metadata = byte_metadata();
    metadata.specials.push(meta("bos", 2));
    assert_eq!(
        Tokenizer::lfm2(&byte_vocab(), "h e\n", metadata).err(),
        Some(TokenizerError::DuplicateField("bos".to_owned()))
    );

    let mut metadata = byte_metadata();
    metadata.specials.push(meta("nope", 40));
    assert_eq!(
        Tokenizer::lfm2(&byte_vocab(), "h e\n", metadata).err(),
        Some(TokenizerError::UnsupportedField("nope".to_owned()))
    );

    let mut metadata = byte_metadata();
    metadata.specials.push(meta("unk", 4242));
    assert_eq!(
        Tokenizer::lfm2(&byte_vocab(), "h e\n", metadata).err(),
        Some(TokenizerError::MismatchedField("unk".to_owned()))
    );

    let mut metadata = byte_metadata();
    metadata.tools = true;
    assert_eq!(
        Tokenizer::lfm2(&byte_vocab(), "h e\n", metadata).err(),
        Some(TokenizerError::UnsupportedFeature("tools".to_owned()))
    );

    let mut metadata = byte_metadata();
    metadata.media = true;
    assert_eq!(
        Tokenizer::lfm2(&byte_vocab(), "h e\n", metadata).err(),
        Some(TokenizerError::UnsupportedFeature("media".to_owned()))
    );

    let mut metadata = byte_metadata();
    metadata.chat_template = Some("{{ messages }}".to_owned());
    assert_eq!(
        Tokenizer::lfm2(&byte_vocab(), "h e\n", metadata).err(),
        Some(TokenizerError::UnsupportedFeature(
            "jinja chat_template".to_owned()
        ))
    );

    let mut metadata = byte_metadata();
    metadata.pre_tokenizer = Some(PreTokenizer::Unicode);
    assert_eq!(
        Tokenizer::lfm2(&byte_vocab(), "h e\n", metadata).err(),
        Some(TokenizerError::UnsupportedField("pre_tokenizer".to_owned()))
    );

    let mut metadata = byte_metadata();
    metadata.specials.retain(|item| item.name != "bos");
    assert_eq!(
        Tokenizer::lfm2(&byte_vocab(), "h e\n", metadata).err(),
        Some(TokenizerError::MissingField("bos".to_owned()))
    );
}
