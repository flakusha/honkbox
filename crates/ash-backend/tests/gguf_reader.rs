//! Strict GGUF container-reader tests over tiny generated fixtures.
//!
//! Every test owns a unique temp directory under `TMPDIR` with PID, timestamp
//! and a local collision counter; RAII removes it on every exit path.
#![cfg(feature = "ash-compute")]
use honkbox_ash_backend::checkpoint::{GgufMap, GgufType, GgufValue, decode_gguf_tensor};
use std::fs;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let stem = format!("honkbox-ash-gguf-{label}-{}-{stamp}", std::process::id());
        for attempt in 0..64u32 {
            let dir = std::env::temp_dir().join(format!("{stem}-{attempt}"));
            if fs::create_dir(&dir).is_ok() {
                return Self(dir);
            }
        }
        panic!("no unique temp directory for {label}");
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, bytes).expect("fixture write");
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Default)]
struct Bytes(Vec<u8>);

impl Bytes {
    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn f64(&mut self, value: f64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn bytes(&mut self, value: &[u8]) {
        self.u64(value.len() as u64);
        self.raw(value);
    }
}

struct Kv {
    key: Vec<u8>,
    type_id: u32,
    value: Vec<u8>,
}

fn kv_raw(key: &[u8], type_id: u32, value: Vec<u8>) -> Kv {
    Kv {
        key: key.to_vec(),
        type_id,
        value,
    }
}

fn kv_scalar(key: &str, type_id: u32, value: Vec<u8>) -> Kv {
    kv_raw(key.as_bytes(), type_id, value)
}

fn kv_str(key: &str, value: &str) -> Kv {
    let mut bytes = Bytes::default();
    bytes.bytes(value.as_bytes());
    kv_raw(key.as_bytes(), 8, bytes.0)
}

fn kv_u8(key: &str, value: u8) -> Kv {
    kv_scalar(key, 0, vec![value])
}
fn kv_u16(key: &str, value: u16) -> Kv {
    let mut bytes = Bytes::default();
    bytes.0.extend_from_slice(&value.to_le_bytes());
    kv_scalar(key, 2, bytes.0)
}

fn kv_i16(key: &str, value: i16) -> Kv {
    let mut bytes = Bytes::default();
    bytes.0.extend_from_slice(&value.to_le_bytes());
    kv_scalar(key, 3, bytes.0)
}

fn kv_u32(key: &str, value: u32) -> Kv {
    let mut bytes = Bytes::default();
    bytes.u32(value);
    kv_scalar(key, 4, bytes.0)
}

fn kv_i32(key: &str, value: i32) -> Kv {
    let mut bytes = Bytes::default();
    bytes.0.extend_from_slice(&value.to_le_bytes());
    kv_scalar(key, 5, bytes.0)
}

fn kv_f32(key: &str, value: f32) -> Kv {
    let mut bytes = Bytes::default();
    bytes.f32(value);
    kv_scalar(key, 6, bytes.0)
}

fn kv_u64(key: &str, value: u64) -> Kv {
    let mut bytes = Bytes::default();
    bytes.u64(value);
    kv_scalar(key, 10, bytes.0)
}

fn kv_i64(key: &str, value: i64) -> Kv {
    let mut bytes = Bytes::default();
    bytes.i64(value);
    kv_scalar(key, 11, bytes.0)
}

fn kv_f64(key: &str, value: f64) -> Kv {
    let mut bytes = Bytes::default();
    bytes.f64(value);
    kv_scalar(key, 12, bytes.0)
}

fn kv_array(key: &str, elem_type: u32, count: u64, elements: Vec<u8>) -> Kv {
    let mut bytes = Bytes::default();
    bytes.u32(elem_type);
    bytes.u64(count);
    bytes.raw(&elements);
    kv_raw(key.as_bytes(), 9, bytes.0)
}

fn base_kvs(alignment: Option<u32>) -> Vec<Kv> {
    let mut kvs = vec![kv_str("general.architecture", "llama")];
    if let Some(value) = alignment {
        kvs.push(kv_u32("general.alignment", value));
    }
    kvs
}

struct TensorSpec {
    name: String,
    dims: Vec<i64>,
    dtype: u32,
    offset: u64,
}

fn tensor(name: &str, dims: &[i64], dtype: u32, offset: u64) -> TensorSpec {
    TensorSpec {
        name: name.to_owned(),
        dims: dims.to_vec(),
        dtype,
        offset,
    }
}

fn declared_alignment(kvs: &[Kv]) -> u32 {
    kvs.iter()
        .find(|kv| kv.key.as_slice() == &b"general.alignment"[..] && kv.type_id == 4)
        .and_then(|kv| kv.value.get(..4))
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("u32 bytes")))
        .unwrap_or(32)
}

fn build(version: u32, kvs: &[Kv], tensors: &[TensorSpec], payload: &[u8]) -> Vec<u8> {
    build_with_counts(
        version,
        kvs.len() as u64,
        tensors.len() as u64,
        kvs,
        tensors,
        payload,
        declared_alignment(kvs),
    )
}

fn build_with_counts(
    version: u32,
    kv_count: u64,
    tensor_count: u64,
    kvs: &[Kv],
    tensors: &[TensorSpec],
    payload: &[u8],
    alignment: u32,
) -> Vec<u8> {
    let mut bytes = Bytes::default();
    bytes.raw(b"GGUF");
    bytes.u32(version);
    bytes.u64(tensor_count);
    bytes.u64(kv_count);
    for kv in kvs {
        bytes.bytes(&kv.key);
        bytes.u32(kv.type_id);
        bytes.raw(&kv.value);
    }
    for entry in tensors {
        bytes.bytes(entry.name.as_bytes());
        bytes.u32(entry.dims.len() as u32);
        for dim in &entry.dims {
            bytes.i64(*dim);
        }
        bytes.u32(entry.dtype);
        bytes.u64(entry.offset);
    }
    let step = if alignment == 0 {
        1
    } else {
        alignment as usize
    };
    while bytes.0.len() % step != 0 {
        bytes.u8(0);
    }
    bytes.raw(payload);
    bytes.0
}

fn fails(bytes: &[u8], label: &str, needle: &str) {
    let dir = TempDir::new(label);
    let path = dir.write("bad.gguf", bytes);
    let error = GgufMap::read(&path).expect_err(label);
    assert_eq!(error.kind(), ErrorKind::InvalidData, "{label}: {error}");
    assert!(error.to_string().contains(needle), "{label}: {error}");
}

fn valid_bytes() -> Vec<u8> {
    build(
        3,
        &base_kvs(None),
        &[tensor("w", &[2], 0, 0)],
        &[1, 2, 3, 4, 5, 6, 7, 8],
    )
}

#[test]
fn reads_v2_v3_metadata_tensors_and_exact_payloads() {
    for (version, alignment) in [(2u32, None), (3, Some(64u32))] {
        let label = format!("valid-v{version}-a{}", alignment.unwrap_or(32));
        let dir = TempDir::new(&label);
        let mut kvs = base_kvs(alignment);
        kvs.extend([
            kv_u32("llama.context_length", 4096),
            kv_u64("llama.embedding_length", 2560),
            kv_u32("llama.block_count", 16),
            kv_u64("llama.feed_forward_length", 1024),
            kv_u32("llama.attention_head_count", 8),
            kv_u32("llama.attention_head_count_kv", 4),
            kv_str("general.license", "apache-2.0"),
        ]);
        let tensors = vec![
            tensor("blk.0.attn.weight", &[4, 8], 0, 0),
            tensor("blk.0.ffn.weight", &[16], 1, 128),
        ];
        let payload: Vec<u8> = (0..160u16).map(|value| (value % 251) as u8).collect();
        let bytes = build(version, &kvs, &tensors, &payload);
        let path = dir.write("model.gguf", &bytes);
        let map = GgufMap::read(&path).expect("valid GGUF");

        assert_eq!(map.version(), version);
        assert_eq!(map.architecture(), "llama");
        assert_eq!(map.alignment(), alignment.unwrap_or(32));
        assert_eq!(map.data_start() % u64::from(map.alignment()), 0);
        assert_eq!(map.file_len(), bytes.len() as u64);
        assert_eq!(map.len(), 2);

        let attn = map.get("blk.0.attn.weight").expect("attention tensor");
        assert_eq!(
            (
                attn.dtype,
                attn.shape.as_slice(),
                attn.offset,
                attn.byte_len
            ),
            (0, &[4usize, 8][..], 0, 128)
        );
        let ffn = map.get("blk.0.ffn.weight").expect("ffn tensor");
        assert_eq!(
            (ffn.dtype, ffn.shape.as_slice(), ffn.offset, ffn.byte_len),
            (1, &[16usize][..], 128, 32)
        );
        assert_eq!(
            map.read_tensor(&path, "blk.0.attn.weight").expect("bytes"),
            payload[..128]
        );
        assert_eq!(
            map.read_tensor(&path, "blk.0.ffn.weight").expect("bytes"),
            payload[128..]
        );

        assert_eq!(map.context_length().expect("context"), Some(4096));
        assert_eq!(map.embedding_length().expect("embedding"), Some(2560));
        assert_eq!(map.block_count().expect("blocks"), Some(16));
        assert_eq!(map.feed_forward_length().expect("ffn"), Some(1024));
        assert_eq!(map.attention_head_count().expect("heads"), Some(8));
        assert_eq!(map.attention_head_count_kv().expect("kv heads"), Some(4));
        assert_eq!(map.arch_u64("rope_dimension_count").expect("absent"), None);
        assert_eq!(
            map.get_metadata("general.license"),
            Some(&GgufValue::String("apache-2.0".to_owned()))
        );
    }
}

#[test]
fn parses_all_scalar_kv_types_and_nested_arrays() {
    let dir = TempDir::new("all-kv-types");
    let mut kvs = base_kvs(None);
    kvs.extend([
        kv_u8("t.u8", 7),
        kv_scalar("t.i8", 1, vec![0xff]),
        kv_u16("t.u16", 300),
        kv_i16("t.i16", -300),
        kv_u32("t.u32", 70_000),
        kv_i32("t.i32", -70_000),
        kv_f32("t.f32", 1.5),
        kv_scalar("t.bool", 7, vec![1]),
        kv_str("t.str", "value"),
        kv_u64("t.u64", u64::from(u32::MAX) + 9),
        kv_i64("t.i64", -(i64::from(i32::MAX) + 9)),
        kv_f64("t.f64", 2.25),
    ]);

    let mut u32_elements = Bytes::default();
    u32_elements.u32(1);
    u32_elements.u32(2);
    u32_elements.u32(3);
    kvs.push(kv_array("t.arr.u32", 4, 3, u32_elements.0));

    let mut string_elements = Bytes::default();
    string_elements.bytes(b"a");
    string_elements.bytes(b"bb");
    kvs.push(kv_array("t.arr.str", 8, 2, string_elements.0));

    let mut inner = Bytes::default();
    inner.u32(0);
    inner.u64(2);
    inner.u8(5);
    inner.u8(6);
    let mut outer = Bytes::default();
    outer.u32(9);
    outer.u64(1);
    outer.raw(&inner.0);
    kvs.push(kv_raw(b"t.arr.nested", 9, outer.0));

    let path = dir.write("meta-only.gguf", &build(3, &kvs, &[], &[]));
    let map = GgufMap::read(&path).expect("metadata-only GGUF");
    assert!(map.is_empty());
    assert_eq!(map.get_metadata("t.u8"), Some(&GgufValue::Uint8(7)));
    assert_eq!(map.get_metadata("t.i8"), Some(&GgufValue::Int8(-1)));
    assert_eq!(map.get_metadata("t.u16"), Some(&GgufValue::Uint16(300)));
    assert_eq!(map.get_metadata("t.i16"), Some(&GgufValue::Int16(-300)));
    assert_eq!(map.get_metadata("t.u32"), Some(&GgufValue::Uint32(70_000)));
    assert_eq!(map.get_metadata("t.i32"), Some(&GgufValue::Int32(-70_000)));
    assert_eq!(map.get_metadata("t.f32"), Some(&GgufValue::Float32(1.5)));
    assert_eq!(map.get_metadata("t.bool"), Some(&GgufValue::Bool(true)));
    assert_eq!(
        map.get_metadata("t.str"),
        Some(&GgufValue::String("value".to_owned()))
    );
    assert_eq!(
        map.get_metadata("t.u64"),
        Some(&GgufValue::Uint64(u64::from(u32::MAX) + 9))
    );
    assert_eq!(
        map.get_metadata("t.i64"),
        Some(&GgufValue::Int64(-(i64::from(i32::MAX) + 9)))
    );
    assert_eq!(map.get_metadata("t.f64"), Some(&GgufValue::Float64(2.25)));
    assert_eq!(
        map.get_metadata("t.arr.u32"),
        Some(&GgufValue::Array {
            elem_type: GgufType::Uint32,
            len: 3
        })
    );
    assert_eq!(
        map.get_metadata("t.arr.str"),
        Some(&GgufValue::Array {
            elem_type: GgufType::String,
            len: 2
        })
    );
    assert_eq!(
        map.get_metadata("t.arr.nested").expect("nested").kind(),
        GgufType::Array
    );
}

#[test]
fn exact_lookup_wins_over_aliases_and_reads_only_requested_bytes() {
    let dir = TempDir::new("alias-reads");
    let tensors = vec![tensor("primary", &[2], 0, 0), tensor("legacy", &[2], 0, 32)];
    let mut payload = vec![0u8; 40];
    for (index, value) in payload.iter_mut().enumerate().take(8) {
        *value = (index + 1) as u8;
    }
    for (index, value) in payload.iter_mut().enumerate().skip(32) {
        *value = (100 + index) as u8;
    }
    let bytes = build(3, &base_kvs(None), &tensors, &payload);
    let path = dir.write("aliases.gguf", &bytes);
    let map = GgufMap::read(&path).expect("valid GGUF");

    assert_eq!(
        map.get_aliased("primary", &["legacy"])
            .expect("exact")
            .offset,
        0
    );
    assert_eq!(
        map.get_aliased("absent", &["legacy", "primary"])
            .expect("ordered alias")
            .offset,
        32
    );
    assert_eq!(map.get_aliased("absent", &["none"]), None);
    assert_eq!(
        map.read_tensor_aliased(&path, "primary", &["legacy"])
            .expect("exact bytes"),
        payload[..8]
    );
    assert_eq!(
        map.read_tensor_aliased(&path, "absent", &["legacy"])
            .expect("alias bytes"),
        payload[32..]
    );
    let missing = map
        .read_tensor_aliased(&path, "absent", &["none"])
        .expect_err("missing tensor");
    assert_eq!(missing.kind(), ErrorKind::InvalidData);
    assert!(missing.to_string().contains("no tensor named 'absent'"));
    assert!(map.read_tensor(&path, "missing").is_err());

    fs::write(&path, &bytes[..bytes.len() - 1]).expect("truncate after read");
    let changed = map
        .read_tensor(&path, "primary")
        .expect_err("file length changed");
    assert_eq!(changed.kind(), ErrorKind::InvalidData);
    assert!(changed.to_string().contains("file length changed"));
}

#[test]
fn computes_block_lengths_and_alignment_gaps() {
    let dir = TempDir::new("blocks-gaps");
    let tensors = vec![
        tensor("q", &[256], 12, 0),
        tensor("bf", &[4], 30, 160),
        tensor("f", &[2, 2], 1, 192),
    ];
    let payload = vec![0xa5u8; 200];
    let path = dir.write(
        "blocks.gguf",
        &build(3, &base_kvs(Some(32)), &tensors, &payload),
    );
    let map = GgufMap::read(&path).expect("valid GGUF");

    assert_eq!(map.get("q").expect("q4").byte_len, 144);
    assert_eq!(map.get("bf").expect("bf16").byte_len, 8);
    assert_eq!(map.get("f").expect("f16").byte_len, 8);
    assert_eq!(map.read_tensor(&path, "q").expect("bytes").len(), 144);
    assert_eq!(map.read_tensor(&path, "bf").expect("bytes").len(), 8);
    assert_eq!(map.read_tensor(&path, "f").expect("bytes").len(), 8);
}

#[test]
fn rejects_wrong_arch_field_types_without_coercion() {
    let dir = TempDir::new("wrong-field-type");
    let mut kvs = base_kvs(None);
    kvs.extend([
        kv_str("llama.context_length", "4096"),
        kv_f32("llama.block_count", 16.0),
    ]);
    let path = dir.write("wrong-type.gguf", &build(3, &kvs, &[], &[]));
    let map = GgufMap::read(&path).expect("valid container");

    let context = map.context_length().expect_err("string is not an integer");
    assert_eq!(context.kind(), ErrorKind::InvalidData);
    assert!(context.to_string().contains("expected uint32 or uint64"));
    let blocks = map.block_count().expect_err("float is not an integer");
    assert_eq!(blocks.kind(), ErrorKind::InvalidData);
    assert_eq!(map.arch_u64("embedding_length").expect("absent"), None);
}

fn nested_arrays(levels: usize) -> Vec<u8> {
    let mut value = Bytes::default();
    value.u32(0);
    value.u64(1);
    value.u8(1);
    for _ in 1..levels {
        let mut wrapper = Bytes::default();
        wrapper.u32(9);
        wrapper.u64(1);
        wrapper.raw(&value.0);
        value = wrapper;
    }
    value.0
}

#[test]
fn rejects_malformed_truncated_and_oversized_inputs() {
    let mut bad_magic = valid_bytes();
    bad_magic[0] = b'X';
    let mut bad_architecture = base_kvs(None);
    bad_architecture[0] = kv_str("general.architecture", "Llama");
    let mut duplicate_key = base_kvs(None);
    duplicate_key.push(kv_str("general.architecture", "llama"));
    let mut wrong_arch_type = base_kvs(None);
    wrong_arch_type[0] = kv_u32("general.architecture", 7);
    let mut zero_alignment = base_kvs(None);
    zero_alignment.push(kv_u32("general.alignment", 0));
    let mut odd_alignment = base_kvs(None);
    odd_alignment.push(kv_u32("general.alignment", 12));
    let mut huge_alignment = base_kvs(None);
    huge_alignment.push(kv_u32("general.alignment", u32::MAX & !7));
    let mut bad_type = base_kvs(None);
    bad_type.push(kv_raw(b"t.bad", 13, vec![]));
    let mut bad_bool = base_kvs(None);
    bad_bool.push(kv_scalar("t.bool", 7, vec![2]));
    let mut invalid_key = base_kvs(None);
    invalid_key.push(kv_raw(&[0xff], 8, vec![]));
    let mut empty_key = base_kvs(None);
    empty_key.push(kv_raw(b"", 8, vec![]));
    let mut huge_string = base_kvs(None);
    huge_string.push(kv_raw(b"t.huge", 8, u64::MAX.to_le_bytes().to_vec()));
    let mut bad_array_type = base_kvs(None);
    bad_array_type.push(kv_array("t.arr", 13, 0, vec![]));
    let mut overflow_array = base_kvs(None);
    overflow_array.push(kv_array("t.arr", 10, u64::MAX, vec![]));
    let mut oversized_array = base_kvs(None);
    oversized_array.push(kv_array("t.arr", 0, (16 * 1024 * 1024) + 1, vec![]));
    let mut deep_array = base_kvs(None);
    deep_array.push(kv_raw(b"t.deep", 9, nested_arrays(5)));

    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        ("bad magic", bad_magic, "bad magic"),
        (
            "unsupported version",
            build(1, &base_kvs(None), &[], &[]),
            "unsupported version",
        ),
        (
            "future version",
            build(4, &base_kvs(None), &[], &[]),
            "unsupported version",
        ),
        (
            "huge KV count",
            build_with_counts(3, u64::MAX, 0, &[], &[], &[], 32),
            "KV count",
        ),
        (
            "huge tensor count",
            build_with_counts(3, 1, u64::MAX, &base_kvs(None), &[], &[], 32),
            "tensor count",
        ),
        (
            "unsupported KV type",
            build(3, &bad_type, &[], &[]),
            "unsupported type",
        ),
        (
            "unknown dtype",
            build(3, &base_kvs(None), &[tensor("w", &[2], 99, 0)], &[]),
            "unsupported dtype",
        ),
        (
            "invalid UTF-8 key",
            build(3, &invalid_key, &[], &[]),
            "invalid UTF-8",
        ),
        (
            "invalid metadata key",
            build(3, &empty_key, &[], &[]),
            "invalid metadata key",
        ),
        (
            "duplicate metadata key",
            build(3, &duplicate_key, &[], &[]),
            "duplicate metadata key",
        ),
        (
            "invalid boolean",
            build(3, &bad_bool, &[], &[]),
            "invalid boolean",
        ),
        (
            "missing architecture",
            build(3, &[kv_u32("llama.context_length", 4096)], &[], &[]),
            "missing 'general.architecture'",
        ),
        (
            "non-string architecture",
            build(3, &wrong_arch_type, &[], &[]),
            "must be a string",
        ),
        (
            "uppercase architecture",
            build(3, &bad_architecture, &[], &[]),
            "invalid general.architecture",
        ),
        (
            "zero alignment",
            build(3, &zero_alignment, &[], &[]),
            "general.alignment",
        ),
        (
            "non-multiple alignment",
            build(3, &odd_alignment, &[], &[]),
            "general.alignment",
        ),
        (
            "overflowing alignment",
            build_with_counts(
                3,
                huge_alignment.len() as u64,
                0,
                &huge_alignment,
                &[],
                &[],
                32,
            ),
            "general.alignment",
        ),
        (
            "duplicate tensor name",
            build(
                3,
                &base_kvs(None),
                &[tensor("w", &[2], 0, 0), tensor("w", &[2], 0, 8)],
                &[0u8; 16],
            ),
            "duplicate tensor name",
        ),
        (
            "invalid tensor name",
            build(
                3,
                &base_kvs(None),
                &[tensor("bad name", &[2], 0, 0)],
                &[0u8; 8],
            ),
            "invalid tensor name",
        ),
        (
            "zero dimension",
            build(3, &base_kvs(None), &[tensor("w", &[0], 0, 0)], &[]),
            "nonpositive dimension",
        ),
        (
            "excessive rank",
            build(
                3,
                &base_kvs(None),
                &[tensor("w", &[1, 1, 1, 1, 1], 0, 0)],
                &[],
            ),
            "unsupported dimension count",
        ),
        (
            "shape overflow",
            build(
                3,
                &base_kvs(None),
                &[tensor("w", &[i64::MAX, 4], 0, 0)],
                &[],
            ),
            "shape element overflow",
        ),
        (
            "q4_k block mismatch",
            build(3, &base_kvs(None), &[tensor("w", &[1], 12, 0)], &[]),
            "not a multiple of 256",
        ),
        (
            "oversized string",
            build(3, &huge_string, &[], &[]),
            "exceeds 64 KiB",
        ),
        (
            "bad array element type",
            build(3, &bad_array_type, &[], &[]),
            "unsupported array type",
        ),
        (
            "array byte overflow",
            build(3, &overflow_array, &[], &[]),
            "array byte size overflow",
        ),
        (
            "array over prologue budget",
            build(3, &oversized_array, &[], &[]),
            "exceeds prologue budget",
        ),
        (
            "array nesting too deep",
            build(3, &deep_array, &[], &[]),
            "array nesting exceeds 4 levels",
        ),
        (
            "unaligned tensor offset",
            build(3, &base_kvs(None), &[tensor("w", &[2], 0, 8)], &[0u8; 8]),
            "not aligned to 32",
        ),
        (
            "overlapping tensors",
            build(
                3,
                &base_kvs(None),
                &[tensor("a", &[16], 0, 0), tensor("b", &[16], 0, 32)],
                &[0u8; 96],
            ),
            "overlap",
        ),
        (
            "tensor past EOF",
            build(3, &base_kvs(None), &[tensor("w", &[2], 0, 0)], &[]),
            "exceeds file length",
        ),
    ];
    for (label, bytes, needle) in cases {
        fails(&bytes, label, needle);
    }

    let valid = valid_bytes();
    for (index, prefix) in [2usize, 6, 10, 20, 24, 48, 68, 101, 110, 130]
        .into_iter()
        .enumerate()
    {
        let needle = if prefix < 102 {
            "truncated"
        } else {
            "exceeds file length"
        };
        fails(
            &valid[..prefix],
            &format!("truncated section {index}"),
            needle,
        );
    }
}

#[test]
fn gguf_dequant_container_labels_do_not_change_tensor_dtype() {
    let mut payload = vec![0; 144];
    payload[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    payload[4..8].fill(1);
    payload[12..16].fill(1);
    payload[16..].fill(0xff);
    let mut decoded = Vec::new();
    for file_type in [14_u32, 15] {
        let dir = TempDir::new(&format!("q4-k-file-type-{file_type}"));
        let mut kvs = base_kvs(None);
        kvs.push(kv_u32("general.file_type", file_type));
        let bytes = build(3, &kvs, &[tensor("w", &[256], 12, 0)], &payload);
        let path = dir.write("model.gguf", &bytes);
        let map = GgufMap::read(&path).expect("GGUF map");
        let tensor = map.get("w").expect("tensor");
        let values = decode_gguf_tensor(&map.read_tensor(&path, "w").expect("payload"), tensor)
            .expect("decode");
        assert!(values.iter().all(|&value| value == 15.0));
        decoded.push(values);
    }
    assert_eq!(decoded[0], decoded[1]);
}

#[test]
fn gguf_dequant_honors_each_tensor_dtype_not_file_label() {
    let dir = TempDir::new("mixed-dtypes");
    let mut kvs = base_kvs(None);
    kvs.push(kv_u32("general.file_type", 15));
    let mut q4_k = vec![0; 144];
    q4_k[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    q4_k[4..8].fill(1);
    q4_k[12..16].fill(1);
    q4_k[16..].fill(0xff);
    let f16 = 0x3c00_u16.to_le_bytes().repeat(256);
    let mut payload = q4_k.clone();
    payload.resize(160, 0);
    payload.extend_from_slice(&f16);
    let bytes = build(
        3,
        &kvs,
        &[
            tensor("quant", &[256], 12, 0),
            tensor("half", &[256], 1, 160),
        ],
        &payload,
    );
    let path = dir.write("mixed.gguf", &bytes);
    let map = GgufMap::read(&path).expect("GGUF map");
    let quant = decode_gguf_tensor(
        &map.read_tensor(&path, "quant").expect("quant payload"),
        map.get("quant").expect("quant tensor"),
    )
    .expect("quant decode");
    let half = decode_gguf_tensor(
        &map.read_tensor(&path, "half").expect("half payload"),
        map.get("half").expect("half tensor"),
    )
    .expect("half decode");
    assert!(quant.iter().all(|&value| value == 15.0));
    assert!(half.iter().all(|&value| value == 1.0));
}
