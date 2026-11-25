//! Strict GGUF v2/v3 metadata and tensor reader.
//!
//! Same contract as the safetensors [`super::CheckpointMap`]: metadata only at
//! construction, exact/alias-aware lookup, on-demand payload reads. This is a
//! container reader only — no dequantization, tokenizer, model graph or
//! readiness claim.

use std::collections::BTreeMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// Max metadata key/value pairs in one file.
const MAX_KVS: u64 = 4096;
/// Max bytes in one string (keys and values).
const MAX_STRING: u64 = 64 * 1024;
/// Max nested array levels (the outer array counts as one).
const MAX_ARRAY_DEPTH: u8 = 4;
/// Minimum bytes a tensor-info entry can occupy; bounds the count before work.
const MIN_TENSOR_BYTES: u64 = 8 + 4 + 4 + 8;
/// GGUF default alignment when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u32 = 32;
/// Fail-closed bound: larger alignments cannot describe a sane data section.
const MAX_ALIGNMENT: u32 = 1024 * 1024;
/// `GGML_MAX_DIMS` from the pinned ggml header.
const MAX_DIMS: u32 = 4;

/// A GGUF metadata value type (the `gguf_type` IDs 0–12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufType {
    /// `uint8` (0).
    Uint8,
    /// `int8` (1).
    Int8,
    /// `uint16` (2).
    Uint16,
    /// `int16` (3).
    Int16,
    /// `uint32` (4).
    Uint32,
    /// `int32` (5).
    Int32,
    /// `float32` (6).
    Float32,
    /// `bool` (7), stored as one byte with values 0 or 1.
    Bool,
    /// UTF-8 string (8).
    String,
    /// Array (9), with a declared element type and count.
    Array,
    /// `uint64` (10).
    Uint64,
    /// `int64` (11).
    Int64,
    /// `float64` (12).
    Float64,
}

impl GgufType {
    fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return None,
        })
    }

    /// Numeric type ID as stored in GGUF files.
    pub fn id(self) -> u32 {
        match self {
            Self::Uint8 => 0,
            Self::Int8 => 1,
            Self::Uint16 => 2,
            Self::Int16 => 3,
            Self::Uint32 => 4,
            Self::Int32 => 5,
            Self::Float32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::Uint64 => 10,
            Self::Int64 => 11,
            Self::Float64 => 12,
        }
    }

    fn width(self) -> Option<u64> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

/// One parsed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    /// `uint8` value.
    Uint8(u8),
    /// `int8` value.
    Int8(i8),
    /// `uint16` value.
    Uint16(u16),
    /// `int16` value.
    Int16(i16),
    /// `uint32` value.
    Uint32(u32),
    /// `int32` value.
    Int32(i32),
    /// `float32` value.
    Float32(f32),
    /// `bool` value.
    Bool(bool),
    /// UTF-8 string value.
    String(String),
    /// Array declaration: element type and element count. Elements are
    /// parsed and bounded but not retained; tokenizer interpretation is a
    /// separate seam.
    Array {
        /// Declared element type (`Array` for nested arrays).
        elem_type: GgufType,
        /// Declared element count.
        len: u64,
    },
    /// `uint64` value.
    Uint64(u64),
    /// `int64` value.
    Int64(i64),
    /// `float64` value.
    Float64(f64),
}

impl GgufValue {
    /// GGUF type of this value.
    pub fn kind(&self) -> GgufType {
        match self {
            Self::Uint8(_) => GgufType::Uint8,
            Self::Int8(_) => GgufType::Int8,
            Self::Uint16(_) => GgufType::Uint16,
            Self::Int16(_) => GgufType::Int16,
            Self::Uint32(_) => GgufType::Uint32,
            Self::Int32(_) => GgufType::Int32,
            Self::Float32(_) => GgufType::Float32,
            Self::Bool(_) => GgufType::Bool,
            Self::String(_) => GgufType::String,
            Self::Array { .. } => GgufType::Array,
            Self::Uint64(_) => GgufType::Uint64,
            Self::Int64(_) => GgufType::Int64,
            Self::Float64(_) => GgufType::Float64,
        }
    }
}

/// One tensor located in a GGUF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    /// Tensor name from the GGUF tensor-info table.
    pub name: String,
    /// ggml type ID (0 F32, 1 F16, 12 Q4_K, 30 BF16).
    pub dtype: u32,
    /// Dimensions in file (`ne`) order, never reversed.
    pub shape: Vec<usize>,
    /// Byte offset relative to the aligned tensor-data section.
    pub offset: usize,
    /// Exact payload byte length computed from shape and dtype.
    pub byte_len: usize,
}

/// Validated metadata and tensor map for one GGUF v2/v3 file.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufMap {
    metadata: BTreeMap<String, GgufValue>,
    tensors: BTreeMap<String, GgufTensor>,
    architecture: String,
    alignment: u32,
    version: u32,
    data_start: u64,
    file_len: u64,
}

/// ggml type ID → (values per block, bytes per block).
///
/// ponytail: storage layout only; S2 extends this table when adding decoders.
fn storage_layout(dtype: u32) -> Option<(u64, u64)> {
    match dtype {
        0 => Some((1, 4)),      // F32
        1 | 30 => Some((1, 2)), // F16, BF16
        12 => Some((256, 144)), // Q4_K
        _ => None,
    }
}

fn align_up(value: u64, alignment: u32) -> io::Result<u64> {
    let step = u64::from(alignment);
    let remainder = value % step;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(step - remainder)
            .ok_or_else(|| super::invalid("GGUF: tensor data alignment overflow"))
    }
}

fn validate_key(key: &str, kind: &str) -> io::Result<()> {
    if key.is_empty() || !key.bytes().all(|byte| byte.is_ascii_graphic()) {
        return Err(super::invalid(format!("GGUF: invalid {kind} '{key}'")));
    }
    Ok(())
}

fn validate_architecture(architecture: &str) -> io::Result<()> {
    let mut bytes = architecture.bytes();
    let valid = bytes.next().is_some_and(|first| first.is_ascii_lowercase())
        && bytes.all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || byte == b'_'
                || byte == b'-'
                || byte == b'.'
        });
    if !valid {
        return Err(super::invalid(format!(
            "GGUF: invalid general.architecture '{architecture}'"
        )));
    }
    Ok(())
}

/// Sequential prologue reader: every declaration is charged against both the
/// prologue budget and the observed file length before any allocation.
struct Bounded<'a> {
    file: &'a mut std::fs::File,
    pos: u64,
    file_len: u64,
    budget: u64,
}

impl Bounded<'_> {
    fn charge(&mut self, bytes: u64, ctx: &str) -> io::Result<()> {
        if bytes > self.budget {
            return Err(super::invalid(format!(
                "GGUF: {ctx}: exceeds 16 MiB prologue budget"
            )));
        }
        if bytes > self.file_len.saturating_sub(self.pos) {
            return Err(super::invalid(format!(
                "GGUF: {ctx}: truncated at end of file"
            )));
        }
        self.budget -= bytes;
        self.pos += bytes;
        Ok(())
    }

    fn fixed<const N: usize>(&mut self, ctx: &str) -> io::Result<[u8; N]> {
        self.charge(N as u64, ctx)?;
        let mut bytes = [0u8; N];
        self.file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn bytes(&mut self, len: u64, ctx: &str) -> io::Result<Vec<u8>> {
        self.charge(len, ctx)?;
        let len = usize::try_from(len)
            .map_err(|_| super::invalid(format!("GGUF: {ctx}: size exceeds usize")))?;
        let mut bytes = vec![0u8; len];
        self.file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn skip(&mut self, len: u64, ctx: &str) -> io::Result<()> {
        self.charge(len, ctx)?;
        let delta = i64::try_from(len)
            .map_err(|_| super::invalid(format!("GGUF: {ctx}: skip exceeds i64")))?;
        self.file.seek(SeekFrom::Current(delta))?;
        Ok(())
    }

    fn u8(&mut self, ctx: &str) -> io::Result<u8> {
        Ok(self.fixed::<1>(ctx)?[0])
    }

    fn u32(&mut self, ctx: &str) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.fixed::<4>(ctx)?))
    }

    fn u64(&mut self, ctx: &str) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.fixed::<8>(ctx)?))
    }

    fn i64(&mut self, ctx: &str) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.fixed::<8>(ctx)?))
    }

    fn string(&mut self, ctx: &str) -> io::Result<String> {
        let len = self.u64(ctx)?;
        if len > MAX_STRING {
            return Err(super::invalid(format!(
                "GGUF: {ctx}: string length {len} exceeds 64 KiB"
            )));
        }
        let bytes = self.bytes(len, ctx)?;
        String::from_utf8(bytes).map_err(|_| super::invalid(format!("GGUF: {ctx}: invalid UTF-8")))
    }

    /// Read one value, checking array counts against the remaining budget
    /// before entering a loop or reserving memory.
    fn value(&mut self, kind: GgufType, depth: u8, ctx: &str) -> io::Result<GgufValue> {
        Ok(match kind {
            GgufType::Uint8 => GgufValue::Uint8(self.u8(ctx)?),
            GgufType::Int8 => GgufValue::Int8(self.u8(ctx)? as i8),
            GgufType::Uint16 => GgufValue::Uint16(u16::from_le_bytes(self.fixed::<2>(ctx)?)),
            GgufType::Int16 => GgufValue::Int16(i16::from_le_bytes(self.fixed::<2>(ctx)?)),
            GgufType::Uint32 => GgufValue::Uint32(self.u32(ctx)?),
            GgufType::Int32 => GgufValue::Int32(self.u32(ctx)? as i32),
            GgufType::Float32 => GgufValue::Float32(f32::from_le_bytes(self.fixed::<4>(ctx)?)),
            GgufType::Bool => {
                let raw = self.u8(ctx)?;
                if raw > 1 {
                    return Err(super::invalid(format!(
                        "GGUF: {ctx}: invalid boolean {raw}"
                    )));
                }
                GgufValue::Bool(raw == 1)
            }
            GgufType::String => GgufValue::String(self.string(ctx)?),
            GgufType::Array => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(super::invalid(format!(
                        "GGUF: {ctx}: array nesting exceeds {MAX_ARRAY_DEPTH} levels"
                    )));
                }
                let elem_id = self.u32(ctx)?;
                let elem_type = GgufType::from_id(elem_id).ok_or_else(|| {
                    super::invalid(format!("GGUF: {ctx}: unsupported array type {elem_id}"))
                })?;
                let count = self.u64(ctx)?;
                let count_usize = usize::try_from(count).map_err(|_| {
                    super::invalid(format!("GGUF: {ctx}: array count exceeds usize"))
                })?;
                if let Some(width) = elem_type.width() {
                    let bytes = width.checked_mul(count).ok_or_else(|| {
                        super::invalid(format!("GGUF: {ctx}: array byte size overflow"))
                    })?;
                    if bytes > self.budget {
                        return Err(super::invalid(format!(
                            "GGUF: {ctx}: array byte size exceeds prologue budget"
                        )));
                    }
                    if elem_type == GgufType::Bool {
                        for _ in 0..count_usize {
                            let raw = self.u8(ctx)?;
                            if raw > 1 {
                                return Err(super::invalid(format!(
                                    "GGUF: {ctx}: invalid boolean {raw}"
                                )));
                            }
                        }
                    } else {
                        self.skip(bytes, ctx)?;
                    }
                } else {
                    if count > self.budget {
                        return Err(super::invalid(format!(
                            "GGUF: {ctx}: array count exceeds prologue budget"
                        )));
                    }
                    for _ in 0..count_usize {
                        self.value(elem_type, depth + 1, ctx)?;
                    }
                }
                GgufValue::Array {
                    elem_type,
                    len: count,
                }
            }
            GgufType::Uint64 => GgufValue::Uint64(self.u64(ctx)?),
            GgufType::Int64 => GgufValue::Int64(self.i64(ctx)?),
            GgufType::Float64 => GgufValue::Float64(f64::from_le_bytes(self.fixed::<8>(ctx)?)),
        })
    }
}

impl GgufMap {
    /// Read and validate the metadata and tensor table of `path`. No payload
    /// bytes are decoded or copied.
    pub fn read(path: &Path) -> io::Result<Self> {
        let file_len = std::fs::metadata(path)?.len();
        let mut file = std::fs::File::open(path)?;
        let mut reader = Bounded {
            file: &mut file,
            pos: 0,
            file_len,
            budget: super::MAX_HEADER,
        };

        if reader.fixed::<4>("magic")? != *b"GGUF" {
            return Err(super::invalid("GGUF: bad magic"));
        }
        let version = reader.u32("version")?;
        if version != 2 && version != 3 {
            return Err(super::invalid(format!(
                "GGUF: unsupported version {version}"
            )));
        }
        let tensor_count = reader.u64("tensor count")?;
        let kv_count = reader.u64("KV count")?;
        if kv_count > MAX_KVS {
            return Err(super::invalid(format!(
                "GGUF: KV count {kv_count} exceeds {MAX_KVS}"
            )));
        }
        if tensor_count > reader.budget / MIN_TENSOR_BYTES {
            return Err(super::invalid(format!(
                "GGUF: tensor count {tensor_count} exceeds prologue budget"
            )));
        }
        let tensor_count = usize::try_from(tensor_count)
            .map_err(|_| super::invalid("GGUF: tensor count exceeds usize"))?;

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = reader.string("metadata key")?;
            validate_key(&key, "metadata key")?;
            if metadata.contains_key(&key) {
                return Err(super::invalid(format!(
                    "GGUF: duplicate metadata key '{key}'"
                )));
            }
            let ctx = format!("metadata '{key}'");
            let type_id = reader.u32(&ctx)?;
            let kind = GgufType::from_id(type_id).ok_or_else(|| {
                super::invalid(format!("GGUF: {ctx}: unsupported type {type_id}"))
            })?;
            let value = reader.value(kind, 0, &ctx)?;
            metadata.insert(key, value);
        }

        let architecture = match metadata.get("general.architecture") {
            Some(GgufValue::String(value)) => value.clone(),
            Some(_) => {
                return Err(super::invalid(
                    "GGUF: 'general.architecture' must be a string",
                ));
            }
            None => return Err(super::invalid("GGUF: missing 'general.architecture'")),
        };
        validate_architecture(&architecture)?;
        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(GgufValue::Uint32(value))
                if *value != 0 && *value % 8 == 0 && *value <= MAX_ALIGNMENT =>
            {
                *value
            }
            Some(_) => {
                return Err(super::invalid(
                    "GGUF: 'general.alignment' must be a nonzero uint32 multiple of 8",
                ));
            }
        };

        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = reader.string("tensor name")?;
            validate_key(&name, "tensor name")?;
            if tensors.contains_key(&name) {
                return Err(super::invalid(format!(
                    "GGUF: duplicate tensor name '{name}'"
                )));
            }
            let ctx = format!("tensor '{name}'");
            let dims = reader.u32(&ctx)?;
            if dims == 0 || dims > MAX_DIMS {
                return Err(super::invalid(format!(
                    "GGUF: {ctx}: unsupported dimension count {dims}"
                )));
            }
            let mut shape = Vec::with_capacity(dims as usize);
            let mut elements = 1u64;
            for _ in 0..dims {
                let dim = reader.i64(&ctx)?;
                if dim <= 0 {
                    return Err(super::invalid(format!(
                        "GGUF: {ctx}: nonpositive dimension"
                    )));
                }
                let dim = u64::try_from(dim)
                    .map_err(|_| super::invalid(format!("GGUF: {ctx}: invalid dimension")))?;
                elements = elements.checked_mul(dim).ok_or_else(|| {
                    super::invalid(format!("GGUF: {ctx}: shape element overflow"))
                })?;
                shape.push(usize::try_from(dim).map_err(|_| {
                    super::invalid(format!("GGUF: {ctx}: dimension exceeds usize"))
                })?);
            }
            let dtype = reader.u32(&ctx)?;
            let (values_per_block, block_bytes) = storage_layout(dtype)
                .ok_or_else(|| super::invalid(format!("GGUF: {ctx}: unsupported dtype {dtype}")))?;
            if elements % values_per_block != 0 {
                return Err(super::invalid(format!(
                    "GGUF: {ctx}: element count not a multiple of {values_per_block}"
                )));
            }
            let byte_len = elements
                .checked_div(values_per_block)
                .and_then(|blocks| blocks.checked_mul(block_bytes))
                .ok_or_else(|| super::invalid(format!("GGUF: {ctx}: byte length overflow")))?;
            let offset = reader.u64(&ctx)?;
            if offset % u64::from(alignment) != 0 {
                return Err(super::invalid(format!(
                    "GGUF: {ctx}: offset {offset} not aligned to {alignment}"
                )));
            }
            tensors.insert(
                name.clone(),
                GgufTensor {
                    name,
                    dtype,
                    shape,
                    offset: usize::try_from(offset).map_err(|_| {
                        super::invalid(format!("GGUF: {ctx}: offset exceeds usize"))
                    })?,
                    byte_len: usize::try_from(byte_len).map_err(|_| {
                        super::invalid(format!("GGUF: {ctx}: byte length exceeds usize"))
                    })?,
                },
            );
        }

        let data_start = align_up(reader.pos, alignment)?;
        let mut ranges = Vec::with_capacity(tensors.len());
        for tensor in tensors.values() {
            let ctx = format!("tensor '{}'", tensor.name);
            let offset = u64::try_from(tensor.offset)
                .map_err(|_| super::invalid(format!("GGUF: {ctx}: offset exceeds u64")))?;
            let start = data_start
                .checked_add(offset)
                .ok_or_else(|| super::invalid(format!("GGUF: {ctx}: tensor offset overflow")))?;
            let byte_len = u64::try_from(tensor.byte_len)
                .map_err(|_| super::invalid(format!("GGUF: {ctx}: byte length exceeds u64")))?;
            let end = start
                .checked_add(byte_len)
                .ok_or_else(|| super::invalid(format!("GGUF: {ctx}: tensor range overflow")))?;
            if end > file_len {
                return Err(super::invalid(format!(
                    "GGUF: {ctx}: tensor data exceeds file length"
                )));
            }
            ranges.push((start, end, tensor.name.as_str()));
        }
        ranges.sort_unstable();
        let mut cursor = 0u64;
        for (start, end, name) in ranges {
            if start < cursor {
                return Err(super::invalid(format!(
                    "GGUF: tensor ranges overlap at '{name}'"
                )));
            }
            cursor = end;
        }

        Ok(Self {
            metadata,
            tensors,
            architecture,
            alignment,
            version,
            data_start,
            file_len,
        })
    }

    /// Look up one tensor by exact GGUF name.
    pub fn get(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.get(name)
    }

    /// Look up one tensor by exact name, then caller-supplied aliases in order.
    /// The shared reader has no family alias table.
    pub fn get_aliased<'a>(&'a self, name: &str, aliases: &[&str]) -> Option<&'a GgufTensor> {
        self.get(name)
            .or_else(|| aliases.iter().find_map(|alias| self.get(alias)))
    }

    /// Number of tensors in the map.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// True when the file declares no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// GGUF container version (2 or 3).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Validated `general.architecture` identifier.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Effective alignment used for the tensor-data section.
    pub fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Byte offset of the aligned tensor-data section in the file.
    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    /// Total file length observed at read time.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// All parsed metadata values, including unknown keys.
    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
    }

    /// Look up one metadata value by exact key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Read a typed `<architecture>.<field>` integer (uint32/uint64 only).
    /// Missing fields yield `None`; wrong types are an error, never coerced.
    pub fn arch_u64(&self, field: &str) -> io::Result<Option<u64>> {
        let key = format!("{}.{}", self.architecture, field);
        match self.metadata.get(&key) {
            None => Ok(None),
            Some(GgufValue::Uint32(value)) => Ok(Some(u64::from(*value))),
            Some(GgufValue::Uint64(value)) => Ok(Some(*value)),
            Some(_) => Err(super::invalid(format!(
                "GGUF: metadata '{key}': expected uint32 or uint64"
            ))),
        }
    }

    /// `<architecture>.context_length`, when present and correctly typed.
    pub fn context_length(&self) -> io::Result<Option<u64>> {
        self.arch_u64("context_length")
    }

    /// `<architecture>.embedding_length`, when present and correctly typed.
    pub fn embedding_length(&self) -> io::Result<Option<u64>> {
        self.arch_u64("embedding_length")
    }

    /// `<architecture>.block_count`, when present and correctly typed.
    pub fn block_count(&self) -> io::Result<Option<u64>> {
        self.arch_u64("block_count")
    }

    /// `<architecture>.feed_forward_length`, when present and correctly typed.
    pub fn feed_forward_length(&self) -> io::Result<Option<u64>> {
        self.arch_u64("feed_forward_length")
    }

    /// `<architecture>.attention_head_count`, when present and typed.
    pub fn attention_head_count(&self) -> io::Result<Option<u64>> {
        self.arch_u64("attention_head_count")
    }

    /// `<architecture>.attention_head_count_kv`, when present and typed.
    pub fn attention_head_count_kv(&self) -> io::Result<Option<u64>> {
        self.arch_u64("attention_head_count_kv")
    }

    /// Read exactly one tensor's payload bytes from `path`.
    pub fn read_tensor(&self, path: &Path, name: &str) -> io::Result<Vec<u8>> {
        let tensor = self
            .get(name)
            .ok_or_else(|| super::invalid(format!("GGUF: no tensor named '{name}'")))?;
        self.read_range(path, name, tensor)
    }

    /// Read exactly one payload by exact name, then caller-supplied aliases.
    pub fn read_tensor_aliased(
        &self,
        path: &Path,
        name: &str,
        aliases: &[&str],
    ) -> io::Result<Vec<u8>> {
        if let Some(tensor) = self.get(name) {
            return self.read_range(path, name, tensor);
        }
        for alias in aliases {
            if let Some(tensor) = self.get(alias) {
                return self.read_range(path, alias, tensor);
            }
        }
        Err(super::invalid(format!(
            "GGUF: no tensor named '{name}' or aliases"
        )))
    }

    fn read_range(&self, path: &Path, name: &str, tensor: &GgufTensor) -> io::Result<Vec<u8>> {
        let observed = std::fs::metadata(path)?.len();
        if observed != self.file_len {
            return Err(super::invalid("GGUF: file length changed since read"));
        }
        let offset = u64::try_from(tensor.offset)
            .map_err(|_| super::invalid(format!("GGUF: tensor '{name}': offset exceeds u64")))?;
        let byte_len = u64::try_from(tensor.byte_len).map_err(|_| {
            super::invalid(format!("GGUF: tensor '{name}': byte length exceeds u64"))
        })?;
        let start = self
            .data_start
            .checked_add(offset)
            .ok_or_else(|| super::invalid(format!("GGUF: tensor '{name}': offset overflow")))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| super::invalid(format!("GGUF: tensor '{name}': range overflow")))?;
        if end > observed {
            return Err(super::invalid(format!(
                "GGUF: tensor '{name}': tensor data exceeds file length"
            )));
        }
        let mut file = std::fs::File::open(path)?;
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = vec![0u8; tensor.byte_len];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}
