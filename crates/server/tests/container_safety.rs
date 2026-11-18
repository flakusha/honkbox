//! The bounded container reader (TASK-container-safety): the crafted
//! archive corpus.
//!
//! Every fixture is built in memory with the `zip` crate — there are no
//! fixture files on disk and the reader never extracts anything, so the
//! whole corpus is bytes in a `Vec`. Each test asserts the NAMED error
//! variant, not just "something went wrong": a reader that rejects a
//! traversal with a different variant than a symlink is a reader whose
//! contract drifted.
//!
//! The bomb fixture is the interesting one: it declares 8 MiB and holds
//! 8 MiB of zeros, and a second copy of it has its header rewritten to
//! UNDERSTATE the entry — the declared size is only a claim, so the
//! streamed bytes are what the cap must stop.
//!
//! # Resource contract (parallel safety)
//!
//! Each test owns ONLY its own in-memory bytes: it builds its archive
//! with `zip::ZipWriter` into a `Vec`, reads it through the reader, and
//! drops both. There is no fixture file, no temp directory, no fixed
//! path, no port, no environment variable, no process-global registry
//! and no shared mutable state — so the corpus needs no lock, has no
//! ordering dependence and passes in any order under the runner's
//! default parallel execution.

// No lib target on this binary crate, so include the module directly.
#[allow(dead_code)]
#[path = "../src/container.rs"]
mod container;

use std::io::{Cursor, Write};

use sha2::{Digest, Sha256};
use zip::CompressionMethod;
use zip::write::{SimpleFileOptions, ZipWriter};

use crate::container::{Container, ContainerError, ContainerLimits, GlbInfo, validate_glb};

/// The limits the corpus runs with: generous enough that only the rule
/// under test fires.
fn limits() -> ContainerLimits {
    ContainerLimits {
        max_entries: 64,
        max_entry_bytes: 1024 * 1024,
        max_total_bytes: 8 * 1024 * 1024,
    }
}

/// Build a zip archive in memory from `(name, bytes)` pairs.
fn zip_of(entries: &[(&str, &[u8])], method: CompressionMethod) -> Vec<u8> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    for (name, body) in entries {
        writer
            .start_file(
                *name,
                SimpleFileOptions::default().compression_method(method),
            )
            .expect("start the zip entry");
        writer.write_all(body).expect("write the zip entry");
    }
    writer.finish().expect("finish the zip").into_inner()
}

/// Lowercase-hex SHA-256 of `bytes` — the digest the reader must report.
fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Rewrite the uncompressed size the archive DECLARES for its single
/// entry, so the archive understates what it holds (the classic bomb
/// shape: a tiny header, an enormous expansion).
fn patch_declared_size(bytes: &mut [u8], size: u32) {
    let eocd = bytes
        .windows(4)
        .rposition(|window| window == b"PK\x05\x06")
        .expect("the archive must carry an end of central directory record");
    let directory = usize::try_from(u32::from_le_bytes([
        bytes[eocd + 16],
        bytes[eocd + 17],
        bytes[eocd + 18],
        bytes[eocd + 19],
    ]))
    .expect("the central directory offset must fit");
    // Central directory header: the uncompressed size sits 24 bytes in.
    bytes[directory + 24..directory + 28].copy_from_slice(&size.to_le_bytes());
    // Local file header: the uncompressed size sits 22 bytes in.
    bytes[22..26].copy_from_slice(&size.to_le_bytes());
}

/// Build a GLB container: the 12-byte header plus a JSON chunk and an
/// optional binary chunk.
fn glb(json: &[u8], bin: Option<&[u8]>) -> Vec<u8> {
    let mut body = Vec::new();
    push_chunk(&mut body, 0x4E4F_534A, json);
    if let Some(bin) = bin {
        push_chunk(&mut body, 0x004E_4942, bin);
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"glTF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    let length = u32::try_from(12 + body.len()).expect("the container length must fit");
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&body);
    bytes
}

/// Append one GLB chunk (length, type, payload), padded to four bytes.
fn push_chunk(bytes: &mut Vec<u8>, kind: u32, payload: &[u8]) {
    let length = u32::try_from(payload.len()).expect("the chunk length must fit");
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(payload);
    let padding = (4 - (payload.len() % 4)) % 4;
    bytes.resize(bytes.len() + padding, 0u8);
}

#[test]
fn reads_a_well_formed_zip() {
    let bytes = zip_of(
        &[("a.txt", b"hello"), ("dir/b.txt", b"world")],
        CompressionMethod::Stored,
    );
    let container = Container::open_zip(&bytes, limits()).expect("the archive must open");

    let entries = container.entries();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].name, "a.txt");
    assert_eq!(entries[0].size, 5);
    assert!(entries[0].stored);
    assert!(!entries[0].is_dir);
    assert_eq!(entries[1].name, "dir/b.txt");
    assert_eq!(entries[1].size, 5);
    assert!(entries[1].stored);
    assert!(!entries[1].is_dir);
    assert!(container.entry("a.txt").is_some());
    assert!(container.entry("missing.txt").is_none());

    let file = container
        .read_entry("a.txt", limits())
        .expect("the entry must read");
    assert_eq!(file.bytes.as_slice(), b"hello".as_slice());
    assert_eq!(file.sha256, sha256_hex(b"hello"));
}

#[test]
fn rejects_a_parent_traversal_name() {
    let bytes = zip_of(&[("../evil.txt", b"x")], CompressionMethod::Stored);
    let err = Container::open_zip(&bytes, limits()).expect_err("the archive must be rejected");
    match err {
        ContainerError::UnsafePath { reason, .. } => assert_eq!(reason, "parent_traversal"),
        other => panic!("expected an unsafe path rejection, got {other:?}"),
    }
}

#[test]
fn rejects_an_absolute_name() {
    let bytes = zip_of(&[("/etc/passwd", b"x")], CompressionMethod::Stored);
    let err = Container::open_zip(&bytes, limits()).expect_err("the archive must be rejected");
    match err {
        ContainerError::UnsafePath { reason, .. } => assert_eq!(reason, "absolute"),
        other => panic!("expected an unsafe path rejection, got {other:?}"),
    }
}

#[test]
fn rejects_a_drive_letter_name() {
    let bytes = zip_of(&[("C:/evil.txt", b"x")], CompressionMethod::Stored);
    let err = Container::open_zip(&bytes, limits()).expect_err("the archive must be rejected");
    match err {
        ContainerError::UnsafePath { reason, .. } => assert_eq!(reason, "drive_letter"),
        other => panic!("expected an unsafe path rejection, got {other:?}"),
    }
}

#[test]
fn rejects_a_backslash_name() {
    let bytes = zip_of(&[("..\\evil.txt", b"x")], CompressionMethod::Stored);
    let err = Container::open_zip(&bytes, limits()).expect_err("the archive must be rejected");
    match err {
        ContainerError::UnsafePath { reason, .. } => assert_eq!(reason, "backslash"),
        other => panic!("expected an unsafe path rejection, got {other:?}"),
    }
}

#[test]
fn rejects_a_symlink_entry() {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .add_symlink(
            "link.txt",
            "target.txt",
            SimpleFileOptions::default().unix_permissions(0o120_777),
        )
        .expect("write the symlink entry");
    let bytes = writer.finish().expect("finish the zip").into_inner();

    let err = Container::open_zip(&bytes, limits()).expect_err("the archive must be rejected");
    match err {
        ContainerError::UnsafePath { reason, .. } => assert_eq!(reason, "symlink"),
        other => panic!("expected an unsafe path rejection, got {other:?}"),
    }
}

#[test]
fn rejects_too_many_entries() {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    for index in 0..20 {
        writer
            .start_file(
                format!("f{index}.txt"),
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored),
            )
            .expect("start the zip entry");
        writer.write_all(b"x").expect("write the zip entry");
    }
    let bytes = writer.finish().expect("finish the zip").into_inner();

    let err = Container::open_zip(
        &bytes,
        ContainerLimits {
            max_entries: 10,
            ..limits()
        },
    )
    .expect_err("the archive must be rejected");
    match err {
        ContainerError::TooManyEntries { limit, found } => {
            assert_eq!(limit, 10);
            assert_eq!(found, 20);
        }
        other => panic!("expected an entry count rejection, got {other:?}"),
    }
}

#[test]
fn rejects_a_decompression_bomb() {
    let bomb = vec![0u8; 8 * 1024 * 1024];
    let bytes = zip_of(
        &[("bomb.bin", bomb.as_slice())],
        CompressionMethod::Deflated,
    );
    let limits = ContainerLimits {
        max_entry_bytes: 1024 * 1024,
        ..limits()
    };

    // The DECLARED size is checked before a byte is read.
    let err = Container::open_zip(&bytes, limits).expect_err("the archive must be rejected");
    match err {
        ContainerError::EntryTooLarge { name, limit, found } => {
            assert_eq!(name, "bomb.bin");
            assert_eq!(limit, 1024 * 1024);
            assert_eq!(found, 8 * 1024 * 1024);
        }
        other => panic!("expected a per entry rejection, got {other:?}"),
    }

    // A header that UNDERSTATES the entry does not help: the decompressed
    // bytes are capped as they stream out.
    let mut lying = bytes.clone();
    patch_declared_size(&mut lying, 1024);
    let container = Container::open_zip(&lying, limits).expect("the lying header must open");
    let err = container
        .read_entry("bomb.bin", limits)
        .expect_err("the read must be rejected");
    match err {
        ContainerError::EntryTooLarge { name, limit, found } => {
            assert_eq!(name, "bomb.bin");
            assert_eq!(limit, 1024 * 1024);
            assert!(
                found > 1024 * 1024,
                "the streamed size {found} must pass the cap"
            );
        }
        other => panic!("expected a per entry rejection, got {other:?}"),
    }
}

#[test]
fn rejects_a_total_that_exceeds_the_limit() {
    let chunk = [0u8; 100];
    let bytes = zip_of(
        &[("a", &chunk), ("b", &chunk), ("c", &chunk)],
        CompressionMethod::Stored,
    );
    let err = Container::open_zip(
        &bytes,
        ContainerLimits {
            max_entry_bytes: 1000,
            max_total_bytes: 250,
            ..limits()
        },
    )
    .expect_err("the archive must be rejected");
    match err {
        ContainerError::TotalTooLarge { limit, found } => {
            assert_eq!(limit, 250);
            assert_eq!(found, 300);
        }
        other => panic!("expected a total size rejection, got {other:?}"),
    }
}

#[test]
fn rejects_a_broken_archive_without_panicking() {
    let garbage = b"this is not a zip archive at all";
    let err = Container::open_zip(garbage, limits()).expect_err("the archive must be rejected");
    assert!(
        matches!(&err, ContainerError::Malformed { .. }),
        "got {err:?}"
    );

    let bytes = zip_of(&[("a.txt", b"hello")], CompressionMethod::Stored);
    let truncated = &bytes[..bytes.len() / 2];
    let err = Container::open_zip(truncated, limits()).expect_err("the archive must be rejected");
    assert!(
        matches!(&err, ContainerError::Malformed { .. }),
        "got {err:?}"
    );
}

#[test]
fn caps_the_bytes_read_across_the_whole_container() {
    let chunk = vec![0u8; 600 * 1024];
    let bytes = zip_of(
        &[("a.bin", chunk.as_slice()), ("b.bin", chunk.as_slice())],
        CompressionMethod::Stored,
    );
    let container = Container::open_zip(&bytes, limits()).expect("the archive must open");

    let limits = ContainerLimits {
        max_entry_bytes: 1024 * 1024,
        max_total_bytes: 1024 * 1024,
        ..limits()
    };
    container
        .read_entry("a.bin", limits)
        .expect("the first read must pass");
    let err = container
        .read_entry("b.bin", limits)
        .expect_err("the second read must be rejected");
    match err {
        ContainerError::TotalTooLarge { limit, found } => {
            assert_eq!(limit, 1024 * 1024);
            assert!(
                found > 1024 * 1024,
                "the running total {found} must pass the cap"
            );
        }
        other => panic!("expected a total size rejection, got {other:?}"),
    }
}

#[test]
fn validates_a_well_formed_glb() {
    let json = br#"{"asset":{"version":"2.0"}}"#;
    let bin = [1u8, 2, 3, 4];
    let bytes = glb(json, Some(&bin));

    let info: GlbInfo = validate_glb(&bytes, limits()).expect("the container must validate");
    assert_eq!(info.version, 2);
    assert_eq!(
        info.length,
        u32::try_from(bytes.len()).expect("the length must fit")
    );
    assert_eq!(
        info.json_bytes,
        u32::try_from(json.len()).expect("the length must fit")
    );
    assert_eq!(info.bin_bytes, 4);
}

#[test]
fn accepts_a_glb_with_an_embedded_buffer() {
    let json =
        br#"{"buffers":[{"uri":"data:application/octet-stream;base64,AAAA","byteLength":3}]}"#;
    let bytes = glb(json, None);
    let info = validate_glb(&bytes, limits()).expect("the container must validate");
    assert_eq!(
        info.json_bytes,
        u32::try_from(json.len()).expect("the length must fit")
    );
    assert_eq!(info.bin_bytes, 0);
}

#[test]
fn rejects_a_glb_with_an_external_buffer() {
    let json = br#"{"buffers":[{"uri":"external.bin","byteLength":4}]}"#;
    let bytes = glb(json, None);

    let err = validate_glb(&bytes, limits()).expect_err("the container must be rejected");
    match err {
        ContainerError::UnsafePath { name, reason } => {
            assert_eq!(reason, "external_buffer");
            assert_eq!(name, "<glb json chunk>");
        }
        other => panic!("expected an external buffer rejection, got {other:?}"),
    }
}

#[test]
fn rejects_a_glb_bin_chunk_over_the_entry_limit() {
    let bin = vec![0u8; 2 * 1024 * 1024];
    let bytes = glb(br#"{"asset":{}}"#, Some(&bin));

    let err = validate_glb(
        &bytes,
        ContainerLimits {
            max_entry_bytes: 1024 * 1024,
            ..limits()
        },
    )
    .expect_err("the container must be rejected");
    match err {
        ContainerError::EntryTooLarge { name, limit, found } => {
            assert_eq!(name, "<glb bin chunk>");
            assert_eq!(limit, 1024 * 1024);
            assert_eq!(found, 2 * 1024 * 1024);
        }
        other => panic!("expected a per entry rejection, got {other:?}"),
    }
}

#[test]
fn reads_a_deflated_entry() {
    // The deflate path IS the ORA path: a layered document is a deflated
    // zip, so the decompressed bytes and their digest must be exact.
    let payload = b"the quick brown fox jumps over the lazy dog".repeat(100);
    let bytes = zip_of(
        &[("a.txt", payload.as_slice())],
        CompressionMethod::Deflated,
    );
    let container = Container::open_zip(&bytes, limits()).expect("the archive must open");
    assert!(!container.entries()[0].stored, "the entry must be deflated");

    let file = container
        .read_entry("a.txt", limits())
        .expect("the entry must read");
    assert_eq!(file.bytes, payload);
    assert_eq!(file.sha256, sha256_hex(&payload));
}

#[test]
fn reads_a_directory_entry_as_empty_bytes() {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .add_directory("dir/", SimpleFileOptions::default())
        .expect("add the directory");
    let bytes = writer.finish().expect("finish the zip").into_inner();

    let container = Container::open_zip(&bytes, limits()).expect("the archive must open");
    assert!(container.entries()[0].is_dir);
    let file = container
        .read_entry("dir/", limits())
        .expect("the entry must read");
    assert_eq!(file.bytes.len(), 0);
    assert_eq!(file.sha256, sha256_hex(b""));
}

#[test]
fn rejects_a_nul_byte_name() {
    let bytes = zip_of(&[("a\0b", b"x")], CompressionMethod::Stored);
    let err = Container::open_zip(&bytes, limits()).expect_err("the archive must be rejected");
    match err {
        ContainerError::UnsafePath { reason, .. } => assert_eq!(reason, "nul_byte"),
        other => panic!("expected an unsafe path rejection, got {other:?}"),
    }
}

#[test]
fn accepts_a_dotted_name() {
    // `a..b` is a legal file name: only a `..` PATH COMPONENT escapes the
    // container root, and the reader must not over-reject.
    let bytes = zip_of(&[("a..b", b"x"), ("a/b", b"y")], CompressionMethod::Stored);
    let container = Container::open_zip(&bytes, limits()).expect("the archive must open");
    assert_eq!(container.entries().len(), 2);
}

#[test]
fn rejects_a_glb_with_a_broken_header() {
    let mut bytes = glb(br#"{"asset":{}}"#, None);
    bytes[0] = b'X';
    let err = validate_glb(&bytes, limits()).expect_err("the container must be rejected");
    assert!(
        matches!(&err, ContainerError::Malformed { .. }),
        "got {err:?}"
    );

    let mut bytes = glb(br#"{"asset":{}}"#, None);
    bytes[4] = 3;
    let err = validate_glb(&bytes, limits()).expect_err("the container must be rejected");
    assert!(
        matches!(&err, ContainerError::Malformed { .. }),
        "got {err:?}"
    );

    let err = validate_glb(b"glTF", limits()).expect_err("the container must be rejected");
    assert!(
        matches!(&err, ContainerError::Malformed { .. }),
        "got {err:?}"
    );
}
