/// The image-ops batch cap: inputs (and thus outputs) per request.
pub(super) const IMAGE_OP_MAX_INPUTS: usize = 8;
/// `resize` edge cap (px). With the registry pixel budget
/// ([`crate::formats::MAX_DECODED_PIXELS`]) this bounds both allocation
/// and encode time on the blocking pool.
pub(super) const IMAGE_OP_MAX_EDGE: u32 = 8192;
/// Decoder-level strict caps (`image::Limits`): no decoded edge above this,
/// and no more than this much decoder allocation at once — a small PNG can
/// decode to gigabytes without these. Both caps come from the format
/// registry (the same budget table the pixel check below enforces).
pub(super) fn image_decode_limits() -> image::Limits {
    // `Limits` is `#[non_exhaustive]`: start from `default()` (512 MiB
    // allocation allowance) and tighten the strict caps.
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(crate::formats::MAX_DECODED_EDGE);
    limits.max_image_height = Some(crate::formats::MAX_DECODED_EDGE);
    // RGBA of the pixel cap is ~67 MB; intermediates stay well under
    // this allowance.
    limits.max_alloc = Some(crate::formats::MAX_DECODED_ALLOC_BYTES);
    limits
}
/// Fallback ceiling for the SVG source size, used only when the sniffed
/// registry row carries no byte budget (named, never `u64::MAX`).
pub(super) const SVG_FALLBACK_MAX_SOURCE_BYTES: u64 = 16 * 1024 * 1024;
/// Fallback ceiling for the SVG element count, used only when the sniffed
/// registry row carries no element budget.
pub(super) const SVG_FALLBACK_MAX_ELEMENTS: u64 = 100_000;
/// Fallback ceiling for the ORA archive entry count, used only when the
/// sniffed registry row carries no entry budget.
pub(super) const ORA_FALLBACK_MAX_ENTRIES: u64 = 4096;
/// Fallback ceiling for ONE ORA archive entry, used only when the sniffed
/// registry row carries no per-entry budget.
pub(super) const ORA_FALLBACK_MAX_ENTRY_BYTES: u64 = 64 * 1024 * 1024;
/// Fallback ceiling for the ORA archive's total uncompressed size, used
/// only when the sniffed registry row carries no size budget.
pub(super) const ORA_FALLBACK_MAX_TOTAL_BYTES: u64 = 512 * 1024 * 1024;
/// Fallback pixel ceiling for a rasterized vector or a flattened layered
/// document, used only when the sniffed registry row carries no pixel
/// budget.
pub(super) const RASTER_FALLBACK_MAX_PIXELS: u64 = crate::formats::MAX_DECODED_PIXELS;

/// The SVG read limits, built from the sniffed registry row: the registry
/// stays the only source of truth for limits, and the fallbacks apply only
/// to a row that carries no budget for that field.
pub(super) fn svg_read_limits(row: &crate::formats::FormatInfo) -> crate::svg::SvgLimits {
    crate::svg::SvgLimits {
        max_source_bytes: row
            .budget
            .uncompressed_bytes
            .unwrap_or(SVG_FALLBACK_MAX_SOURCE_BYTES),
        max_elements: row.budget.elements.unwrap_or(SVG_FALLBACK_MAX_ELEMENTS),
        max_pixels: row
            .budget
            .decoded_pixels
            .unwrap_or(RASTER_FALLBACK_MAX_PIXELS),
    }
}

/// The ORA read limits, built from the sniffed registry row. The
/// registry's container-entry ceiling is also the layer ceiling: a stack's
/// layers are archive entries.
pub(super) fn ora_read_limits(row: &crate::formats::FormatInfo) -> crate::ora::OraLimits {
    crate::ora::OraLimits {
        max_entries: row.budget.entries.unwrap_or(ORA_FALLBACK_MAX_ENTRIES),
        max_entry_bytes: row
            .budget
            .entry_bytes
            .unwrap_or(ORA_FALLBACK_MAX_ENTRY_BYTES),
        max_total_bytes: row
            .budget
            .uncompressed_bytes
            .unwrap_or(ORA_FALLBACK_MAX_TOTAL_BYTES),
        max_layers: row.budget.entries.unwrap_or(ORA_FALLBACK_MAX_ENTRIES),
        max_pixels: row
            .budget
            .decoded_pixels
            .unwrap_or(RASTER_FALLBACK_MAX_PIXELS),
    }
}
/// `request_id` / `client` string cap, characters.
pub(super) const IMAGE_OP_ID_MAX_CHARS: usize = 128;
/// Idempotency memory bounds (TASK-image-ops-command-dto): at most this
/// many `request_id`s remembered and this many retained response bytes —
/// oldest-first eviction, so retries replay exactly but memory stays
/// capped. Failures are never remembered (a retry re-executes).
pub(super) const IMAGE_OP_DEDUP_MAX_ENTRIES: usize = 128;
pub(super) const IMAGE_OP_DEDUP_MAX_BYTES: usize = 64 * 1024 * 1024;
