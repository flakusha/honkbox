//! The decoded-raster type shared by the vector and layered read paths
//! (TASK-svg-ora-support).
//!
//! A rasterized SVG ([`crate::svg`]) and a flattened ORA ([`crate::ora`])
//! both produce the same RGBA8 buffer, so the image-ops decode seam has
//! exactly ONE shape to consume — and both paths hand it over without a
//! copy: the pixels move into the `image` crate's buffer.

/// A decoded raster image: RGBA8 pixels, row-major, `width * height * 4`
/// bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RasterImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA8 pixels, row-major, `width * height * 4` bytes.
    pub pixels: Vec<u8>,
}

impl RasterImage {
    /// The raster's pixel count (`width * height`).
    pub(crate) fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Consume the raster into an `image::DynamicImage` (RGBA8). No copy:
    /// the pixel buffer moves.
    pub(crate) fn into_dynamic_image(self) -> image::DynamicImage {
        image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(self.width, self.height, self.pixels)
                .expect("raster pixels match the raster dimensions"),
        )
    }
}
