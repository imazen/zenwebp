//! Container-only edit: re-mux the VP8/VP8L payload, drop redundant
//! metadata, normalize chunk ordering.
//!
//! Zero generation loss. Wins when no recompression strategy beats the
//! source size at the target quality.

use crate::error::Error;
use crate::source::SourceAnalysis;
use zenwebp::mux::WebPMux;

/// Re-pack the input WebP with stripped EXIF / XMP and a normalized chunk
/// order. Preserves ICC by default. The decoded pixels are byte-identical
/// to the input.
pub fn run_lossless_remux(webp_bytes: &[u8], _analysis: &SourceAnalysis) -> Result<Vec<u8>, Error> {
    // Round-trip through the mux to normalize chunk layout. Strip EXIF and
    // XMP — they're metadata, not pixels. ICC is preserved (it changes the
    // displayed pixels).
    let mut mux = WebPMux::from_data(webp_bytes)
        .map_err(|e| Error::DecodeFailed(format!("mux parse failed: {e:?}")))?;

    mux.clear_exif();
    mux.clear_xmp();

    let out = mux
        .assemble()
        .map_err(|e| Error::EncodeFailed(format!("mux assemble failed: {e:?}")))?;

    // Sanity: if our re-mux somehow grew the file, ship the original
    // bytes. Defensive — should not happen, since we only ever strip.
    if out.len() > webp_bytes.len() {
        return Ok(webp_bytes.to_vec());
    }
    Ok(out)
}
