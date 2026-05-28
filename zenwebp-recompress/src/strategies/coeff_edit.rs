//! VP8 coefficient-domain tighten — re-quantize per-segment and re-pack
//! the bitstream without an IDCT/FDCT round trip.
//!
//! This is the WebP analog of zenjpeg-recompress's `Preserve` strategy.
//! The full implementation requires:
//!
//! 1. Parsing the VP8 partition headers + per-segment quantizers.
//! 2. Dequantizing the AC residual coefficients per block.
//! 3. Multiplying by `quant_scale(source_q, target_zensim_a)`.
//! 4. Re-encoding the token stream with the new coefficients.
//!
//! Today this is **unshipped**: the calibration table projects worse-
//! than-pass-through size for `CoeffEdit` so the router never picks it.
//! When the module lands, the router will start consuming it without an
//! API change.
//!
//! The fallback below routes to `Reencode`, with a panic-on-debug guard
//! so we notice if the router ever picks this strategy before the real
//! implementation lands.

use crate::api::RecompressOptions;
use crate::error::Error;
use crate::source::SourceAnalysis;

pub fn run_coeff_edit(
    webp_bytes: &[u8],
    analysis: &SourceAnalysis,
    opts: &RecompressOptions,
) -> Result<Vec<u8>, Error> {
    debug_assert!(
        false,
        "router should never dispatch CoeffEdit until its implementation lands"
    );
    super::reencode::run_reencode(webp_bytes, analysis, opts)
}
