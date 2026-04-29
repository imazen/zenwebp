//! Runtime that loads the baked picker model and runs argmin
//! against zenanalyze features + `(target_zensim, log_pixels)`.
//!
//! Embeds `zenwebp_picker_v0.bin` produced by phase 4 of the spike
//! (`scripts/zenwebp_picker_distill.py` → `bake_picker.py`). Loads
//! the model once on first call (lazy `OnceLock`) and reuses the
//! `Picker` scratch state across invocations.
//!
//! Status: spike. The current bake does NOT verify schema_hash
//! against a compile-time constant — the spike's purpose is to
//! measure lift. Productionization adds the schema_hash check.

use alloc::vec::Vec;
use core::f32;

use super::spec::{CONFIGS, FEAT_COLS, N_CELLS};

/// Baked picker model (v1 binary). During the spike, the model
/// committed to `src/encoder/picker/zenwebp_picker_v0.bin` may be
/// empty before phase 4 of the spike runs — `pick_tuning_inner`
/// returns `None` for empty bytes and the caller falls back to the
/// bucket table.
static MODEL_BYTES_RAW: &[u8] = include_bytes!("zenwebp_picker_v0.bin");

/// Build features in the order the bake expects:
///   raw 14 zenanalyze + size_oh(4) + log_px + target_z_norm
/// + log_px_sq + target_z_norm_sq + target_z_norm * log_px
/// + target_z_norm * each feat_i
/// + icc_bytes (= 0 for the spike; caller can override via the
/// future API).
///
/// Returns a Vec because the size depends on the model's
/// declared n_inputs but the schema is fixed for this build.
fn engineered_features(
    raw_feats: &[f32],
    width: u32,
    height: u32,
    target_zensim: f32,
) -> Vec<f32> {
    debug_assert_eq!(
        raw_feats.len(),
        FEAT_COLS.len(),
        "raw feature count mismatch"
    );
    let pixels = (width as f32) * (height as f32);
    let log_px = libm::logf(pixels.max(1.0));
    let target_norm = target_zensim / 100.0;

    // size class one-hot — match Python `SIZE_INDEX` ordering.
    let size_oh = match (width as u64) * (height as u64) {
        n if n < 64 * 64 => [1.0_f32, 0.0, 0.0, 0.0],
        n if n < 256 * 256 => [0.0, 1.0, 0.0, 0.0],
        n if n < 1024 * 1024 => [0.0, 0.0, 1.0, 0.0],
        _ => [0.0, 0.0, 0.0, 1.0],
    };

    let mut out = Vec::with_capacity(FEAT_COLS.len() + 4 + 5 + FEAT_COLS.len() + 1);
    out.extend_from_slice(raw_feats);
    out.extend_from_slice(&size_oh);
    out.extend_from_slice(&[
        log_px,
        log_px * log_px,
        target_norm,
        target_norm * target_norm,
        target_norm * log_px,
    ]);
    for f in raw_feats {
        out.push(target_norm * f);
    }
    out.push(0.0); // icc_bytes placeholder for v0.1 spike.
    out
}

/// Pick the encoder tuning tuple via the baked picker. Falls back
/// to the bucket table when the model is missing or fails to load.
///
/// `raw_feats` must be `FEAT_COLS.len()` floats in `FEAT_COLS`
/// order. `target_zensim` is the user's target perceptual quality
/// in [0, 100]. `width`/`height` are the input image dimensions.
///
/// Returns `(sns, filter_strength, filter_sharpness, segments)` —
/// the same shape as `content_type_to_tuning`.
#[allow(dead_code)]
pub fn pick_tuning(
    raw_feats: &[f32],
    width: u32,
    height: u32,
    target_zensim: f32,
) -> (u8, u8, u8, u8) {
    pick_tuning_inner(raw_feats, width, height, target_zensim).unwrap_or_else(|| {
        // Fallback: Drawing/Default tuple from the bucket table.
        // We don't pull in the classifier here on the fallback path —
        // by the time pick_tuning is called the caller already has
        // bucket info from the same features.
        (50, 60, 0, 4)
    })
}

fn pick_tuning_inner(
    raw_feats: &[f32],
    width: u32,
    height: u32,
    target_zensim: f32,
) -> Option<(u8, u8, u8, u8)> {
    // Empty bytes = no baked model committed yet. Fallback.
    if MODEL_BYTES_RAW.is_empty() {
        return None;
    }
    // Re-align bytes for zero-copy borrow inside zenpicker.
    let aligned = align_bytes(MODEL_BYTES_RAW);
    let model = zenpicker::Model::from_bytes(&aligned).ok()?;
    let mut picker = zenpicker::Picker::new(model);
    let feats = engineered_features(raw_feats, width, height, target_zensim);
    let mask_vec = [true; N_CELLS];
    let mask = zenpicker::AllowedMask::new(&mask_vec);
    let pick = picker.argmin_masked(&feats, &mask, None).ok()??;
    if pick >= N_CELLS {
        return None;
    }
    Some(CONFIGS[pick].as_tuning())
}

fn align_bytes(src: &[u8]) -> Vec<u8> {
    // zenpicker's `from_bytes` zero-copy borrows the f32 sections.
    // include_bytes! is byte-aligned so we copy into a u64-backed
    // Vec to bring it up to 8-byte alignment.
    let n_u64 = src.len().div_ceil(8);
    let storage: Vec<u64> = alloc::vec![0; n_u64];
    let mut out = bytemuck::cast_slice::<u64, u8>(&storage).to_vec();
    out.truncate(src.len());
    out.copy_from_slice(src);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engineered_features_layout_correct() {
        let raw = [0.1_f32; 14];
        let v = engineered_features(&raw, 512, 512, 80.0);
        // 14 raw + 4 size_oh + 5 poly + 14 cross + 1 icc = 38
        assert_eq!(v.len(), 14 + 4 + 5 + 14 + 1);
        // size_oh for 512*512 = 262144 < 1024*1024 = 1048576 → "medium"
        assert_eq!(v[14..18], [0.0, 0.0, 1.0, 0.0]);
    }
}
