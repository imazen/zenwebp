//! Loop filter dispatch functions for VP8 decoder.
//!
//! The SIMD fast path uses a single `#[arcane]` entry point (`filter_row_simd`)
//! that calls `#[rite]` filter functions inline — all within one target_feature
//! region. The scalar functions here are the fallback when no SIMD token is available.

#![allow(clippy::too_many_arguments)]

use super::loop_filter;

/// Precomputed filter parameters for a single macroblock.
/// Computed by `calculate_filter_parameters` before entering the filter loop.
#[derive(Clone, Copy)]
pub(crate) struct MbFilterParams {
    pub filter_level: u8,
    pub interior_limit: u8,
    pub hev_threshold: u8,
    pub mbedge_limit: u8,
    pub sub_bedge_limit: u8,
    pub do_subblock_filtering: bool,
}

// ============================================================================
// Scalar fallback filter functions (used when no SIMD token is available)
// ============================================================================

/// Apply simple horizontal filter to 16 rows (scalar fallback).
#[inline]
pub(crate) fn simple_filter_horizontal_16_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    edge_limit: u8,
) {
    for y in 0usize..16 {
        let y0 = y_start + y;
        loop_filter::simple_segment_horizontal(edge_limit, &mut buf[y0 * stride + x0 - 4..][..8]);
    }
}

/// Apply simple vertical filter to 16 columns (scalar fallback).
#[inline]
pub(crate) fn simple_filter_vertical_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    edge_limit: u8,
) {
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        loop_filter::simple_segment_vertical(edge_limit, buf, point, stride);
    }
}

/// Apply normal vertical macroblock filter to 16 columns (scalar fallback).
#[inline]
pub(crate) fn normal_filter_vertical_mb_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        loop_filter::macroblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            buf,
            point,
            stride,
        );
    }
}

/// Apply normal vertical subblock filter to 16 columns (scalar fallback).
#[inline]
pub(crate) fn normal_filter_vertical_sub_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        loop_filter::subblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            buf,
            point,
            stride,
        );
    }
}

/// Apply normal horizontal macroblock filter to 16 rows (scalar fallback).
#[inline]
pub(crate) fn normal_filter_horizontal_mb_16_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for y in 0usize..16 {
        let row = y_start + y;
        loop_filter::macroblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Apply normal horizontal subblock filter to 16 rows (scalar fallback).
#[inline]
pub(crate) fn normal_filter_horizontal_sub_16_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for y in 0usize..16 {
        let row = y_start + y;
        loop_filter::subblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Apply normal horizontal macroblock filter to U and V chroma planes (scalar fallback).
#[inline]
pub(crate) fn normal_filter_horizontal_uv_mb(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for y in 0usize..8 {
        let row = y_start + y;
        loop_filter::macroblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut u_buf[row * stride + x0 - 4..][..8],
        );
        loop_filter::macroblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut v_buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Apply normal horizontal subblock filter to U and V chroma planes (scalar fallback).
#[inline]
pub(crate) fn normal_filter_horizontal_uv_sub(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for y in 0usize..8 {
        let row = y_start + y;
        loop_filter::subblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut u_buf[row * stride + x0 - 4..][..8],
        );
        loop_filter::subblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut v_buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Apply normal vertical macroblock filter to U and V chroma planes (scalar fallback).
#[inline]
pub(crate) fn normal_filter_vertical_uv_mb(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for x in 0usize..8 {
        let point = y0 * stride + x_start + x;
        loop_filter::macroblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            u_buf,
            point,
            stride,
        );
        loop_filter::macroblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            v_buf,
            point,
            stride,
        );
    }
}

/// Apply normal vertical subblock filter to U and V chroma planes (scalar fallback).
#[inline]
pub(crate) fn normal_filter_vertical_uv_sub(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
) {
    for x in 0usize..8 {
        let point = y0 * stride + x_start + x;
        loop_filter::subblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            u_buf,
            point,
            stride,
        );
        loop_filter::subblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            v_buf,
            point,
            stride,
        );
    }
}

// ============================================================================
// Single #[arcane] entry point for the entire filter-row loop.
//
// All arch-specific filter functions are #[rite] (target_feature + inline),
// so they get inlined into this single #[arcane] boundary. This eliminates
// the per-call dispatch overhead (~7.6M instructions per decode for a
// 1024x1024 image).
// ============================================================================

/// Filter a full row of macroblocks using SIMD, with a single target_feature boundary.
///
/// All `#[rite]` filter functions inline into this one target_feature region,
/// eliminating per-call dispatch overhead.
#[archmage::arcane]
pub(crate) fn filter_row_simd(
    _token: archmage::X64V3Token,
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_params: &[MbFilterParams],
) {
    let extra_uv_rows = extra_y_rows / 2;
    let mbwidth = mb_params.len();

    for mbx in 0..mbwidth {
        let p = mb_params[mbx];
        if p.filter_level == 0 {
            continue;
        }

        let mbedge_limit_i = i32::from(p.mbedge_limit);
        let sub_bedge_limit_i = i32::from(p.sub_bedge_limit);
        let hev_i = i32::from(p.hev_threshold);
        let interior_i = i32::from(p.interior_limit);

        // Filter across left of macroblock (horizontal filter on vertical edge)
        if mbx > 0 {
            if filter_type {
                super::loop_filter_avx2::simple_h_filter16(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    mbedge_limit_i,
                );
            } else {
                super::loop_filter_avx2::normal_h_filter16_edge(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                super::loop_filter_avx2::normal_h_filter_uv_edge(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        // Filter across vertical subblocks
        if p.do_subblock_filtering {
            if filter_type {
                for x in (4usize..16 - 1).step_by(4) {
                    super::loop_filter_avx2::simple_h_filter16(
                        _token,
                        cache_y,
                        mbx * 16 + x,
                        extra_y_rows,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                // Use fused 3-edge horizontal filter for luma subblocks
                super::loop_filter_avx2::normal_h_filter16i(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
                super::loop_filter_avx2::normal_h_filter_uv_inner(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8 + 4,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }

        // Filter across top of macroblock (vertical filter on horizontal edge)
        if mby > 0 {
            if filter_type {
                let point = extra_y_rows * cache_y_stride + mbx * 16;
                super::loop_filter_avx2::simple_v_filter16(
                    _token,
                    cache_y,
                    point,
                    cache_y_stride,
                    mbedge_limit_i,
                );
            } else {
                let point_y = extra_y_rows * cache_y_stride + mbx * 16;
                super::loop_filter_avx2::normal_v_filter16_edge(
                    _token,
                    cache_y,
                    point_y,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                let point_uv = extra_uv_rows * cache_uv_stride + mbx * 8;
                super::loop_filter_avx2::normal_v_filter_uv_edge(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        // Filter across horizontal subblock edges
        if p.do_subblock_filtering {
            if filter_type {
                for y in (4usize..16 - 1).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    super::loop_filter_avx2::simple_v_filter16(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                for y in (4usize..16 - 3).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    super::loop_filter_avx2::normal_v_filter16_inner(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        hev_i,
                        interior_i,
                        sub_bedge_limit_i,
                    );
                }
                let point_uv = (extra_uv_rows + 4) * cache_uv_stride + mbx * 8;
                super::loop_filter_avx2::normal_v_filter_uv_inner(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }
    }
}

/// NEON filter row entry point.
#[archmage::arcane]
pub(crate) fn filter_row_simd(
    _token: archmage::NeonToken,
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_params: &[MbFilterParams],
) {
    let extra_uv_rows = extra_y_rows / 2;
    let mbwidth = mb_params.len();

    for mbx in 0..mbwidth {
        let p = mb_params[mbx];
        if p.filter_level == 0 {
            continue;
        }

        let mbedge_limit_i = i32::from(p.mbedge_limit);
        let sub_bedge_limit_i = i32::from(p.sub_bedge_limit);
        let hev_i = i32::from(p.hev_threshold);
        let interior_i = i32::from(p.interior_limit);

        if mbx > 0 {
            if filter_type {
                super::loop_filter_neon::simple_h_filter16_neon(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    mbedge_limit_i,
                );
            } else {
                super::loop_filter_neon::normal_h_filter16_edge_neon(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                super::loop_filter_neon::normal_h_filter_uv_edge_neon(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        if p.do_subblock_filtering {
            if filter_type {
                for x in (4usize..16 - 1).step_by(4) {
                    super::loop_filter_neon::simple_h_filter16_neon(
                        _token,
                        cache_y,
                        mbx * 16 + x,
                        extra_y_rows,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                for x in (4usize..16 - 3).step_by(4) {
                    super::loop_filter_neon::normal_h_filter16_inner_neon(
                        _token,
                        cache_y,
                        mbx * 16 + x,
                        extra_y_rows,
                        cache_y_stride,
                        hev_i,
                        interior_i,
                        sub_bedge_limit_i,
                    );
                }
                super::loop_filter_neon::normal_h_filter_uv_inner_neon(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8 + 4,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }

        if mby > 0 {
            if filter_type {
                let point = extra_y_rows * cache_y_stride + mbx * 16;
                super::loop_filter_neon::simple_v_filter16_neon(
                    _token,
                    cache_y,
                    point,
                    cache_y_stride,
                    mbedge_limit_i,
                );
            } else {
                let point_y = extra_y_rows * cache_y_stride + mbx * 16;
                super::loop_filter_neon::normal_v_filter16_edge_neon(
                    _token,
                    cache_y,
                    point_y,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                let point_uv = extra_uv_rows * cache_uv_stride + mbx * 8;
                super::loop_filter_neon::normal_v_filter_uv_edge_neon(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        if p.do_subblock_filtering {
            if filter_type {
                for y in (4usize..16 - 1).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    super::loop_filter_neon::simple_v_filter16_neon(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                for y in (4usize..16 - 3).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    super::loop_filter_neon::normal_v_filter16_inner_neon(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        hev_i,
                        interior_i,
                        sub_bedge_limit_i,
                    );
                }
                let point_uv = (extra_uv_rows + 4) * cache_uv_stride + mbx * 8;
                super::loop_filter_neon::normal_v_filter_uv_inner_neon(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }
    }
}

/// WASM SIMD128 filter row entry point.
#[archmage::arcane]
pub(crate) fn filter_row_simd(
    _token: archmage::Wasm128Token,
    cache_y: &mut [u8],
    cache_u: &mut [u8],
    cache_v: &mut [u8],
    cache_y_stride: usize,
    cache_uv_stride: usize,
    extra_y_rows: usize,
    filter_type: bool,
    mby: usize,
    mb_params: &[MbFilterParams],
) {
    let extra_uv_rows = extra_y_rows / 2;
    let mbwidth = mb_params.len();

    for mbx in 0..mbwidth {
        let p = mb_params[mbx];
        if p.filter_level == 0 {
            continue;
        }

        let mbedge_limit_i = i32::from(p.mbedge_limit);
        let sub_bedge_limit_i = i32::from(p.sub_bedge_limit);
        let hev_i = i32::from(p.hev_threshold);
        let interior_i = i32::from(p.interior_limit);

        if mbx > 0 {
            if filter_type {
                super::loop_filter_wasm::simple_h_filter16_wasm(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    mbedge_limit_i,
                );
            } else {
                super::loop_filter_wasm::normal_h_filter16_edge_wasm(
                    _token,
                    cache_y,
                    mbx * 16,
                    extra_y_rows,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                super::loop_filter_wasm::normal_h_filter_uv_edge_wasm(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        if p.do_subblock_filtering {
            if filter_type {
                for x in (4usize..16 - 1).step_by(4) {
                    super::loop_filter_wasm::simple_h_filter16_wasm(
                        _token,
                        cache_y,
                        mbx * 16 + x,
                        extra_y_rows,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                for x in (4usize..16 - 3).step_by(4) {
                    super::loop_filter_wasm::normal_h_filter16_inner_wasm(
                        _token,
                        cache_y,
                        mbx * 16 + x,
                        extra_y_rows,
                        cache_y_stride,
                        hev_i,
                        interior_i,
                        sub_bedge_limit_i,
                    );
                }
                super::loop_filter_wasm::normal_h_filter_uv_inner_wasm(
                    _token,
                    cache_u,
                    cache_v,
                    mbx * 8 + 4,
                    extra_uv_rows,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }

        if mby > 0 {
            if filter_type {
                let point = extra_y_rows * cache_y_stride + mbx * 16;
                super::loop_filter_wasm::simple_v_filter16_wasm(
                    _token,
                    cache_y,
                    point,
                    cache_y_stride,
                    mbedge_limit_i,
                );
            } else {
                let point_y = extra_y_rows * cache_y_stride + mbx * 16;
                super::loop_filter_wasm::normal_v_filter16_edge_wasm(
                    _token,
                    cache_y,
                    point_y,
                    cache_y_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
                let point_uv = extra_uv_rows * cache_uv_stride + mbx * 8;
                super::loop_filter_wasm::normal_v_filter_uv_edge_wasm(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    mbedge_limit_i,
                );
            }
        }

        if p.do_subblock_filtering {
            if filter_type {
                for y in (4usize..16 - 1).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    super::loop_filter_wasm::simple_v_filter16_wasm(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        sub_bedge_limit_i,
                    );
                }
            } else {
                for y in (4usize..16 - 3).step_by(4) {
                    let point = (extra_y_rows + y) * cache_y_stride + mbx * 16;
                    super::loop_filter_wasm::normal_v_filter16_inner_wasm(
                        _token,
                        cache_y,
                        point,
                        cache_y_stride,
                        hev_i,
                        interior_i,
                        sub_bedge_limit_i,
                    );
                }
                let point_uv = (extra_uv_rows + 4) * cache_uv_stride + mbx * 8;
                super::loop_filter_wasm::normal_v_filter_uv_inner_wasm(
                    _token,
                    cache_u,
                    cache_v,
                    point_uv,
                    cache_uv_stride,
                    hev_i,
                    interior_i,
                    sub_bedge_limit_i,
                );
            }
        }
    }
}
