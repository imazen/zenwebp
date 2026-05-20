//! libwebp 1.6 C ABI shim over zenwebp — *prototype*.
//!
//! Goal: ship a `libwebp.{so,dll,dylib}` that satisfies the 33-function
//! subset of libwebp's C ABI which libwebp-net actually calls, backed by
//! pure-Rust zenwebp under the hood. Drop-in for libwebp-net without any
//! managed-side changes.
//!
//! Status of each entry below:
//!   `WIRED`   — implemented against zenwebp's Rust API.
//!   `STUB`    — `extern "C"` declared with the right ABI; body returns an
//!               error code or sentinel. Needs follow-up to fully wire.
//!   `PLANNED` — not declared yet; called out in DESIGN.md.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]
// FFI is inherently unsafe — confined to this crate, not zenwebp itself.

use core::ffi::{c_int, c_void};
use core::ptr;
use std::slice;

use zenwebp::encoder::PixelLayout;
use zenwebp::{EncodeRequest, ImageInfo, LosslessConfig, LossyConfig};

// =============================================================================
//  ABI versions — must match libwebp 1.6 headers exactly so the version-init
//  internals accept the encoder/decoder/mux/demux contexts allocated by
//  libwebp-net's managed code.
// =============================================================================

const WEBP_ENCODER_ABI_VERSION: c_int = 0x0210;
const WEBP_DECODER_ABI_VERSION: c_int = 0x0210;
const WEBP_MUX_ABI_VERSION: c_int = 0x0109;
const WEBP_DEMUX_ABI_VERSION: c_int = 0x0107;

/// libwebp version word: (major << 16) | (minor << 8) | revision.
/// Reported as 1.6.0 so AbiVersionCheck on the managed side passes.
const WEBP_VERSION: c_int = 0x010600;

// =============================================================================
//  Status codes / errors
// =============================================================================

#[repr(u32)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub enum VP8StatusCode {
    Ok = 0,
    OutOfMemory = 1,
    InvalidParam = 2,
    BitstreamError = 3,
    UnsupportedFeature = 4,
    Suspended = 5,
    UserAbort = 6,
    NotEnoughData = 7,
}

#[repr(u32)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub enum WebPEncodingError {
    Ok = 0,
    OutOfMemory = 1,
    BitstreamOutOfMemory = 2,
    NullParameter = 3,
    InvalidConfiguration = 4,
    BadDimension = 5,
    Partition0Overflow = 6,
    PartitionOverflow = 7,
    BadWrite = 8,
    FileTooBig = 9,
    UserAbort = 10,
}

// =============================================================================
//  WebPBitstreamFeatures — populated by WebPGetFeaturesInternal.
//  Layout must match libwebp/decode.h exactly.
// =============================================================================

#[repr(C)]
#[derive(Default)]
pub struct WebPBitstreamFeatures {
    pub width: c_int,
    pub height: c_int,
    pub has_alpha: c_int,
    pub has_animation: c_int,
    pub format: c_int, // 0=undefined, 1=lossy, 2=lossless
    pub pad: [u32; 5],
}

// =============================================================================
//  Simple lossy/lossless still encode entry points — WIRED.
//
//  All four input layouts route through zenwebp::EncodeRequest::{lossy,lossless}
//  with the appropriate PixelLayout. The output buffer must be allocated with
//  WebPMalloc so libwebp-net's WebPSafeFree (which calls our WebPFree) can free
//  it; for the prototype we allocate via Vec and leak the pointer to the
//  caller, freeing on WebPFree.
// =============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPGetEncoderVersion() -> c_int { WEBP_VERSION }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPGetDecoderVersion() -> c_int { WEBP_VERSION }

unsafe fn encode_simple(
    pixels: *const u8,
    width: c_int,
    height: c_int,
    stride: c_int,
    layout: PixelLayout,
    quality: Option<f32>, // None = lossless
    output: *mut *mut u8,
) -> usize {
    if pixels.is_null() || output.is_null() || width <= 0 || height <= 0 || stride <= 0 {
        return 0;
    }
    let needed = (stride as usize).saturating_mul(height as usize);
    let pixels_slice = unsafe { slice::from_raw_parts(pixels, needed) };
    let result = match quality {
        Some(q) => {
            let cfg = LossyConfig::new().with_quality(q);
            EncodeRequest::lossy(&cfg, pixels_slice, layout, width as u32, height as u32)
                .encode()
        }
        None => {
            let cfg = LosslessConfig::new();
            EncodeRequest::lossless(&cfg, pixels_slice, layout, width as u32, height as u32)
                .encode()
        }
    };
    match result {
        Ok(bytes) => {
            // Hand the buffer to the caller; freed via WebPFree. We use
            // Box::leak so the lifetime ends precisely when WebPFree is called.
            let boxed = bytes.into_boxed_slice();
            let len = boxed.len();
            let raw = Box::into_raw(boxed) as *mut u8;
            unsafe { *output = raw; }
            len
        }
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPEncodeRGB(rgb: *const u8, w: c_int, h: c_int, s: c_int, q: f32, out: *mut *mut u8) -> usize {
    unsafe { encode_simple(rgb, w, h, s, PixelLayout::Rgb8, Some(q), out) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPEncodeBGR(bgr: *const u8, w: c_int, h: c_int, s: c_int, q: f32, out: *mut *mut u8) -> usize {
    unsafe { encode_simple(bgr, w, h, s, PixelLayout::Bgr8, Some(q), out) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPEncodeRGBA(rgba: *const u8, w: c_int, h: c_int, s: c_int, q: f32, out: *mut *mut u8) -> usize {
    unsafe { encode_simple(rgba, w, h, s, PixelLayout::Rgba8, Some(q), out) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPEncodeBGRA(bgra: *const u8, w: c_int, h: c_int, s: c_int, q: f32, out: *mut *mut u8) -> usize {
    unsafe { encode_simple(bgra, w, h, s, PixelLayout::Bgra8, Some(q), out) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPEncodeLosslessRGB(rgb: *const u8, w: c_int, h: c_int, s: c_int, out: *mut *mut u8) -> usize {
    unsafe { encode_simple(rgb, w, h, s, PixelLayout::Rgb8, None, out) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPEncodeLosslessBGR(bgr: *const u8, w: c_int, h: c_int, s: c_int, out: *mut *mut u8) -> usize {
    unsafe { encode_simple(bgr, w, h, s, PixelLayout::Bgr8, None, out) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPEncodeLosslessRGBA(rgba: *const u8, w: c_int, h: c_int, s: c_int, out: *mut *mut u8) -> usize {
    unsafe { encode_simple(rgba, w, h, s, PixelLayout::Rgba8, None, out) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPEncodeLosslessBGRA(bgra: *const u8, w: c_int, h: c_int, s: c_int, out: *mut *mut u8) -> usize {
    unsafe { encode_simple(bgra, w, h, s, PixelLayout::Bgra8, None, out) }
}

// =============================================================================
//  Simple decode entry points — WIRED.
// =============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPGetInfo(
    data: *const u8,
    data_size: usize,
    width: *mut c_int,
    height: *mut c_int,
) -> c_int {
    if data.is_null() { return 0; }
    let bytes = unsafe { slice::from_raw_parts(data, data_size) };
    match ImageInfo::from_bytes(bytes) {
        Ok(info) => {
            if !width.is_null()  { unsafe { *width  = info.width  as c_int }; }
            if !height.is_null() { unsafe { *height = info.height as c_int }; }
            1
        }
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPGetFeaturesInternal(
    data: *const u8,
    data_size: usize,
    features: *mut WebPBitstreamFeatures,
    abi_version: c_int,
) -> VP8StatusCode {
    if data.is_null() || features.is_null() {
        return VP8StatusCode::InvalidParam;
    }
    if (abi_version >> 8) != (WEBP_DECODER_ABI_VERSION >> 8) {
        return VP8StatusCode::InvalidParam;
    }
    let bytes = unsafe { slice::from_raw_parts(data, data_size) };
    match ImageInfo::from_bytes(bytes) {
        Ok(info) => {
            let f = WebPBitstreamFeatures {
                width: info.width as c_int,
                height: info.height as c_int,
                has_alpha: info.has_alpha as c_int,
                has_animation: info.has_animation as c_int,
                format: if info.is_lossy { 1 } else { 2 },
                pad: [0; 5],
            };
            unsafe { ptr::write(features, f); }
            VP8StatusCode::Ok
        }
        Err(_) => VP8StatusCode::BitstreamError,
    }
}

// libwebp's *Into APIs pass stride in *bytes* and rely on the WebP header for
// width/height; zenwebp's *_into helpers want stride in *pixels*. We divide
// by bytes-per-pixel to bridge the two conventions.
unsafe fn decode_into(
    data: *const u8,
    data_size: usize,
    output_buffer: *mut u8,
    output_buffer_size: usize,
    output_stride_bytes: c_int,
    bytes_per_pixel: c_int,
    decode_fn: fn(&[u8], &mut [u8], u32) -> zenwebp::DecodeResult<(u32, u32)>,
) -> *mut u8 {
    if data.is_null() || output_buffer.is_null() || output_stride_bytes <= 0 || bytes_per_pixel <= 0 {
        return ptr::null_mut();
    }
    if output_stride_bytes % bytes_per_pixel != 0 {
        return ptr::null_mut();
    }
    let stride_pixels = (output_stride_bytes / bytes_per_pixel) as u32;
    let input = unsafe { slice::from_raw_parts(data, data_size) };
    let out = unsafe { slice::from_raw_parts_mut(output_buffer, output_buffer_size) };
    match decode_fn(input, out, stride_pixels) {
        Ok(_) => output_buffer,
        Err(_) => ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPDecodeRGBAInto(d: *const u8, s: usize, o: *mut u8, os: usize, stride: c_int) -> *mut u8 {
    unsafe { decode_into(d, s, o, os, stride, 4, zenwebp::oneshot::decode_rgba_into) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPDecodeBGRAInto(d: *const u8, s: usize, o: *mut u8, os: usize, stride: c_int) -> *mut u8 {
    unsafe { decode_into(d, s, o, os, stride, 4, zenwebp::oneshot::decode_bgra_into) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPDecodeRGBInto(d: *const u8, s: usize, o: *mut u8, os: usize, stride: c_int) -> *mut u8 {
    unsafe { decode_into(d, s, o, os, stride, 3, zenwebp::oneshot::decode_rgb_into) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPDecodeBGRInto(d: *const u8, s: usize, o: *mut u8, os: usize, stride: c_int) -> *mut u8 {
    unsafe { decode_into(d, s, o, os, stride, 3, zenwebp::oneshot::decode_bgr_into) }
}

// =============================================================================
//  Memory — WIRED. WebPMalloc is unused by libwebp-net but ships for parity.
//  WebPFree pairs with the Box::into_raw in encode_simple above.
// =============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPFree(ptr: *mut c_void) {
    if ptr.is_null() { return; }
    // SAFETY: pointer originated from Box::into_raw in encode_simple. We
    // don't know the length, so this prototype leaks the slice's length.
    // Real implementation needs a parallel length table OR uses libc::malloc/free
    // so that boundary can be crossed without Rust-side knowledge of size.
    let _ = unsafe { Box::from_raw(ptr as *mut u8) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPMalloc(size: usize) -> *mut c_void {
    // Matches libwebp's WebPMalloc — caller frees via WebPFree.
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap_or(std::alloc::Layout::new::<u8>());
    unsafe { std::alloc::alloc(layout) as *mut c_void }
}

// =============================================================================
//  WebPPicture / WebPConfig pipeline — STUB.
//
//  libwebp-net's WebPEncoderConfig-based path calls:
//    WebPConfigInitInternal, WebPConfigLosslessPreset, WebPValidateConfig,
//    WebPPictureInitInternal, WebPPictureImport{RGB,RGBA,BGR,BGRA}, WebPEncode,
//    WebPPictureFree
//  These pass two large repr(C) structs (WebPConfig, WebPPicture) by pointer
//  whose layouts must exactly match libwebp/encode.h. The picture struct also
//  carries a function pointer (WebPWriterFunction) that libwebp-net uses to
//  stream the encoded output back into a managed Stream.
//
//  Implementation sketch (DESIGN.md has the full plan):
//    1. Define #[repr(C)] WebPConfig + WebPPicture matching libwebp 1.6 byte
//       layout exactly (field order, alignment, padding arrays).
//    2. WebPConfigInitInternal writes defaults into the caller's WebPConfig.
//       Validate abi_version against WEBP_ENCODER_ABI_VERSION.
//    3. WebPPictureImport* reads pixels from caller-provided pointers and
//       stashes them in private side-band storage keyed by &mut WebPPicture
//       (we cannot grow the struct, so use a HashMap<*mut WebPPicture, ...>
//       in a Mutex; or fold the imported buffer into the existing memory_argb_
//       field that's documented as opaque-to-callers).
//    4. WebPEncode reads (config, picture), runs zenwebp::EncodeRequest,
//       chunks the output, and calls picture.writer(data, len, &picture) for
//       each chunk. The writer is a libwebp-defined fn pointer the managed
//       side has populated.
//    5. WebPPictureFree clears the side-band storage.
//
//  Stub bodies return error codes so libwebp-net's advanced encode path
//  fails-fast until this is implemented.
// =============================================================================

#[repr(C)]
pub struct WebPConfig {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct WebPPicture {
    _opaque: [u8; 0],
}

// Note: the *real* WebPConfig / WebPPicture are NOT opaque — they're caller-
// allocated, fixed-layout structs with field-level access from managed code.
// The opaque placeholder here is just so the FFI signatures type-check; a
// follow-up commit will replace them with the full #[repr(C)] layout.

pub type WebPWriterFunction = Option<
    unsafe extern "C" fn(data: *const u8, data_size: usize, picture: *const WebPPicture) -> c_int,
>;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPConfigInitInternal(
    _config: *mut WebPConfig,
    _preset: c_int,
    _quality: f32,
    _abi_version: c_int,
) -> c_int {
    0 // STUB — returns failure until layout-matched config is wired
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPConfigLosslessPreset(_config: *mut WebPConfig, _level: c_int) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPValidateConfig(_config: *const WebPConfig) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPPictureInitInternal(_picture: *mut WebPPicture, _abi_version: c_int) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPPictureImportRGB(_pic: *mut WebPPicture, _rgb: *const u8, _stride: c_int) -> c_int { 0 }
#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPPictureImportBGR(_pic: *mut WebPPicture, _bgr: *const u8, _stride: c_int) -> c_int { 0 }
#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPPictureImportRGBA(_pic: *mut WebPPicture, _rgba: *const u8, _stride: c_int) -> c_int { 0 }
#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPPictureImportBGRA(_pic: *mut WebPPicture, _bgra: *const u8, _stride: c_int) -> c_int { 0 }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPEncode(_config: *const WebPConfig, _picture: *mut WebPPicture) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPPictureFree(_picture: *mut WebPPicture) {
    // STUB
}

// =============================================================================
//  Animation encoder (libwebpmux) — STUB.
//
//  libwebp-net uses:
//    WebPAnimEncoderOptionsInitInternal, WebPAnimEncoderNewInternal,
//    WebPAnimEncoderAdd, WebPAnimEncoderAssemble, WebPAnimEncoderGetError,
//    WebPAnimEncoderDelete
//
//  Backed by zenwebp::mux::anim::AnimationEncoder. State held in a heap-allocated
//  context returned as an opaque *mut.
// =============================================================================

#[repr(C)]
pub struct WebPAnimEncoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct WebPAnimEncoderOptions {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct WebPData {
    pub bytes: *const u8,
    pub size: usize,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimEncoderOptionsInitInternal(
    _options: *mut WebPAnimEncoderOptions,
    _abi_version: c_int,
) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimEncoderNewInternal(
    _width: c_int,
    _height: c_int,
    _options: *const WebPAnimEncoderOptions,
    _abi_version: c_int,
) -> *mut WebPAnimEncoder {
    ptr::null_mut() // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimEncoderAdd(
    _enc: *mut WebPAnimEncoder,
    _frame: *mut WebPPicture,
    _timestamp_ms: c_int,
    _config: *const WebPConfig,
) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimEncoderAssemble(
    _enc: *mut WebPAnimEncoder,
    _webp_data: *mut WebPData,
) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimEncoderGetError(_enc: *mut WebPAnimEncoder) -> *const u8 {
    b"shim: animation encode not yet implemented\0".as_ptr() // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimEncoderDelete(_enc: *mut WebPAnimEncoder) {
    // STUB
}

// =============================================================================
//  Animation decoder (libwebpdemux) — STUB.
//
//  Backed by zenwebp::mux::demux::WebPDemuxer + AnimationDecoder.
// =============================================================================

#[repr(C)]
pub struct WebPAnimDecoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct WebPAnimDecoderOptions {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct WebPAnimInfo {
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub loop_count: u32,
    pub bgcolor: u32,
    pub frame_count: u32,
    pub pad: [u32; 4],
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimDecoderOptionsInitInternal(
    _options: *mut WebPAnimDecoderOptions,
    _abi_version: c_int,
) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimDecoderNewInternal(
    _webp_data: *const WebPData,
    _options: *const WebPAnimDecoderOptions,
    _abi_version: c_int,
) -> *mut WebPAnimDecoder {
    ptr::null_mut() // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimDecoderGetInfo(
    _dec: *mut WebPAnimDecoder,
    _info: *mut WebPAnimInfo,
) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimDecoderGetNext(
    _dec: *mut WebPAnimDecoder,
    _buf: *mut *mut u8,
    _timestamp: *mut c_int,
) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimDecoderHasMoreFrames(_dec: *mut WebPAnimDecoder) -> c_int {
    0 // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimDecoderReset(_dec: *mut WebPAnimDecoder) {
    // STUB
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn WebPAnimDecoderDelete(_dec: *mut WebPAnimDecoder) {
    // STUB
}
