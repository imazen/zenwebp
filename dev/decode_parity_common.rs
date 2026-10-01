//! zenwebp side of the decoder parity sweep, shared by `decode_parity_sweep`
//! (native, compares against libwebp) and `decode_parity_dump` (any target,
//! incl. wasm32-wasip1: hashes zenwebp's outputs for an offline comparison).
//!
//! Every mode decodes one input through one public zenwebp entry point and
//! returns `(bytes, width, height)`. Modes whose zenwebp output layout has no
//! direct libwebp twin are normalized to RGBA here so the libwebp side only
//! needs `MODE_RGBA` for them; `libwebp_spec` names the libwebp call each mode
//! is compared against.

use std::borrow::Cow;

use zenwebp::decoder::UpsamplingMethod;
use zenwebp::mux::AnimationDecoder;
use zenwebp::{DecodeConfig, DecodeRequest, StreamingDecoder, WebPDecoder};

pub type Out = Result<(Vec<u8>, u32, u32), String>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Lossy,
    Lossless,
    Animated,
    Unknown,
}

pub fn kind(data: &[u8]) -> Kind {
    if data.len() < 16 || &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Kind::Unknown;
    }
    match &data[12..16] {
        b"VP8 " => Kind::Lossy,
        b"VP8L" => Kind::Lossless,
        b"VP8X" => {
            if data.len() > 20 && data[20] & 0x02 != 0 {
                return Kind::Animated;
            }
            let mut off = 12usize;
            while off + 8 <= data.len() {
                let id = &data[off..off + 4];
                let sz = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
                if id == b"VP8 " {
                    return Kind::Lossy;
                }
                if id == b"VP8L" {
                    return Kind::Lossless;
                }
                off = off.saturating_add(8).saturating_add(sz + (sz & 1));
            }
            Kind::Unknown
        }
        _ => Kind::Unknown,
    }
}

/// Which libwebp decode a mode is compared against.
/// (Unread by `decode_parity_dump`, which only digests zenwebp output.)
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct LibSpec {
    /// libwebp `WEBP_CSP_MODE` name (`"RGBA"`, `"rgbA"`, `"RGB_565"`, ... or `"YUV"`/`"ANIM"`).
    pub csp: &'static str,
    pub no_fancy: bool,
    pub dithering: u8,
}

const fn spec(csp: &'static str) -> LibSpec {
    LibSpec {
        csp,
        no_fancy: false,
        dithering: 0,
    }
}

/// Modes that apply to an input of `k`, in a fixed order.
pub fn modes(k: Kind) -> &'static [(&'static str, LibSpec)] {
    const STILL: &[(&str, LibSpec)] = &[
        ("rgba", spec("RGBA")),
        ("rgb", spec("RGB")),
        ("bgra", spec("BGRA")),
        ("bgr", spec("BGR")),
        ("argb", spec("ARGB")),
        ("rgba_premul", spec("rgbA")),
        ("bgra_premul", spec("bgrA")),
        ("argb_premul", spec("Argb")),
        ("rgb565", spec("RGB_565")),
        ("rgba4444", spec("RGBA_4444")),
        ("rgba_into_stride", spec("RGBA")),
        ("rgb_into_stride", spec("RGB")),
        ("bgra_into_stride", spec("BGRA")),
        ("argb_into_stride", spec("ARGB")),
        ("bgr_into_stride", spec("BGR")),
        ("streaming_rgba", spec("RGBA")),
        ("webpdecoder", spec("RGBA")),
        ("zc_decode", spec("RGBA")),
        ("zc_push_rgba", spec("RGBA")),
        ("zc_push_bgra", spec("RGBA")),
    ];
    const LOSSY_EXTRA: &[(&str, LibSpec)] = &[
        ("yuv", spec("YUV")),
        (
            "rgba_nofancy",
            LibSpec {
                csp: "RGBA",
                no_fancy: true,
                dithering: 0,
            },
        ),
        (
            "rgb_nofancy",
            LibSpec {
                csp: "RGB",
                no_fancy: true,
                dithering: 0,
            },
        ),
        (
            "webpdecoder_nofancy",
            LibSpec {
                csp: "RGBA",
                no_fancy: true,
                dithering: 0,
            },
        ),
        (
            "zc_push_nofancy",
            LibSpec {
                csp: "RGBA",
                no_fancy: true,
                dithering: 0,
            },
        ),
        (
            "rgba_dither50",
            LibSpec {
                csp: "RGBA",
                no_fancy: false,
                dithering: 50,
            },
        ),
        (
            "rgba_dither100",
            LibSpec {
                csp: "RGBA",
                no_fancy: false,
                dithering: 100,
            },
        ),
        (
            "rgb_dither100",
            LibSpec {
                csp: "RGB",
                no_fancy: false,
                dithering: 100,
            },
        ),
        (
            "rgba_nofancy_dither100",
            LibSpec {
                csp: "RGBA",
                no_fancy: true,
                dithering: 100,
            },
        ),
        (
            "zc_push_dither100",
            LibSpec {
                csp: "RGBA",
                no_fancy: false,
                dithering: 100,
            },
        ),
    ];
    const ANIM: &[(&str, LibSpec)] = &[("anim", spec("ANIM"))];
    static LOSSY: std::sync::OnceLock<Vec<(&'static str, LibSpec)>> = std::sync::OnceLock::new();
    match k {
        Kind::Animated => ANIM,
        Kind::Lossy => LOSSY.get_or_init(|| [STILL, LOSSY_EXTRA].concat()),
        Kind::Lossless | Kind::Unknown => STILL,
    }
}

fn e<E: core::fmt::Display>(x: E) -> String {
    x.to_string()
}

/// RGB/BGRA/BGR/ARGB → RGBA by descriptor-ish tag.
fn to_rgba(px: &[u8], layout: &str) -> Vec<u8> {
    match layout {
        "rgba" => px.to_vec(),
        "rgb" => px
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        "bgra" => px
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[2], p[1], p[0], p[3]])
            .collect(),
        "bgr" => px
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[2], p[1], p[0], 255])
            .collect(),
        other => panic!("to_rgba: {other}"),
    }
}

/// A zenwebp `decode_*_into(data, output, stride_pixels)` entry point.
type IntoFn = fn(&[u8], &mut [u8], u32) -> zenwebp::DecodeResult<(u32, u32)>;

/// `*_into` with a padded stride: fill with a sentinel, decode, require the
/// padding to be untouched, and return the tight rows.
fn into_stride(data: &[u8], bpp: usize, f: IntoFn) -> Out {
    let info = zenwebp::ImageInfo::from_webp(data).map_err(e)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let stride = w + 3;
    let mut buf = vec![0xA5u8; stride * h * bpp];
    let (rw, rh) = f(data, &mut buf, stride as u32).map_err(e)?;
    let mut tight = Vec::with_capacity(w * h * bpp);
    for y in 0..h {
        let row = &buf[y * stride * bpp..(y + 1) * stride * bpp];
        if row[w * bpp..].iter().any(|&b| b != 0xA5) {
            return Err(format!("padding clobbered in row {y}"));
        }
        tight.extend_from_slice(&row[..w * bpp]);
    }
    Ok((tight, rw, rh))
}

struct CollectSink {
    buf: Vec<u8>,
    w: u32,
    h: u32,
    desc: Option<zenpixels::PixelDescriptor>,
}

impl zencodec::decode::DecodeRowSink for CollectSink {
    fn begin(
        &mut self,
        width: u32,
        height: u32,
        descriptor: zenpixels::PixelDescriptor,
    ) -> Result<(), zencodec::decode::SinkError> {
        self.w = width;
        self.h = height;
        self.desc = Some(descriptor);
        self.buf = vec![0; width as usize * height as usize * descriptor.bytes_per_pixel()];
        Ok(())
    }
    fn provide_next_buffer(
        &mut self,
        y: u32,
        height: u32,
        width: u32,
        descriptor: zenpixels::PixelDescriptor,
    ) -> Result<zenpixels::PixelSliceMut<'_>, zencodec::decode::SinkError> {
        let row = width as usize * descriptor.bytes_per_pixel();
        let start = y as usize * row;
        zenpixels::PixelSliceMut::new(
            &mut self.buf[start..start + height as usize * row],
            width,
            height,
            row,
            descriptor,
        )
        .map_err(|x| zencodec::decode::SinkError::from(format!("{x}")))
    }
    fn finish(&mut self) -> Result<(), zencodec::decode::SinkError> {
        Ok(())
    }
}

fn desc_layout(d: zenpixels::PixelDescriptor) -> Result<&'static str, String> {
    use zenpixels::PixelDescriptor as P;
    Ok(if d == P::RGBA8_SRGB {
        "rgba"
    } else if d == P::RGB8_SRGB {
        "rgb"
    } else if d == P::BGRA8_SRGB {
        "bgra"
    } else {
        return Err(format!("unexpected descriptor {d:?}"));
    })
}

fn zc_push(
    data: &[u8],
    cfg: zenwebp::zencodec::WebpDecoderConfig,
    preferred: &[zenpixels::PixelDescriptor],
) -> Out {
    use zencodec::decode::{DecodeJob, DecoderConfig as _};
    let mut sink = CollectSink {
        buf: Vec::new(),
        w: 0,
        h: 0,
        desc: None,
    };
    cfg.job()
        .push_decoder(Cow::Borrowed(data), &mut sink, preferred)
        .map_err(e)?;
    let layout = desc_layout(sink.desc.ok_or("no begin")?)?;
    Ok((to_rgba(&sink.buf, layout), sink.w, sink.h))
}

fn request(data: &[u8], cfg: DecodeConfig, rgb: bool) -> Out {
    let r = DecodeRequest::new(&cfg, data);
    if rgb { r.decode_rgb() } else { r.decode_rgba() }.map_err(e)
}

fn webpdecoder(data: &[u8], up: UpsamplingMethod) -> Out {
    let mut d = WebPDecoder::new(data).map_err(e)?;
    d.set_lossy_upsampling(up);
    let mut buf = vec![0u8; d.output_buffer_size().ok_or("size")?];
    d.read_image(&mut buf).map_err(e)?;
    let (w, h) = d.dimensions();
    let layout = if d.has_alpha() { "rgba" } else { "rgb" };
    Ok((to_rgba(&buf, layout), w, h))
}

/// Pack YUV planes tightly (y, then u, then v).
#[allow(clippy::too_many_arguments)]
pub fn pack_planes(
    y: &[u8],
    ys: usize,
    u: &[u8],
    us: usize,
    v: &[u8],
    vs: usize,
    w: usize,
    h: usize,
) -> Option<Vec<u8>> {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    for r in 0..h {
        out.extend_from_slice(y.get(r * ys..r * ys + w)?);
    }
    for r in 0..ch {
        out.extend_from_slice(u.get(r * us..r * us + cw)?);
    }
    for r in 0..ch {
        out.extend_from_slice(v.get(r * vs..r * vs + cw)?);
    }
    Some(out)
}

/// Decode `data` through zenwebp in `mode`.
pub fn zen_decode(mode: &str, data: &[u8]) -> Out {
    use zenpixels::PixelDescriptor as P;
    use zenwebp::oneshot as o;
    use zenwebp::zencodec::WebpDecoderConfig as Zc;
    let simple = || DecodeConfig::default().upsampling(UpsamplingMethod::Simple);
    let dither = |s| DecodeConfig::default().with_dithering_strength(s);
    match mode {
        "rgba" => o::decode_rgba(data).map_err(e),
        "rgb" => o::decode_rgb(data).map_err(e),
        "bgra" => o::decode_bgra(data).map_err(e),
        "bgr" => o::decode_bgr(data).map_err(e),
        "argb" => o::decode_argb(data).map_err(e),
        "rgba_premul" => o::decode_rgba_premultiplied(data).map_err(e),
        "bgra_premul" => o::decode_bgra_premultiplied(data).map_err(e),
        "argb_premul" => o::decode_argb_premultiplied(data).map_err(e),
        "rgb565" => o::decode_rgb565(data).map_err(e),
        "rgba4444" => o::decode_rgba4444(data).map_err(e),
        "rgba_into_stride" => into_stride(data, 4, o::decode_rgba_into),
        "rgb_into_stride" => into_stride(data, 3, o::decode_rgb_into),
        "bgra_into_stride" => into_stride(data, 4, o::decode_bgra_into),
        "argb_into_stride" => into_stride(data, 4, o::decode_argb_into),
        "bgr_into_stride" => into_stride(data, 3, o::decode_bgr_into),
        "streaming_rgba" => {
            let mut s = StreamingDecoder::new();
            for chunk in data.chunks(977) {
                s.append(chunk).map_err(e)?;
            }
            s.finish_rgba().map_err(e)
        }
        "webpdecoder" => webpdecoder(data, UpsamplingMethod::Bilinear),
        "webpdecoder_nofancy" => webpdecoder(data, UpsamplingMethod::Simple),
        "zc_decode" => {
            let out = Zc::new().decode(data).map_err(e)?;
            let px = out.pixels();
            let (w, h) = (px.width(), px.rows());
            let layout = desc_layout(px.descriptor())?;
            let bpp = px.descriptor().bytes_per_pixel();
            let mut tight = Vec::with_capacity(w as usize * h as usize * bpp);
            for y in 0..h {
                tight.extend_from_slice(&px.row(y)[..w as usize * bpp]);
            }
            Ok((to_rgba(&tight, layout), w, h))
        }
        "zc_push_rgba" => zc_push(data, Zc::new(), &[P::RGBA8_SRGB]),
        "zc_push_bgra" => zc_push(data, Zc::new(), &[P::BGRA8_SRGB]),
        "zc_push_nofancy" => zc_push(
            data,
            Zc::new().with_upsampling(UpsamplingMethod::Simple),
            &[P::RGBA8_SRGB],
        ),
        "zc_push_dither100" => zc_push(
            data,
            Zc::new().with_dithering_strength(100),
            &[P::RGBA8_SRGB],
        ),
        "rgba_nofancy" => request(data, simple(), false),
        "rgb_nofancy" => request(data, simple(), true),
        "rgba_dither50" => request(data, dither(50), false),
        "rgba_dither100" => request(data, dither(100), false),
        "rgb_dither100" => request(data, dither(100), true),
        "rgba_nofancy_dither100" => request(data, simple().with_dithering_strength(100), false),
        "yuv" => {
            let p = o::decode_yuv420(data).map_err(e)?;
            let (w, h) = (p.y_width as usize, p.y_height as usize);
            let uvw = p.uv_width as usize;
            let packed = pack_planes(&p.y, w, &p.u, uvw, &p.v, uvw, w, h).ok_or("short plane")?;
            Ok((packed, p.y_width, p.y_height))
        }
        "anim" => {
            let mut d = AnimationDecoder::new(data).map_err(e)?;
            let frames = d.decode_all().map_err(e)?;
            let (w, h) = frames
                .first()
                .map(|f| (f.width, f.height))
                .unwrap_or((0, 0));
            let mut all = Vec::new();
            for f in &frames {
                if f.data.len() == (f.width * f.height * 3) as usize {
                    all.extend(to_rgba(&f.data, "rgb"));
                } else {
                    all.extend_from_slice(&f.data);
                }
            }
            Ok((all, w, h * frames.len() as u32))
        }
        other => Err(format!("unknown mode {other}")),
    }
}

/// FNV-1a 64 over `w`, `h` (LE) and the bytes: a stable, target-independent
/// digest so outputs from different targets can be compared offline.
pub fn digest(out: &Out) -> String {
    match out {
        Err(_) => "ERR".to_string(),
        Ok((b, w, h)) => {
            let mut x = 0xcbf2_9ce4_8422_2325u64;
            for byte in w
                .to_le_bytes()
                .iter()
                .chain(&h.to_le_bytes())
                .chain(b.iter())
            {
                x ^= u64::from(*byte);
                x = x.wrapping_mul(0x100_0000_01b3);
            }
            format!("{x:016x}")
        }
    }
}
