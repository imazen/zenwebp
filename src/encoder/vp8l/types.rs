//! Core data structures for VP8L encoding.

use alloc::vec::Vec;

/// LZ77 hash chain constants (matching libwebp).
pub const HASH_BITS: u32 = 18;
pub const HASH_SIZE: usize = 1 << HASH_BITS;
pub const MAX_LENGTH_BITS: u32 = 12;
pub const WINDOW_SIZE_BITS: u32 = 20;
/// Maximum match length (4095).
pub const MAX_LENGTH: usize = (1 << MAX_LENGTH_BITS) - 1;
/// Maximum window size (1M - 120 for 2D codes).
pub const WINDOW_SIZE: usize = (1 << WINDOW_SIZE_BITS) - 120;
/// Minimum profitable match length.
pub const MIN_LENGTH: usize = 4;

/// Number of length prefix codes (256 literals + 24 length codes).
pub const NUM_LITERAL_CODES: usize = 256;
pub const NUM_LENGTH_CODES: usize = 24;
/// Number of distance codes.
pub const NUM_DISTANCE_CODES: usize = 40;

/// Alphabet sizes for the 5 Huffman trees.
pub const ALPHABET_SIZE_GREEN: usize = NUM_LITERAL_CODES + NUM_LENGTH_CODES; // 280 without cache
pub const ALPHABET_SIZE_RED: usize = 256;
pub const ALPHABET_SIZE_BLUE: usize = 256;
pub const ALPHABET_SIZE_ALPHA: usize = 256;
pub const ALPHABET_SIZE_DISTANCE: usize = NUM_DISTANCE_CODES;

/// Encoder quality/speed tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp8lQuality {
    /// Quality level 0-100.
    pub quality: u8,
    /// Method 0-6 (speed/quality tradeoff).
    pub method: u8,
}

impl Default for Vp8lQuality {
    fn default() -> Self {
        Self {
            quality: 75,
            method: 4,
        }
    }
}

impl Vp8lQuality {
    /// Maximum hash chain iterations based on quality.
    pub fn max_iters(&self) -> usize {
        8 + (self.quality as usize * self.quality as usize) / 128
    }

    /// Window size based on quality.
    pub fn window_size(&self, width: usize) -> usize {
        let max = if self.quality > 75 {
            WINDOW_SIZE
        } else if self.quality > 50 {
            width << 8
        } else if self.quality > 25 {
            width << 6
        } else {
            width << 4
        };
        max.min(WINDOW_SIZE)
    }
}

/// VP8L encoder configuration.
#[derive(Debug, Clone)]
pub struct Vp8lConfig {
    /// Quality and method settings.
    pub quality: Vp8lQuality,
    /// Color cache bits maximum (0 = auto-detect optimal, 1-10 = explicit max).
    pub cache_bits: u8,
    /// Use predictor transform.
    pub use_predictor: bool,
    /// Use cross-color transform.
    pub use_cross_color: bool,
    /// Use subtract green transform.
    pub use_subtract_green: bool,
    /// Use color indexing (palette) transform.
    pub use_palette: bool,
    /// Use meta-Huffman (spatially-varying codes).
    pub use_meta_huffman: bool,
    /// Predictor transform block size bits (0 = auto-detect from method, 2-8 = explicit).
    pub predictor_bits: u8,
    /// Cross-color transform block size bits (0 = auto-detect from method, 2-8 = explicit).
    pub cross_color_bits: u8,
}

impl Default for Vp8lConfig {
    fn default() -> Self {
        Self {
            quality: Vp8lQuality::default(),
            cache_bits: 0, // Auto-detect optimal cache bits
            use_predictor: true,
            use_cross_color: true, // Enabled: reduces cross-channel correlation in residuals
            use_subtract_green: true,
            use_palette: true,
            use_meta_huffman: true, // Enable meta-Huffman for spatially-varying codes
            predictor_bits: 0,      // Auto-detect from method (matching libwebp)
            cross_color_bits: 0,    // Auto-detect from method (matching libwebp)
        }
    }
}

/// A pixel or copy operation in the backward reference stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixOrCopy {
    /// Literal ARGB pixel value.
    Literal(u32),
    /// Color cache index (0-2047 depending on cache_bits).
    CacheIdx(u16),
    /// Backward reference copy.
    Copy {
        /// Copy length (4-4095).
        len: u16,
        /// Distance code (1-based).
        dist: u32,
    },
}

impl PixOrCopy {
    /// Create a literal pixel.
    #[inline]
    pub fn literal(argb: u32) -> Self {
        Self::Literal(argb)
    }

    /// Create a cache index reference.
    #[inline]
    pub fn cache_idx(idx: u16) -> Self {
        Self::CacheIdx(idx)
    }

    /// Create a backward reference copy.
    #[inline]
    pub fn copy(len: u16, dist: u32) -> Self {
        Self::Copy { len, dist }
    }

    /// Is this a literal?
    #[inline]
    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal(_))
    }

    /// Is this a cache reference?
    #[inline]
    pub fn is_cache(&self) -> bool {
        matches!(self, Self::CacheIdx(_))
    }

    /// Is this a copy?
    #[inline]
    pub fn is_copy(&self) -> bool {
        matches!(self, Self::Copy { .. })
    }

    /// Get copy length (0 if not a copy).
    #[inline]
    pub fn copy_len(&self) -> u16 {
        match self {
            Self::Copy { len, .. } => *len,
            _ => 0,
        }
    }
}

/// Backward reference storage.
#[derive(Debug, Clone, Default)]
pub struct BackwardRefs {
    /// Tokens (pixels or copies).
    pub tokens: Vec<PixOrCopy>,
}

impl BackwardRefs {
    /// Create empty backward refs.
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    /// Create with capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(cap),
        }
    }

    /// Add a token.
    #[inline]
    pub fn push(&mut self, token: PixOrCopy) {
        self.tokens.push(token);
    }

    /// Clear all tokens.
    pub fn clear(&mut self) {
        self.tokens.clear();
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Iterate over tokens.
    pub fn iter(&self) -> impl Iterator<Item = &PixOrCopy> {
        self.tokens.iter()
    }
}

/// ARGB pixel helpers.
#[inline]
pub const fn argb_alpha(argb: u32) -> u8 {
    (argb >> 24) as u8
}

#[inline]
pub const fn argb_red(argb: u32) -> u8 {
    (argb >> 16) as u8
}

#[inline]
pub const fn argb_green(argb: u32) -> u8 {
    (argb >> 8) as u8
}

#[inline]
pub const fn argb_blue(argb: u32) -> u8 {
    argb as u8
}

#[inline]
pub const fn make_argb(a: u8, r: u8, g: u8, b: u8) -> u32 {
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Subsample size calculation (ceiling division).
#[inline]
pub const fn subsample_size(size: u32, bits: u8) -> u32 {
    (size + (1 << bits) - 1) >> bits
}
