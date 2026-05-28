//! VP8 boolean entropy coder — a matched decoder/encoder pair for the
//! coefficient transcoder.
//!
//! The encoder is ported verbatim from zenwebp's `encoder::arithmetic`
//! (itself a faithful port of libwebp's `VP8BitWriter`), so re-emitted
//! streams use the exact same arithmetic a fresh encode would. The decoder
//! is a straightforward RFC 6386 §7 reader written to be the encoder's
//! exact inverse and **validated by a round-trip test** (`bool_roundtrip`)
//! — if the conventions ever diverge, that test fails.

/// kNorm[range] — normalization shift count. From libwebp bit_writer_utils.c.
#[rustfmt::skip]
const K_NORM: [u8; 128] = [
     7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  0,
];

/// kNewRange[range] = ((range + 1) << kNorm[range]) - 1.
#[rustfmt::skip]
const K_NEW_RANGE: [u8; 128] = [
  127, 127, 191, 127, 159, 191, 223, 127, 143, 159, 175, 191, 207, 223, 239,
  127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
  247, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179,
  183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239,
  243, 247, 251, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149,
  151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179,
  181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209,
  211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239,
  241, 243, 245, 247, 249, 251, 253, 127,
];

// ===========================================================================
// Decoder — RFC 6386 §7.3, libwebp VP8GetBit convention (range stored 1..255).
// ===========================================================================

/// VP8 boolean decoder over a byte slice. Canonical RFC 6386 §7.3 form:
/// `value` holds ≥8 significant bits, `range` is 128..=255, `bit_count`
/// counts bits shifted out of `value` (0..=7).
pub struct BoolDecoder<'a> {
    buf: &'a [u8],
    pos: usize,
    value: u32,
    range: u32,
    bit_count: i32,
}

impl<'a> BoolDecoder<'a> {
    /// Initialize from a partition byte slice (RFC 6386 §7.3: load the first
    /// two bytes, range = 255, bit_count = 0).
    pub fn new(buf: &'a [u8]) -> Self {
        let b0 = buf.first().copied().unwrap_or(0);
        let b1 = buf.get(1).copied().unwrap_or(0);
        BoolDecoder {
            buf,
            pos: 2,
            value: (u32::from(b0) << 8) | u32::from(b1),
            range: 255,
            bit_count: 0,
        }
    }

    #[inline]
    fn next_byte(&mut self) -> u32 {
        let b = self.buf.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        u32::from(b)
    }

    /// Read one boolean with the given probability (0..=255). Exact inverse
    /// of [`BoolEncoder::put_bit`] (validated by `bool_roundtrip`).
    #[inline]
    pub fn get_bit(&mut self, prob: u8) -> u32 {
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
        let big_split = split << 8;
        let bit;
        if self.value >= big_split {
            bit = 1;
            self.range -= split;
            self.value -= big_split;
        } else {
            bit = 0;
            self.range = split;
        }
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.bit_count += 1;
            if self.bit_count == 8 {
                self.bit_count = 0;
                self.value |= self.next_byte();
            }
        }
        bit
    }

    /// Read `n` raw bits MSB-first (each at probability 128).
    #[inline]
    pub fn get_literal(&mut self, n: u8) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.get_bit(128);
        }
        v
    }

    /// Read a flag (probability 128).
    #[inline]
    pub fn get_flag(&mut self) -> bool {
        self.get_bit(128) != 0
    }

    /// Read a magnitude of `n` bits then a sign flag; returns signed value.
    #[inline]
    pub fn get_signed_literal(&mut self, n: u8) -> i32 {
        let mag = self.get_literal(n) as i32;
        if self.get_flag() { -mag } else { mag }
    }

    /// Bytes consumed so far (for partition bookkeeping).
    pub fn position(&self) -> usize {
        self.pos
    }
}

// ===========================================================================
// Encoder — ported verbatim from zenwebp::encoder::arithmetic.
// ===========================================================================

/// VP8 boolean encoder (libwebp `VP8BitWriter` semantics).
pub struct BoolEncoder {
    buf: Vec<u8>,
    range: i32,
    value: i32,
    run: i32,
    nb_bits: i32,
}

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    pub fn new() -> Self {
        BoolEncoder {
            buf: Vec::new(),
            range: 254,
            value: 0,
            run: 0,
            nb_bits: -8,
        }
    }

    fn flush(&mut self) {
        let s = 8 + self.nb_bits;
        let bits = self.value >> s;
        self.value -= bits << s;
        self.nb_bits -= 8;
        if (bits & 0xff) != 0xff {
            if (bits & 0x100) != 0
                && let Some(last) = self.buf.last_mut()
            {
                *last = last.wrapping_add(1);
            }
            if self.run > 0 {
                let fill = if (bits & 0x100) != 0 { 0x00u8 } else { 0xFFu8 };
                for _ in 0..self.run {
                    self.buf.push(fill);
                }
                self.run = 0;
            }
            self.buf.push((bits & 0xff) as u8);
        } else {
            self.run += 1;
        }
    }

    #[inline]
    pub fn put_bit(&mut self, bit: u32, probability: u8) {
        let split = (self.range * i32::from(probability)) >> 8;
        if bit != 0 {
            self.value += split + 1;
            self.range -= split + 1;
        } else {
            self.range = split;
        }
        if self.range < 127 {
            let shift = K_NORM[self.range as usize];
            self.range = K_NEW_RANGE[self.range as usize] as i32;
            self.value <<= shift;
            self.nb_bits += shift as i32;
            if self.nb_bits > 0 {
                self.flush();
            }
        }
    }

    #[inline]
    pub fn put_flag(&mut self, flag: bool) {
        self.put_bit(flag as u32, 128);
    }

    #[inline]
    pub fn put_literal(&mut self, num_bits: u8, value: u32) {
        for bit in (0..num_bits).rev() {
            self.put_bit((value >> bit) & 1, 128);
        }
    }

    /// Write a magnitude of `n` bits then a sign flag (VP8: 1 = negative).
    #[inline]
    pub fn put_signed_literal(&mut self, num_bits: u8, value: i32) {
        self.put_literal(num_bits, value.unsigned_abs());
        self.put_flag(value < 0);
    }

    pub fn finish(mut self) -> Vec<u8> {
        let pad_bits = 9 - self.nb_bits;
        for _ in 0..pad_bits {
            self.put_bit(0, 128);
        }
        self.nb_bits = 0;
        self.flush();
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_roundtrip() {
        // Deterministic pseudo-random bit/prob stream; encode then decode and
        // assert exact recovery. This validates the decoder is the encoder's
        // inverse (the whole transcoder rests on this).
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };

        let n = 5000;
        let mut bits = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);
        for _ in 0..n {
            let r = next();
            // Probabilities span the full 1..=255 range; bits ~50/50.
            let prob = ((r >> 32) as u8).max(1);
            let bit = ((r >> 16) & 1) as u32;
            probs.push(prob);
            bits.push(bit);
        }

        let mut enc = BoolEncoder::new();
        for i in 0..n {
            enc.put_bit(bits[i], probs[i]);
        }
        let buf = enc.finish();

        let mut dec = BoolDecoder::new(&buf);
        for i in 0..n {
            let got = dec.get_bit(probs[i]);
            assert_eq!(got, bits[i], "mismatch at bit {i} (prob {})", probs[i]);
        }
    }

    #[test]
    fn literal_roundtrip() {
        let mut enc = BoolEncoder::new();
        let vals: [(u8, u32); 5] = [(8, 0xA5), (4, 0xC), (7, 0x42), (1, 1), (3, 5)];
        for &(n, v) in &vals {
            enc.put_literal(n, v);
        }
        let buf = enc.finish();
        let mut dec = BoolDecoder::new(&buf);
        for &(n, v) in &vals {
            assert_eq!(dec.get_literal(n), v, "literal {n}-bit");
        }
    }
}
