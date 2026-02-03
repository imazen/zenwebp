//! VP8L bit writer.
//!
//! Writes bits in LSB-first order as required by VP8L format.

use alloc::vec::Vec;

/// VP8L bit writer - writes bits LSB-first.
pub struct BitWriter {
    /// Output buffer.
    buffer: Vec<u8>,
    /// Current partial byte being built.
    bits: u64,
    /// Number of bits in the partial byte (0-63).
    used: u8,
}

impl BitWriter {
    /// Create a new bit writer.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            bits: 0,
            used: 0,
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(cap),
            bits: 0,
            used: 0,
        }
    }

    /// Write `n_bits` from `value` (LSB-first).
    #[inline]
    pub fn write_bits(&mut self, value: u64, n_bits: u8) {
        debug_assert!(n_bits <= 32);
        debug_assert!(n_bits == 0 || (value >> n_bits) == 0);

        self.bits |= value << self.used;
        self.used += n_bits;

        // Flush complete bytes
        while self.used >= 8 {
            self.buffer.push(self.bits as u8);
            self.bits >>= 8;
            self.used -= 8;
        }
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(bit as u64, 1);
    }

    /// Flush any remaining bits (pad with zeros).
    pub fn flush(&mut self) {
        if self.used > 0 {
            self.buffer.push(self.bits as u8);
            self.bits = 0;
            self.used = 0;
        }
    }

    /// Get the output buffer (flushes first).
    pub fn finish(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }

    /// Get current byte position (for size estimation).
    pub fn byte_position(&self) -> usize {
        self.buffer.len()
    }

    /// Get current bit position.
    pub fn bit_position(&self) -> usize {
        self.buffer.len() * 8 + self.used as usize
    }

    /// Borrow the internal buffer.
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_bits() {
        let mut w = BitWriter::new();
        // LSB first: 0b101 goes in bits 0-2, 0b11 in bits 3-4, 0b111 in bits 5-7
        w.write_bits(0b101, 3);
        w.write_bits(0b11, 2);
        w.write_bits(0b111, 3);
        w.flush();
        // Result: bits 0-2=101, bits 3-4=11, bits 5-7=111
        // = 0b111_11_101 = 0xFD = 253
        assert_eq!(w.buffer, &[0xFD]);
    }

    #[test]
    fn test_write_multiple_bytes() {
        let mut w = BitWriter::new();
        w.write_bits(0x12, 8);
        w.write_bits(0x34, 8);
        w.flush();
        assert_eq!(w.buffer, &[0x12, 0x34]);
    }

    #[test]
    fn test_write_14_bits() {
        let mut w = BitWriter::new();
        // VP8L header: width-1 (14 bits) = 1023 (0x3FF)
        w.write_bits(1023, 14);
        w.flush();
        // 0x3FF in 14 bits, LSB first
        // bits 0-7: 0xFF, bits 8-13: 0x03
        assert_eq!(w.buffer, &[0xFF, 0x03]);
    }
}
