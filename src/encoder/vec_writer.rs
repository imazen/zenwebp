//! Vec<u8> writer extension for no_std encoding.
//!
//! Provides write_u16_le, write_u24_le, etc. without requiring std::io::Write.

use alloc::vec::Vec;

/// Extension trait for writing to Vec<u8> without std::io.
#[allow(dead_code)] // Not all methods used yet
pub(crate) trait VecWriter {
    /// Append a slice to the buffer.
    fn write_all(&mut self, data: &[u8]);

    /// Write a u8.
    fn write_u8(&mut self, v: u8);

    /// Write a u16 in little-endian.
    fn write_u16_le(&mut self, v: u16);

    /// Write a u24 (3 bytes) in little-endian.
    fn write_u24_le(&mut self, v: u32);

    /// Write a u32 in little-endian.
    fn write_u32_le(&mut self, v: u32);
}

impl VecWriter for Vec<u8> {
    #[inline]
    fn write_all(&mut self, data: &[u8]) {
        self.extend_from_slice(data);
    }

    #[inline]
    fn write_u8(&mut self, v: u8) {
        self.push(v);
    }

    #[inline]
    fn write_u16_le(&mut self, v: u16) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn write_u24_le(&mut self, v: u32) {
        let bytes = v.to_le_bytes();
        self.extend_from_slice(&bytes[..3]);
    }

    #[inline]
    fn write_u32_le(&mut self, v: u32) {
        self.extend_from_slice(&v.to_le_bytes());
    }
}
