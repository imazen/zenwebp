//! A no_std compatible slice reader for reading bytes with position tracking.
//!
//! This module provides [`SliceReader`] which wraps a byte slice and provides
//! methods for reading primitive types, similar to `std::io::Cursor` but without
//! requiring the standard library.

use alloc::vec::Vec;
use byteorder_lite::{ByteOrder, LittleEndian};
use core::fmt;

use crate::DecodeError;

/// A reader that wraps a byte slice and tracks the current position.
///
/// This is a no_std alternative to `std::io::Cursor<&[u8]>` that provides
/// the subset of functionality needed for WebP decoding.
#[derive(Clone)]
pub struct SliceReader<'a> {
    data: &'a [u8],
    pos: usize,
}

#[allow(dead_code)]
impl<'a> SliceReader<'a> {
    /// Create a new SliceReader wrapping the given byte slice.
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Returns the current position in the slice.
    #[inline]
    pub fn position(&self) -> u64 {
        self.pos as u64
    }

    /// Sets the current position, clamped to the end of the slice.
    #[inline]
    pub fn set_position(&mut self, pos: u64) {
        // Clamp in `u64`, not after narrowing: `pos as usize` truncates on
        // 32-bit targets, so an out-of-range value can land *behind* the
        // current position instead of at the end. See `seek_from_start`.
        self.pos = pos.min(self.data.len() as u64) as usize;
    }

    /// Returns the underlying byte slice.
    #[inline]
    pub fn get_ref(&self) -> &'a [u8] {
        self.data
    }

    /// Returns the total length of the underlying slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the underlying slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the number of bytes remaining from the current position.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Returns a slice of the remaining bytes.
    #[inline]
    pub fn remaining_slice(&self) -> &'a [u8] {
        &self.data[self.pos.min(self.data.len())..]
    }

    /// Seek to a position from the start.
    #[inline]
    pub fn seek_from_start(&mut self, pos: u64) -> Result<u64, DecodeError> {
        // Range-check BEFORE narrowing. `pos` is derived from file-declared
        // chunk sizes, so on 32-bit targets (wasm32) `pos as usize` truncates
        // an out-of-range seek into a valid in-range one — which silently
        // rewinds the caller instead of erroring, and container walks that
        // seek chunk-by-chunk then spin forever at 100% CPU.
        if pos > self.data.len() as u64 {
            return Err(DecodeError::BitStreamError);
        }
        self.pos = pos as usize;
        Ok(pos)
    }

    /// Seek relative to current position.
    #[inline]
    pub fn seek_relative(&mut self, offset: i64) -> Result<(), DecodeError> {
        let new_pos = if offset >= 0 {
            self.pos.checked_add(offset as usize)
        } else {
            self.pos.checked_sub((-offset) as usize)
        };

        match new_pos {
            Some(pos) if pos <= self.data.len() => {
                self.pos = pos;
                Ok(())
            }
            _ => Err(DecodeError::BitStreamError),
        }
    }

    /// Read exactly `n` bytes into the buffer.
    #[inline]
    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), DecodeError> {
        let n = buf.len();
        // `n` is a real buffer length here (can't overflow), but the
        // remaining-bytes form keeps every bounds check in this file uniform.
        if n > self.data.len() - self.pos {
            return Err(DecodeError::BitStreamError);
        }
        buf.copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(())
    }

    /// Read up to `buf.len()` bytes, returning the number of bytes read.
    #[inline]
    pub fn read(&mut self, buf: &mut [u8]) -> usize {
        let available = self.data.len().saturating_sub(self.pos);
        let to_read = buf.len().min(available);
        buf[..to_read].copy_from_slice(&self.data[self.pos..self.pos + to_read]);
        self.pos += to_read;
        to_read
    }

    /// Read a single byte.
    #[inline]
    pub fn read_u8(&mut self) -> Result<u8, DecodeError> {
        if self.pos >= self.data.len() {
            return Err(DecodeError::BitStreamError);
        }
        let byte = self.data[self.pos];
        self.pos += 1;
        Ok(byte)
    }

    /// Read a u16 in little-endian byte order.
    #[inline]
    pub fn read_u16_le(&mut self) -> Result<u16, DecodeError> {
        if self.pos + 2 > self.data.len() {
            return Err(DecodeError::BitStreamError);
        }
        let val = LittleEndian::read_u16(&self.data[self.pos..]);
        self.pos += 2;
        Ok(val)
    }

    /// Read a u24 in little-endian byte order (as u32).
    #[inline]
    pub fn read_u24_le(&mut self) -> Result<u32, DecodeError> {
        if self.pos + 3 > self.data.len() {
            return Err(DecodeError::BitStreamError);
        }
        let val = LittleEndian::read_u24(&self.data[self.pos..]);
        self.pos += 3;
        Ok(val)
    }

    /// Read a u32 in little-endian byte order.
    #[inline]
    pub fn read_u32_le(&mut self) -> Result<u32, DecodeError> {
        if self.pos + 4 > self.data.len() {
            return Err(DecodeError::BitStreamError);
        }
        let val = LittleEndian::read_u32(&self.data[self.pos..]);
        self.pos += 4;
        Ok(val)
    }

    /// Fill the internal buffer (for BufRead compatibility).
    /// Returns a slice of available data without consuming it.
    #[inline]
    pub fn fill_buf(&self) -> &'a [u8] {
        &self.data[self.pos.min(self.data.len())..]
    }

    /// Consume `amt` bytes from the buffer.
    #[inline]
    pub fn consume(&mut self, amt: usize) {
        self.pos = (self.pos + amt).min(self.data.len());
    }

    /// Returns the current stream position (alias for position()).
    #[inline]
    pub fn stream_position(&self) -> u64 {
        self.pos as u64
    }

    /// Read all remaining bytes into the provided Vec.
    #[inline]
    pub fn read_to_end(&mut self, buf: &mut Vec<u8>) -> Result<usize, DecodeError> {
        let remaining = self.remaining_slice();
        let len = remaining.len();
        buf.extend_from_slice(remaining);
        self.pos = self.data.len();
        Ok(len)
    }

    /// Take a slice of n bytes from the current position and advance position.
    /// Returns a slice reference without copying data.
    #[inline]
    pub fn take_slice(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        // Compare against remaining bytes rather than `self.pos + n`: callers
        // pass file-declared chunk sizes (`chunk_size as usize`), so on a
        // 32-bit target `self.pos + n` wraps for a near-u32::MAX `n` and the
        // guard would pass, then the slice index panics. `self.pos <=
        // self.data.len()` is an invariant, so the subtraction can't underflow.
        if n > self.data.len() - self.pos {
            return Err(DecodeError::BitStreamError);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Get a slice of n bytes from the current position without advancing.
    #[inline]
    pub fn peek_slice(&self, n: usize) -> Result<&'a [u8], DecodeError> {
        // See `take_slice`: remaining-bytes comparison avoids a 32-bit wrap on
        // an attacker-controlled `n`.
        if n > self.data.len() - self.pos {
            return Err(DecodeError::BitStreamError);
        }
        Ok(&self.data[self.pos..self.pos + n])
    }
}

impl fmt::Debug for SliceReader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SliceReader")
            .field("len", &self.data.len())
            .field("pos", &self.pos)
            .finish()
    }
}

#[cfg(feature = "std")]
impl<'a> std::io::Read for SliceReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(SliceReader::read(self, buf))
    }
}

#[cfg(feature = "std")]
impl<'a> std::io::BufRead for SliceReader<'a> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        Ok(SliceReader::fill_buf(self))
    }

    fn consume(&mut self, amt: usize) {
        SliceReader::consume(self, amt)
    }
}

#[cfg(feature = "std")]
impl<'a> std::io::Seek for SliceReader<'a> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        // All arithmetic and range checks happen in `u64`; narrowing to
        // `usize` only after `new_pos <= len` is proven. `n as usize` on a
        // 32-bit target truncates i64/u64 offsets, which can turn an invalid
        // seek into a "valid" one (or a no-op) — the same class as the
        // `seek_from_start` truncation that spun container parsing forever.
        let len = self.data.len() as u64;
        let new_pos: Option<u64> = match pos {
            std::io::SeekFrom::Start(n) => Some(n),
            std::io::SeekFrom::End(n) => {
                if n >= 0 {
                    len.checked_add(n as u64)
                } else {
                    len.checked_sub(n.unsigned_abs())
                }
            }
            std::io::SeekFrom::Current(n) => {
                if n >= 0 {
                    (self.pos as u64).checked_add(n as u64)
                } else {
                    (self.pos as u64).checked_sub(n.unsigned_abs())
                }
            }
        };

        match new_pos {
            Some(pos) if pos <= len => {
                self.pos = pos as usize;
                Ok(pos)
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek out of bounds",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SliceReader;

    /// Regression: a seek target past the end must ERROR, not truncate into a
    /// valid position. On 32-bit targets (wasm32) `pos as usize` before the
    /// range check turns `data.len() + 2^32` into `data.len()`, which rewinds
    /// container walks instead of stopping them — an infinite loop at 100% CPU.
    #[test]
    fn seek_past_end_errors_instead_of_truncating() {
        let data = [0u8; 70];
        let mut r = SliceReader::new(&data);
        r.seek_from_start(30).unwrap();

        // 30 + 2^32: truncates to 30 on a 32-bit `usize`.
        assert!(r.seek_from_start(30 + (1u64 << 32)).is_err());
        assert_eq!(r.position(), 30, "failed seek must not move the cursor");

        // Exactly at the end is still a legal seek; one past it is not.
        assert!(r.seek_from_start(70).is_ok());
        assert!(r.seek_from_start(71).is_err());
    }

    /// `set_position` clamps to the end; it must never land *behind* the
    /// cursor because the value truncated into range on a 32-bit `usize`.
    #[test]
    fn set_position_clamps_instead_of_truncating() {
        let data = [0u8; 70];
        let mut r = SliceReader::new(&data);
        r.set_position(30 + (1u64 << 32));
        assert_eq!(r.position(), 70);
    }

    /// Regression: `take_slice`/`peek_slice` must reject an `n` that would
    /// wrap `self.pos + n` on a 32-bit target, rather than passing the bounds
    /// check and panicking on the slice index. Callers pass file-declared
    /// `chunk_size as usize`, so this is reachable from untrusted input.
    /// Trivially passes on 64-bit; gates for real on i686 and wasm32.
    #[test]
    fn take_and_peek_slice_reject_wrapping_length() {
        let data = [0u8; 70];
        let mut r = SliceReader::new(&data);
        r.seek_from_start(60).unwrap();

        // usize::MAX - 8: on 32-bit `60 + n` wraps below 70; must still error.
        let huge = usize::MAX - 8;
        assert!(r.peek_slice(huge).is_err());
        assert!(r.take_slice(huge).is_err());
        assert_eq!(r.position(), 60, "failed take_slice must not advance");

        // Exactly-remaining is legal; one past is not.
        assert!(r.peek_slice(10).is_ok());
        assert!(r.peek_slice(11).is_err());
        let s = r.take_slice(10).unwrap();
        assert_eq!(s.len(), 10);
        assert_eq!(r.position(), 70);
        assert!(r.take_slice(1).is_err());
    }

    /// Same guarantee through the `std::io::Seek` impl.
    #[cfg(feature = "std")]
    #[test]
    fn io_seek_start_past_end_errors_instead_of_truncating() {
        use std::io::{Seek, SeekFrom};
        let data = [0u8; 70];
        let mut r = SliceReader::new(&data);
        r.seek(SeekFrom::Start(30)).unwrap();
        assert!(r.seek(SeekFrom::Start(30 + (1u64 << 32))).is_err());
        assert_eq!(r.position(), 30, "failed seek must not move the cursor");
    }

    /// `SeekFrom::End`/`Current` with a huge positive offset must error, not
    /// truncate. On a 32-bit `usize`, `n as usize` maps `2^32` to `0`, so
    /// `Current(2^32)` "succeeded" at the SAME position (a no-progress loop
    /// for any caller advancing by file-declared sizes) and `End(2^32)`
    /// "succeeded" at the end instead of erroring.
    #[cfg(feature = "std")]
    #[test]
    fn io_seek_end_current_do_not_truncate() {
        use std::io::{Seek, SeekFrom};
        let data = [0u8; 70];
        let mut r = SliceReader::new(&data);
        r.seek(SeekFrom::Start(30)).unwrap();

        assert!(r.seek(SeekFrom::Current(1i64 << 32)).is_err());
        assert_eq!(r.position(), 30, "failed seek must not move the cursor");
        assert!(r.seek(SeekFrom::End(1i64 << 32)).is_err());
        assert_eq!(r.position(), 30, "failed seek must not move the cursor");

        // i64::MIN is the `unsigned_abs` edge (`-n` would overflow).
        assert!(r.seek(SeekFrom::Current(i64::MIN)).is_err());
        assert!(r.seek(SeekFrom::End(i64::MIN)).is_err());

        // Legal seeks through every arm still work.
        assert_eq!(r.seek(SeekFrom::Current(10)).unwrap(), 40);
        assert_eq!(r.seek(SeekFrom::Current(-10)).unwrap(), 30);
        assert_eq!(r.seek(SeekFrom::End(0)).unwrap(), 70);
        assert_eq!(r.seek(SeekFrom::End(-70)).unwrap(), 0);
    }
}
