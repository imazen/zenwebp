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

    /// Sets the current position.
    #[inline]
    pub fn set_position(&mut self, pos: u64) {
        self.pos = pos as usize;
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
        let pos = pos as usize;
        if pos > self.data.len() {
            return Err(DecodeError::BitStreamError);
        }
        self.pos = pos;
        Ok(self.pos as u64)
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
        if self.pos + n > self.data.len() {
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
        if self.pos + n > self.data.len() {
            return Err(DecodeError::BitStreamError);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Get a slice of n bytes from the current position without advancing.
    #[inline]
    pub fn peek_slice(&self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.pos + n > self.data.len() {
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
        let new_pos = match pos {
            std::io::SeekFrom::Start(n) => Some(n as usize),
            std::io::SeekFrom::End(n) => {
                if n >= 0 {
                    self.data.len().checked_add(n as usize)
                } else {
                    self.data.len().checked_sub((-n) as usize)
                }
            }
            std::io::SeekFrom::Current(n) => {
                if n >= 0 {
                    self.pos.checked_add(n as usize)
                } else {
                    self.pos.checked_sub((-n) as usize)
                }
            }
        };

        match new_pos {
            Some(pos) if pos <= self.data.len() => {
                self.pos = pos;
                Ok(self.pos as u64)
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek out of bounds",
            )),
        }
    }
}
