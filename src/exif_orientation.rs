//! Minimal EXIF orientation parser.
//!
//! Reads the EXIF Orientation tag (0x0112) from raw TIFF/EXIF bytes.
//! Handles both raw TIFF (starts with `II`/`MM`) and JPEG-style EXIF
//! (starts with `Exif\0\0` prefix).
//!
//! This is intentionally minimal — just orientation, nothing else.
//! To be factored into zencodec: <https://github.com/imazen/zencodec/issues/5>

/// EXIF Orientation tag number (TIFF tag 274).
const TAG_ORIENTATION: u16 = 0x0112;

/// TIFF type SHORT (unsigned 16-bit).
const TIFF_TYPE_SHORT: u16 = 3;

/// Minimum TIFF header size: byte order (2) + magic (2) + IFD offset (4).
const TIFF_HEADER_MIN: usize = 8;

/// Parse EXIF orientation from raw bytes.
///
/// Accepts either:
/// - Raw TIFF bytes (starting with `II` or `MM` byte order mark)
/// - JPEG-style EXIF with `Exif\0\0` prefix
///
/// Returns `Some(1..=8)` if the orientation tag is found, `None` otherwise.
pub(crate) fn parse_orientation(data: &[u8]) -> Option<u8> {
    // Strip Exif\0\0 prefix if present (JPEG APP1 style)
    let tiff = if data.len() >= 6 && &data[..6] == b"Exif\0\0" {
        &data[6..]
    } else {
        data
    };

    if tiff.len() < TIFF_HEADER_MIN {
        return None;
    }

    let big_endian = match &tiff[0..2] {
        b"MM" => true,
        b"II" => false,
        _ => return None,
    };

    // Verify TIFF magic (42)
    if read_u16(tiff, 2, big_endian) != 42 {
        return None;
    }

    // Read IFD0 offset
    let ifd_offset = read_u32(tiff, 4, big_endian) as usize;
    if ifd_offset.checked_add(2)? > tiff.len() {
        return None;
    }

    let entry_count = read_u16(tiff, ifd_offset, big_endian) as usize;
    let entries_start = ifd_offset + 2;

    for i in 0..entry_count {
        let offset = entries_start + i * 12;
        if offset + 12 > tiff.len() {
            break;
        }

        let tag = read_u16(tiff, offset, big_endian);
        if tag == TAG_ORIENTATION {
            if read_u16(tiff, offset + 2, big_endian) != TIFF_TYPE_SHORT {
                return None;
            }
            let value = read_u16(tiff, offset + 8, big_endian);
            return if (1..=8).contains(&value) {
                Some(value as u8)
            } else {
                None
            };
        }
        // IFD entries are sorted by tag — early exit
        if tag > TAG_ORIENTATION {
            break;
        }
    }

    None
}

fn read_u16(data: &[u8], offset: usize, big_endian: bool) -> u16 {
    let b = &data[offset..offset + 2];
    if big_endian {
        u16::from_be_bytes([b[0], b[1]])
    } else {
        u16::from_le_bytes([b[0], b[1]])
    }
}

fn read_u32(data: &[u8], offset: usize, big_endian: bool) -> u32 {
    let b = &data[offset..offset + 4];
    if big_endian {
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    } else {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tiff_with_orientation(orientation: u16, big_endian: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        let write_u16 = |buf: &mut Vec<u8>, v: u16| {
            if big_endian {
                buf.extend_from_slice(&v.to_be_bytes());
            } else {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        };
        let write_u32 = |buf: &mut Vec<u8>, v: u32| {
            if big_endian {
                buf.extend_from_slice(&v.to_be_bytes());
            } else {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        };

        // Byte order
        if big_endian {
            buf.extend_from_slice(b"MM");
        } else {
            buf.extend_from_slice(b"II");
        }
        // Magic
        write_u16(&mut buf, 42);
        // IFD0 offset (immediately after header)
        write_u32(&mut buf, 8);
        // Entry count: 1
        write_u16(&mut buf, 1);
        // IFD entry: tag=0x0112, type=SHORT(3), count=1, value=orientation
        write_u16(&mut buf, TAG_ORIENTATION);
        write_u16(&mut buf, TIFF_TYPE_SHORT);
        write_u32(&mut buf, 1); // count
        write_u16(&mut buf, orientation);
        write_u16(&mut buf, 0); // padding

        buf
    }

    #[test]
    fn parse_little_endian() {
        let tiff = make_tiff_with_orientation(6, false);
        assert_eq!(parse_orientation(&tiff), Some(6));
    }

    #[test]
    fn parse_big_endian() {
        let tiff = make_tiff_with_orientation(3, true);
        assert_eq!(parse_orientation(&tiff), Some(3));
    }

    #[test]
    fn parse_with_exif_prefix() {
        let tiff = make_tiff_with_orientation(8, false);
        let mut with_prefix = b"Exif\0\0".to_vec();
        with_prefix.extend_from_slice(&tiff);
        assert_eq!(parse_orientation(&with_prefix), Some(8));
    }

    #[test]
    fn normal_orientation() {
        let tiff = make_tiff_with_orientation(1, false);
        assert_eq!(parse_orientation(&tiff), Some(1));
    }

    #[test]
    fn invalid_orientation_zero() {
        let tiff = make_tiff_with_orientation(0, false);
        assert_eq!(parse_orientation(&tiff), None);
    }

    #[test]
    fn invalid_orientation_nine() {
        let tiff = make_tiff_with_orientation(9, false);
        assert_eq!(parse_orientation(&tiff), None);
    }

    #[test]
    fn empty_data() {
        assert_eq!(parse_orientation(&[]), None);
    }

    #[test]
    fn truncated_header() {
        assert_eq!(parse_orientation(b"II*\0"), None);
    }

    #[test]
    fn no_orientation_tag() {
        // TIFF with one entry that isn't orientation (tag 0x0100 = ImageWidth)
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // 1 entry
        buf.extend_from_slice(&0x0100u16.to_le_bytes()); // ImageWidth tag
        buf.extend_from_slice(&3u16.to_le_bytes()); // SHORT
        buf.extend_from_slice(&1u32.to_le_bytes()); // count
        buf.extend_from_slice(&640u16.to_le_bytes()); // value
        buf.extend_from_slice(&0u16.to_le_bytes()); // padding
        // Tag 0x0100 > 0x0112? No, 0x100 < 0x112, so it won't early-exit
        // But there's only 1 entry and it's not orientation
        assert_eq!(parse_orientation(&buf), None);
    }
}
