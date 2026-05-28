//! Strategy implementations. Each module is a single function that takes
//! source bytes + analysis + options and returns recompressed bytes.

pub mod coeff_edit;
pub mod deblock;
pub mod deblock_reencode;
pub mod lossless_reencode;
pub mod lossless_remux;
pub mod reencode;
