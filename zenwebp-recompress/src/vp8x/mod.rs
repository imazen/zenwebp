//! Self-contained VP8 coefficient transcoder for the `CoeffEdit` strategy.
//!
//! This is the project's "least-disruptive" path: edit the VP8 DCT
//! coefficients in place (requantize coarser / drop high-frequency AC) and
//! re-emit the token stream, with **no IDCT/FDCT spatial round-trip** —
//! avoiding the generation loss that `Reencode` incurs.
//!
//! Built bottom-up and validated at each layer:
//! - [`bool`] — matched boolean decoder/encoder pair (round-trip tested).
//! - tables, header/MB/coefficient parse, and re-emit land on top.
//!
//! Everything is gated behind a correctness guard at the `CoeffEdit`
//! strategy: any frame this module cannot handle, or whose re-emit fails
//! validation, falls back to `Reencode`. Partial coverage is therefore
//! always safe.
//!
//! Some constants and helpers describe the full VP8 bitstream for clarity
//! and future edits even though the shipped edits don't touch every field;
//! `dead_code` is allowed so the structural completeness is documented in
//! one place rather than whittled down to only what today's two edits read.
#![allow(dead_code)]

pub mod bool;
pub mod edit;
pub mod emit;
pub mod parse;
pub mod tables;
