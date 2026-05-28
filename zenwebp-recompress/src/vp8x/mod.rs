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
//! NOTE: this module is built bottom-up across milestones; lower layers
//! (the bool coder) are validated by tests but not yet consumed by the
//! not-yet-landed upper layers, so `dead_code` is allowed here until the
//! `CoeffEdit` strategy wires the transcoder in (then this is removed).
#![allow(dead_code)]

pub mod bool;
pub mod parse;
pub mod tables;
