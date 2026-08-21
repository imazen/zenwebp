//! Test-only serialization between lib tests that DISABLE archmage tokens
//! process-wide and lib tests whose call paths summon-and-expect those tokens.
//!
//! `archmage::testing::for_each_token_permutation` (used by the tier tests in
//! `decoder::vp8v2::yuv_exact`) flips the process-wide token-disable statics.
//! libtest runs `#[test]`s on parallel threads in ONE process, so during a
//! disabled window an unrelated test whose production path does
//! `X64V3Token::summon().expect(..)` panics — observed 2026-08-21 as
//! `test_yuv_to_rgb_matches_scalar` failing "SSE4.1 required for SIMD YUV" on
//! an AVX2 machine, roughly a coin flip under load. (The `testable_dispatch`
//! feature reaches the lib tests through dev-dependency feature unification,
//! which is what makes the tokens disableable here at all.)
//!
//! Every lib test that toggles tokens AND every lib test that summon-expects
//! them takes this lock first. It is a spin lock because the
//! no-default-features test build has no `std::sync::Mutex`; hold times are
//! milliseconds and contenders are a handful of tests.
//!
//! This is a class-F instance (global state inside a parallel suite) — see
//! docs/BUG_RETROSPECTIVE_2026-08.md.

use core::sync::atomic::{AtomicBool, Ordering};

static LOCKED: AtomicBool = AtomicBool::new(false);

pub(crate) struct TokenToggleGuard;

/// Acquire the process-wide token-toggle lock for the duration of a test.
pub(crate) fn lock_tokens_for_test() -> TokenToggleGuard {
    while LOCKED
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
    TokenToggleGuard
}

impl Drop for TokenToggleGuard {
    fn drop(&mut self) {
        LOCKED.store(false, Ordering::Release);
    }
}
