//! how-far integration: cancellation plus approximate progress.
//!
//! Public entry points accept a [`how_far::Pulse`] (`Stop + Report`). Inside
//! the crate cancellation stays a `&dyn Stop` and progress a `&dyn Report`, so
//! kernels that only check for cancellation are unchanged. Units are fixed per
//! entry point and documented there: encodes report percent (total 100),
//! still decodes report output rows, animation decoders report frames.

use core::sync::atomic::{AtomicU8, Ordering};

use enough::{Stop, StopReason};
use how_far::{Pulse, Total, WithStop};

use crate::encoder::EncodeProgress;

/// Combine an optional pulse with a stored stop token and run `f` with the
/// resulting stop and report. Without a pulse the stop is passed through and
/// nothing is reported.
pub(crate) fn with_pulse<R>(
    pulse: Option<&dyn Pulse>,
    stop: &dyn Stop,
    f: impl FnOnce(&dyn Stop, Option<&dyn Pulse>) -> R,
) -> R {
    match pulse {
        Some(pulse) => {
            let combined = WithStop::borrowed(pulse, stop);
            f(&combined, Some(&combined))
        }
        None => f(stop, None),
    }
}

/// Declare a leaf's unit count. A pulse that cannot revise its total keeps
/// counting against whatever its owner declared.
pub(crate) fn declare_total(pulse: &dyn Pulse, units: u64) {
    let _ = pulse.set_total(Total::Exact(units));
}

/// Feeds the encoder's percent callbacks into a pulse as units of a 100-unit
/// total, and still forwards them to a caller's [`EncodeProgress`].
pub(crate) struct PercentToPulse<'p> {
    pulse: &'p dyn Pulse,
    legacy: &'p dyn EncodeProgress,
    reported: AtomicU8,
}

impl<'p> PercentToPulse<'p> {
    pub(crate) fn new(pulse: &'p dyn Pulse, legacy: &'p dyn EncodeProgress) -> Self {
        declare_total(pulse, 100);
        Self {
            pulse,
            legacy,
            reported: AtomicU8::new(0),
        }
    }
}

impl EncodeProgress for PercentToPulse<'_> {
    fn on_progress(&self, percent: u8) -> Result<(), StopReason> {
        self.legacy.on_progress(percent)?;
        let percent = percent.min(100);
        let previous = self.reported.fetch_max(percent, Ordering::Relaxed);
        if percent > previous {
            self.pulse.advance(u64::from(percent - previous));
        }
        self.pulse.check()
    }
}

/// Run an encode with an optional pulse: the pulse joins the stop token, and
/// the encoder's percent callbacks are reported to it.
pub(crate) fn encode_with_pulse<R>(
    pulse: Option<&dyn Pulse>,
    stop: &dyn Stop,
    progress: &dyn EncodeProgress,
    f: impl FnOnce(&dyn Stop, &dyn EncodeProgress) -> R,
) -> R {
    with_pulse(pulse, stop, |stop, pulse| match pulse {
        Some(pulse) => {
            let adapter = PercentToPulse::new(pulse, progress);
            f(stop, &adapter)
        }
        None => f(stop, progress),
    })
}

/// [`with_pulse`] for an optional stored stop: without either, `f` sees no
/// stop at all (kernels skip their checks).
pub(crate) fn with_pulse_opt<R>(
    pulse: Option<&dyn Pulse>,
    stop: Option<&dyn Stop>,
    f: impl FnOnce(Option<&dyn Stop>, Option<&dyn Pulse>) -> R,
) -> R {
    match (pulse, stop) {
        (Some(pulse), Some(stop)) => {
            let combined = WithStop::borrowed(pulse, stop);
            f(Some(&combined), Some(&combined))
        }
        (Some(pulse), None) => f(Some(pulse as &dyn Stop), Some(pulse)),
        (None, stop) => f(stop, None),
    }
}
