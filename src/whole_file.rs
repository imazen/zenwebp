//! Admission checks around the shared whole-file adapter, before it asks a
//! driver to retain the complete input. The driver owns the input allocation.

use whereat::{At, at};
use zencodec::decode::{
    DecodeJob, DecodeRowSink, Event, IncrementalDecode, Input, Progress, Request, StepError,
};
use zencodec::{CategorizedError, CodecError, ImageInfo, LimitExceeded, ResourceLimits};
use zenpixels::PixelDescriptor;

pub(crate) struct LimitedWholeFile<J> {
    inner: zencodec::helpers::WholeFileIncremental<J>,
    limits: ResourceLimits,
    codec: &'static str,
    probe: J,
    need: [Request; 1],
}

impl<J> LimitedWholeFile<J>
where
    J: for<'a> DecodeJob<'a, Error = At<CodecError>> + Send + 'static,
{
    pub(crate) fn new(
        job: J,
        probe: J,
        preferred: &[PixelDescriptor],
        limits: ResourceLimits,
        codec: &'static str,
    ) -> Self {
        Self {
            inner: zencodec::helpers::WholeFileIncremental::new(job, preferred),
            probe,
            limits,
            codec,
            need: [Request::at_least(0, 1)],
        }
    }

    fn limit_error(&self, error: LimitExceeded) -> StepError<At<CodecError>> {
        StepError::Codec(at!(CodecError::from_parts(
            Some(self.codec),
            error.category(),
            alloc::boxed::Box::new(error)
        )))
    }

    fn request_cap(&self) -> usize {
        let cap = match (self.limits.max_input_bytes, self.limits.max_memory_bytes) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => u64::MAX,
        };
        // One sentinel byte lets an unknown-length source distinguish EOF at
        // the cap from an input exceeding it. No unbounded prefix is requested.
        usize::try_from(cap).unwrap_or(usize::MAX).saturating_add(1)
    }
}

impl<J> IncrementalDecode for LimitedWholeFile<J>
where
    J: for<'a> DecodeJob<'a, Error = At<CodecError>> + Send + 'static,
{
    type Error = At<CodecError>;

    fn step(
        &mut self,
        input: Input<'_>,
        sink: &mut dyn DecodeRowSink,
        stop: &dyn enough::Stop,
    ) -> Result<Progress, StepError<Self::Error>> {
        stop.check()?;
        // A length hint applies to the source, independently of the offered
        // window. Reject it before the shared adapter requests offset zero.
        let bytes = input.total_len.unwrap_or(0).max(input.bytes.len() as u64);
        self.limits
            .check_input_size(bytes)
            .and_then(|()| self.limits.check_memory(bytes))
            .map_err(|e| self.limit_error(e))?;
        if input.offset == 0 && (self.inner.info().is_none() || input.reaches_end()) {
            match self.probe.probe(input.bytes) {
                Ok(info) => self
                    .limits
                    .check_image_info(&info)
                    .map_err(|e| self.limit_error(e))?,
                // A resource refusal is final, not an incomplete header. The
                // shared adapter otherwise treats every probe error as Need.
                Err(e) if matches!(e.error().category(), zencodec::ErrorCategory::Resource(_)) => {
                    return Err(StepError::Codec(e));
                }
                Err(_) => {}
            }
        }
        let progress = self.inner.step(input, sink, stop)?;
        if let Some(info) = self.inner.info() {
            self.limits
                .check_image_info(info)
                .map_err(|e| self.limit_error(e))?;
        }
        // WholeFileIncremental only asks for an offset-zero prefix.
        if progress.event == Event::Need
            && let Some(Request::AtLeast { offset, min_len }) = self.inner.requested().first()
        {
            self.need = [Request::at_least(
                *offset,
                (*min_len).min(self.request_cap()),
            )];
        }
        Ok(progress)
    }

    fn requested(&self) -> &[Request] {
        &self.need
    }

    fn info(&self) -> Option<&ImageInfo> {
        self.inner.info()
    }

    fn output_descriptor(&self) -> Option<PixelDescriptor> {
        self.inner.output_descriptor()
    }

    fn access_pattern(&self) -> zencodec::source::AccessPattern {
        // No bytes are consumed before the full-file decode; the driver keeps
        // the unconsumed prefix, and never has to replay consumed input.
        zencodec::source::AccessPattern::Sequential
    }
}
