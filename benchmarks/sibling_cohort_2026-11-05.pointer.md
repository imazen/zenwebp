# WebP sibling-cohort metadata boundary (2026-11-05)

Inspected base main7f7ab908712aa62d4484d66a111c8cb46f97ffc9,the incoming VP8L
allocator fix retained from the other lane. The new graph uses core1816b4d0,
pixelsfa836abf, analyzer38dc80f7, PNG02a7f0df and dev resamplere3975fb9.
Root/fuzz/recompress resolved metadata each has one core/pixel identity.
The recompress scorer remains zensim bab88fde,Profile A; its calibration table
and public target scale are unchanged. No encoder qualification was rerun.

- Clean-base fmt/library clippy:rc0,10s,peak RSS0.45GiB.
- Updated default-library compile/lint:rc0,6s,peak RSS0.45GiB.
- Default runtime:831 passes across81test binaries,zero failures,11existing
  ignores. Doctests:33pass,threeexisting ignores. This is DEFAULT coverage:the
  first selector used just --quiet,which suppressed its feature-list stdout.
  The selector is repaired and verifies/prints all18stable features; their full
  qualification has not run.
- Strict default all-target lint:11distinct deprecated setter diagnostics across
  six compiled targets. There are13source call sites across six files(all
  configurations,including the mutually exclusive CMS tests).
- `just sibling-metadata-probe`:empty,ICC and CICP records all refuse with
  `unsupported operation: metadata_retention`(3/3). This reports the inherited
  trait default,not an implemented retention guarantee. Scope:rc0,1s,peak RSS
  0.17GiB,minimum available8,421MiB,peak load1.33.
- Full default test/doc + failed lint scope:rc101,249s,peak RSS0.93GiB,minimum
  available6,812MiB,peak load5.26. Full stable-feature,fuzz and recompress
  execution remain pending. Neither test expectations nor ignored tests changed.

The checked trait's existing contract requires policy-filtered retention on all
encode paths,typed equivalence/opaque byte preservation,unchanged job state on
refusal,and protection from later mutators. The current WebP job implements the
legacy best-effort setter only. Changing the tests to the checked builder would
reject even an empty record. The coordinator must choose the retention scope
before continuing; no locally incomplete cohort is pushed to main.

Contract source:[EncodeJob checked metadata](https://github.com/imazen/zencodec/blob/1816b4d082234c04c3805411b437b8396040f522/src/traits/encoding.rs#L426).
Full logs,source archive,resolved locks and dependency metadata:
`/mnt/v/output/zensim-siblings-20261105/zenwebp/`(checked on dev).
Mirror:`/mnt/tower/output/zensim-siblings-20261105/zenwebp/`.
