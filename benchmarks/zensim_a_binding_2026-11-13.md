# Revision-7 A target binding

Production WebP targeting pins imazen/zensim
`220dfc6a693ef8b1d6fd11c28c6a5e2cad64bd48` and explicitly selects A.
The independent reconstruction-score control failed before migration:
61.284134 reported, versus revision-7 A 44.012302 on the same output.
It passes after migration. Production measurement uses score-only APIs;
separately opted-in research retains its own model and spatial controls.

The legacy regression dependency stays at registry 0.2.7. Its two color-block
quality floors remain >=75 on their original oracle, now independently rescored
from the actual output. The numeric gates are unchanged. Existing target tests
and new A identity/reachability/strict-refusal controls pass.

## Bounded color-boundary witness

`tests/target_zensim_a.rs` constructs the exact 256x256 four-color fixture.
Method 4, q0..100 step5, default conversion versus explicit SharpYUV (42 cells):

| Configuration | Measured A | Legacy 0.2.7 | Bytes |
|---|---:|---:|---:|
| Default conversion, q95 and q100 (highest A in this grid) | 73.496860 | 80.694692 | 276 / 254 |
| SharpYUV, q80 | 82.150421 | 86.602332 | 484 |
| SharpYUV, q100 | 85.361723 | 87.034874 | 544 |

The strict A=82 request refuses on the ordinary conversion and reaches its
configured band with explicitly enabled SharpYUV within six passes. Targeting
does not override the caller's conversion choice. A grid maximum is not proof
about every fractional quality or arbitrary image. This witness does not qualify
a broad calibration or establish a new default conversion policy.

The historical bucket tables are initial guesses, not revision-7 calibration.
No coefficients were fitted and no scientific thresholds were weakened. The
nested recompressor's separate historical profile/calibration is unchanged.

Reproduce: `just target-zensim-check` and `just target-zensim-boundary`.
The bounded cells are below the 100-cell/one-second-per-cell artifact exemption.
