# Migrate SIMD dispatch to incant!

72 dispatch sites use `if let Some(token) = summon() { arcane_fn(token, ...) }`.
This breaks on i686 because `#[arcane]` generates `cfg(target_arch = "x86_64")`.

Replace with `incant!` which handles cfg gating automatically and enables i686.

Files: 16 source files, 72 dispatch sites.
Blocked on: archmage 0.9.15 published to crates.io.
