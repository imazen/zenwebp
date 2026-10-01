//! Golden encoder + decoder hashes over synthetic full-coverage patterns
//! (`tests/golden/cases.rs`). A lib unit test so it runs on every CI target,
//! including wasm32 under wasmtime and `--no-default-features`. The decode
//! hashes were verified against libwebp by `tests/golden_codec.rs` when they
//! were blessed; this test needs no libwebp.

use crate::golden_cases as g;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

const GOLDEN: &str = include_str!("../tests/golden/codec_golden.tsv");

#[test]
fn golden_codec_hashes() {
    let want = g::parse_golden(GOLDEN);
    let cases = g::cases();
    assert_eq!(
        want.len(),
        cases.len(),
        "golden file has {} cases, case list has {} — re-bless with \
         ZENWEBP_GOLDEN_BLESS=1 cargo test --release --test golden_codec",
        want.len(),
        cases.len()
    );
    let mut bad: Vec<String> = Vec::new();
    for (case, (name, enc, dec)) in cases.iter().zip(&want) {
        assert_eq!(&case.name, name, "case order changed");
        let (webp, e, ds) = g::zen_hashes(case);
        let d = g::fold(&ds);
        if e != *enc || d != *dec {
            let modes: Vec<String> = ds.iter().map(|(m, x)| format!("{m}={x:016x}")).collect();
            bad.push(format!(
                "{name}: enc {}{e:016x} ({} bytes) dec {}{d:016x} [{}]",
                if e == *enc { "ok " } else { "CHANGED " },
                webp.len(),
                if d == *dec { "ok " } else { "CHANGED " },
                modes.join(" ")
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} golden cases changed:\n{}",
        bad.len(),
        cases.len(),
        bad.iter().take(25).cloned().collect::<Vec<_>>().join("\n")
    );
}
