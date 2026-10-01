//! Target-independent half of the decoder parity sweep: decode every `.webp`
//! under `--dir` through zenwebp in every mode `decode_parity_sweep` checks,
//! digest each output, and compare against the libwebp digests that
//! `decode_parity_sweep --save-gen DIR --hash-out lib.hashes` wrote natively.
//!
//! Runs where libwebp can't: wasm32-wasip1 under wasmtime (scalar and
//! `+simd128` builds), or any host whose output should be checked against an
//! x86 libwebp run.
//!
//!   cargo build --release --target wasm32-wasip1 --example decode_parity_dump
//!   wasmtime run --dir DIR::/data target/wasm32-wasip1/release/examples/decode_parity_dump.wasm \
//!     -- --dir /data --expect /data/lib.hashes [--shard 0/8]
//!
//! Exit status 0 only when every (file, mode) digest matches.

#[path = "decode_parity_common.rs"]
mod common;

use common::{digest, kind, modes, zen_decode};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "webp") {
            out.push(p);
        }
    }
}

fn main() {
    let mut dir = PathBuf::new();
    let mut expect: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let (mut shard, mut shards) = (0usize, 1usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = PathBuf::from(args.next().unwrap()),
            "--expect" => expect = Some(PathBuf::from(args.next().unwrap())),
            "--out" => out = Some(PathBuf::from(args.next().unwrap())),
            // --raw MODE IN OUT: write one mode's zenwebp output bytes and exit.
            "--raw" => {
                let mode = args.next().unwrap();
                let input = std::fs::read(args.next().unwrap()).unwrap();
                let (b, w, h) = zen_decode(&mode, &input).unwrap();
                std::fs::write(args.next().unwrap(), b).unwrap();
                println!("{w}x{h}");
                return;
            }
            "--shard" => {
                let v = args.next().unwrap();
                let (i, n) = v.split_once('/').expect("--shard i/n");
                (shard, shards) = (i.parse().unwrap(), n.parse().unwrap());
            }
            o => panic!("unknown arg {o}"),
        }
    }
    let expected: HashMap<(String, String), String> = expect
        .map(|p| {
            std::fs::read_to_string(p)
                .unwrap()
                .lines()
                .filter_map(|l| {
                    let mut it = l.split('\t');
                    Some(((it.next()?.into(), it.next()?.into()), it.next()?.into()))
                })
                .collect()
        })
        .unwrap_or_default();

    let mut files = Vec::new();
    walk(&dir, &mut files);
    files.sort();
    let files: Vec<_> = files
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % shards == shard)
        .map(|(_, p)| p)
        .collect();
    eprintln!(
        "{} files (shard {shard}/{shards}), {} expected digests",
        files.len(),
        expected.len()
    );

    let mut lines = Vec::new();
    let mut tally: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut bad = Vec::new();
    for (i, p) in files.iter().enumerate() {
        let data = std::fs::read(p).unwrap();
        let key = p
            .strip_prefix(&dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        for &(mode, _) in modes(kind(&data)) {
            let d = digest(&zen_decode(mode, &data));
            let status = match expected.get(&(key.clone(), mode.to_string())) {
                None => "no_expect",
                Some(x) if *x == d => "match",
                Some(x) if x == "ERR" && d == "ERR" => "match",
                Some(x) => {
                    bad.push(format!("{key} [{mode}] zen={d} lib={x}"));
                    "diff"
                }
            };
            *tally.entry((mode, status)).or_default() += 1;
            lines.push(format!("{key}\t{mode}\t{d}"));
        }
        if (i + 1) % 1000 == 0 {
            eprintln!("  {}/{}", i + 1, files.len());
        }
    }
    if let Some(o) = out {
        std::fs::write(o, lines.join("\n") + "\n").unwrap();
    }
    for ((mode, status), n) in &tally {
        println!("{mode:24} {status:10} {n}");
    }
    println!("NON-MATCH: {}", bad.len());
    for b in bad.iter().take(40) {
        println!("  {b}");
    }
    if !bad.is_empty() || tally.keys().any(|(_, s)| *s == "no_expect") && !expected.is_empty() {
        std::process::exit(1);
    }
}
