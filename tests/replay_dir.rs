//! Replays every file under `$REPLAY_DIR` (recursively) through the public
//! decode/demux entry points, catching panics per entry point and reporting
//! them grouped by panic location. Used to triage fuzz-farm crash sets against
//! this release line. Run explicitly:
//! `REPLAY_DIR=<dir> cargo test --release --test replay_dir -- --nocapture`
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

static LAST_PANIC: Mutex<String> = Mutex::new(String::new());
const MAX_PIXEL_BYTES: usize = 64 * 1024 * 1024;

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("crash-"))
        {
            out.push(p);
        }
    }
}

fn fuzz_config() -> zenwebp::DecodeConfig {
    let limits = zenwebp::Limits::default()
        .max_dimensions(4096, 4096)
        .max_total_pixels(4_000_000)
        .max_memory(64 * 1024 * 1024);
    zenwebp::DecodeConfig::default().limits(limits)
}

fn still(input: &[u8]) {
    if let Ok(mut d) = zenwebp::WebPDecoder::new(input) {
        let (w, h) = d.dimensions();
        let bpp = if d.has_alpha() { 4 } else { 3 };
        let n = (w as usize).saturating_mul(h as usize).saturating_mul(bpp);
        if n <= MAX_PIXEL_BYTES {
            let mut data = vec![0u8; n];
            let _ = d.read_image(&mut data);
        }
    }
}
fn animated(input: &[u8]) {
    if let Ok(mut d) = zenwebp::WebPDecoder::new(input) {
        let (w, h) = d.dimensions();
        let bpp = if d.has_alpha() { 4 } else { 3 };
        let n = (w as usize).saturating_mul(h as usize).saturating_mul(bpp);
        if n <= MAX_PIXEL_BYTES && d.is_animated() {
            let mut data = vec![0u8; n];
            let mut guard = 0;
            while let Ok(_) = d.read_frame(&mut data) {
                guard += 1;
                if guard > 100_000 {
                    panic!("read_frame did not terminate");
                }
            }
        }
    }
}
fn v2(input: &[u8]) {
    let c = fuzz_config();
    let _ = zenwebp::DecodeRequest::new(&c, input).decode_rgba();
    let _ = zenwebp::DecodeRequest::new(&c, input).decode_rgb();
    if let Ok(mut d) = zenwebp::mux::AnimationDecoder::new_with_config(input, &c) {
        while let Ok(Some(_)) = d.next_frame() {}
    }
}
fn demux(input: &[u8]) {
    if let Ok(d) = zenwebp::mux::WebPDemuxer::new(input) {
        let _ = d.num_frames();
        for _f in d.frames() {}
    }
    let _ = zenwebp::metadata::icc_profile(input);
    let _ = zenwebp::metadata::exif(input);
    let _ = zenwebp::metadata::xmp(input);
}

#[test]
fn replay() {
    let root = PathBuf::from(std::env::var("REPLAY_DIR").expect("set REPLAY_DIR"));
    let mut files = Vec::new();
    walk(&root, &mut files);
    files.sort();
    std::panic::set_hook(Box::new(|info| {
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        *LAST_PANIC.lock().unwrap() = loc;
    }));
    let entries: [(&str, fn(&[u8])); 4] = [
        ("still", still),
        ("animated", animated),
        ("v2", v2),
        ("demux", demux),
    ];
    let mut by_sig: BTreeMap<String, (usize, PathBuf)> = BTreeMap::new();
    let mut slow: Vec<(Duration, String, PathBuf)> = Vec::new();
    for f in &files {
        let input = std::fs::read(f).unwrap();
        for (name, run) in entries {
            let t = Instant::now();
            let r = catch_unwind(AssertUnwindSafe(|| run(&input)));
            let dt = t.elapsed();
            if r.is_err() {
                let sig = format!("{name} @ {}", LAST_PANIC.lock().unwrap());
                by_sig.entry(sig).or_insert((0, f.clone())).0 += 1;
            }
            if dt > Duration::from_millis(500) {
                slow.push((dt, name.to_string(), f.clone()));
            }
        }
    }
    let _ = std::panic::take_hook();
    eprintln!("replayed {} files from {}", files.len(), root.display());
    for (sig, (n, ex)) in &by_sig {
        eprintln!("PANIC {n:6} {sig}  e.g. {}", ex.display());
    }
    slow.sort();
    for (dt, name, f) in slow.iter().rev().take(20) {
        eprintln!("SLOW {dt:?} {name} {}", f.display());
    }
    assert!(
        by_sig.is_empty() && slow.is_empty(),
        "{} panic signatures, {} slow",
        by_sig.len(),
        slow.len()
    );
}
