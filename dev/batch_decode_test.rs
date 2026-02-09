// Batch decode test: attempts to decode every .webp file in a directory tree.
// Reports failures with file path and error.
//
// Usage: cargo run --release --example batch_decode_test -- <directory>

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("Usage: batch_decode_test <directory>");
    let dir = PathBuf::from(&dir);
    assert!(dir.is_dir(), "Not a directory: {}", dir.display());

    // Collect all .webp files
    let mut files: Vec<PathBuf> = Vec::new();
    collect_webp_files(&dir, &mut files);
    files.sort();

    let total = files.len();
    eprintln!("Found {} .webp files in {}", total, dir.display());

    let success = AtomicUsize::new(0);
    let fail = AtomicUsize::new(0);
    let mut failures: Vec<(PathBuf, String)> = Vec::new();

    let start = Instant::now();

    for (i, path) in files.iter().enumerate() {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                failures.push((path.clone(), format!("IO error: {e}")));
                fail.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };

        match std::panic::catch_unwind(|| zenwebp::decode_rgb(&data)) {
            Ok(Ok(_)) => {
                success.fetch_add(1, Ordering::Relaxed);
            }
            Ok(Err(e)) => {
                let err_str = format!("{e}");
                eprintln!("FAIL [{}/{}]: {} — {}", i + 1, total, path.display(), err_str);
                failures.push((path.clone(), err_str));
                fail.fetch_add(1, Ordering::Relaxed);
            }
            Err(_panic) => {
                let err_str = "PANIC during decode".to_string();
                eprintln!("PANIC [{}/{}]: {} — {}", i + 1, total, path.display(), err_str);
                failures.push((path.clone(), err_str));
                fail.fetch_add(1, Ordering::Relaxed);
            }
        }

        if (i + 1) % 500 == 0 {
            let elapsed = start.elapsed();
            eprintln!(
                "Progress: {}/{} ({:.1}%) in {:.1}s",
                i + 1,
                total,
                (i + 1) as f64 / total as f64 * 100.0,
                elapsed.as_secs_f64()
            );
        }
    }

    let elapsed = start.elapsed();
    let ok = success.load(Ordering::Relaxed);
    let bad = fail.load(Ordering::Relaxed);

    eprintln!("\n=== Results ===");
    eprintln!("Total:   {total}");
    eprintln!("Success: {ok}");
    eprintln!("Failed:  {bad}");
    eprintln!("Time:    {:.1}s ({:.0} files/sec)", elapsed.as_secs_f64(), total as f64 / elapsed.as_secs_f64());

    if !failures.is_empty() {
        eprintln!("\n=== Failures ===");
        for (path, err) in &failures {
            eprintln!("  {} — {}", path.display(), err);
        }
        std::process::exit(1);
    }
}

fn collect_webp_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Warning: cannot read {}: {}", dir.display(), e);
            return;
        }
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.is_dir() {
            collect_webp_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("webp")) {
            out.push(path);
        }
    }
}
