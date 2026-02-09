// Batch decode comparison: zenwebp vs webpx (libwebp) for pixel-level correctness.
// Reports any files where the decoded pixels differ.
//
// Usage: cargo run --release --example batch_decode_compare -- <directory>

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("Usage: batch_decode_compare <directory>");
    let dir = PathBuf::from(&dir);
    assert!(dir.is_dir(), "Not a directory: {}", dir.display());

    let mut files: Vec<PathBuf> = Vec::new();
    collect_webp_files(&dir, &mut files);
    files.sort();

    let total = files.len();
    eprintln!("Found {} .webp files in {}", total, dir.display());

    let mut match_count = 0usize;
    let mut mismatch_count = 0usize;
    let mut zen_fail = 0usize;
    let mut ref_fail = 0usize;
    let mut both_fail = 0usize;
    let mut mismatches: Vec<(PathBuf, String)> = Vec::new();

    let start = Instant::now();

    for (i, path) in files.iter().enumerate() {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("IO [{}/{}]: {} — {}", i + 1, total, path.display(), e);
                continue;
            }
        };

        let zen_result = std::panic::catch_unwind(|| zenwebp::decode_rgb(&data));
        let ref_result = std::panic::catch_unwind(|| webpx::decode_rgb(&data));

        match (zen_result, ref_result) {
            (Ok(Ok((zen_pixels, zw, zh))), Ok(Ok((ref_pixels, rw, rh)))) => {
                if zw != rw || zh != rh {
                    let msg = format!("dimension mismatch: zen={}x{} ref={}x{}", zw, zh, rw, rh);
                    eprintln!("MISMATCH [{}/{}]: {} — {}", i + 1, total, path.display(), msg);
                    mismatches.push((path.clone(), msg));
                    mismatch_count += 1;
                } else if zen_pixels != ref_pixels {
                    // Compute per-pixel stats
                    let npixels = (zw as usize) * (zh as usize);
                    let mut max_diff: u8 = 0;
                    let mut diff_pixels = 0usize;
                    let mut sum_abs_diff: u64 = 0;
                    for (z, r) in zen_pixels.iter().zip(ref_pixels.iter()) {
                        let d = z.abs_diff(*r);
                        if d > 0 {
                            diff_pixels += 1;
                            sum_abs_diff += d as u64;
                            if d > max_diff {
                                max_diff = d;
                            }
                        }
                    }
                    // diff_pixels counts channels, convert to pixel count
                    let diff_pixel_count = diff_pixels / 3; // approximate
                    let avg_diff = sum_abs_diff as f64 / diff_pixels.max(1) as f64;
                    let msg = format!(
                        "pixel mismatch: {}x{}, {}/{} channels differ ({} pixels ~{:.1}%), max_diff={}, avg_diff={:.2}",
                        zw, zh, diff_pixels, zen_pixels.len(), diff_pixel_count,
                        diff_pixel_count as f64 / npixels as f64 * 100.0,
                        max_diff, avg_diff
                    );
                    if max_diff > 1 {
                        eprintln!("MISMATCH [{}/{}]: {} — {}", i + 1, total, path.display(), msg);
                    }
                    mismatches.push((path.clone(), msg));
                    mismatch_count += 1;
                } else {
                    match_count += 1;
                }
            }
            (Ok(Err(ze)), Ok(Err(_re))) => {
                // Both fail — that's ok (possibly invalid/animated/lossless)
                both_fail += 1;
                let _ = ze;
            }
            (Ok(Err(ze)), Ok(Ok(_))) => {
                let msg = format!("zenwebp failed but webpx succeeded: {}", ze);
                eprintln!("ZEN_FAIL [{}/{}]: {} — {}", i + 1, total, path.display(), msg);
                mismatches.push((path.clone(), msg));
                zen_fail += 1;
            }
            (Ok(Ok(_)), Ok(Err(re))) => {
                // zenwebp succeeded, webpx failed — interesting but not our bug
                ref_fail += 1;
                eprintln!("REF_FAIL [{}/{}]: {} — {}", i + 1, total, path.display(), re);
                let _ = re;
            }
            (Err(_), _) => {
                eprintln!("ZEN_PANIC [{}/{}]: {}", i + 1, total, path.display());
                mismatches.push((path.clone(), "zenwebp panicked".into()));
                zen_fail += 1;
            }
            (_, Err(_)) => {
                eprintln!("REF_PANIC [{}/{}]: {}", i + 1, total, path.display());
                ref_fail += 1;
            }
        }

        if (i + 1) % 500 == 0 {
            let elapsed = start.elapsed();
            eprintln!(
                "Progress: {}/{} ({:.1}%) in {:.1}s — {} match, {} mismatch, {} zen_fail, {} ref_fail, {} both_fail",
                i + 1, total, (i + 1) as f64 / total as f64 * 100.0, elapsed.as_secs_f64(),
                match_count, mismatch_count, zen_fail, ref_fail, both_fail
            );
        }
    }

    let elapsed = start.elapsed();
    eprintln!("\n=== Results ===");
    eprintln!("Total:      {total}");
    eprintln!("Match:      {match_count}");
    eprintln!("Mismatch:   {mismatch_count}");
    eprintln!("Zen fail:   {zen_fail}");
    eprintln!("Ref fail:   {ref_fail}");
    eprintln!("Both fail:  {both_fail}");
    eprintln!("Time:       {:.1}s ({:.0} files/sec)", elapsed.as_secs_f64(), total as f64 / elapsed.as_secs_f64());

    if !mismatches.is_empty() {
        eprintln!("\n=== Mismatches & Failures ===");
        // Summarize by category
        let pixel_mismatches: Vec<_> = mismatches.iter().filter(|(_, m)| m.starts_with("pixel")).collect();
        let dim_mismatches: Vec<_> = mismatches.iter().filter(|(_, m)| m.starts_with("dimension")).collect();
        let zen_failures: Vec<_> = mismatches.iter().filter(|(_, m)| m.contains("zenwebp")).collect();

        if !dim_mismatches.is_empty() {
            eprintln!("\nDimension mismatches ({}):", dim_mismatches.len());
            for (p, m) in &dim_mismatches {
                eprintln!("  {} — {}", p.display(), m);
            }
        }

        if !zen_failures.is_empty() {
            eprintln!("\nZenwebp decode failures ({}):", zen_failures.len());
            for (p, m) in &zen_failures {
                eprintln!("  {} — {}", p.display(), m);
            }
        }

        if !pixel_mismatches.is_empty() {
            // Group by max_diff severity
            let severe: Vec<_> = pixel_mismatches.iter().filter(|(_, m)| {
                m.contains("max_diff=") && {
                    let d: u8 = m.split("max_diff=").nth(1).unwrap_or("0").split(',').next().unwrap_or("0").parse().unwrap_or(0);
                    d > 1
                }
            }).collect();
            let off_by_one: Vec<_> = pixel_mismatches.iter().filter(|(_, m)| {
                m.contains("max_diff=1,") || m.contains("max_diff=1\n") || m.ends_with("max_diff=1")
            }).collect();

            eprintln!("\nPixel mismatches: {} total ({} off-by-one, {} severe)",
                pixel_mismatches.len(), off_by_one.len(), severe.len());

            if !severe.is_empty() {
                eprintln!("\nSevere pixel mismatches (max_diff > 1):");
                for (p, m) in severe.iter().take(20) {
                    eprintln!("  {} — {}", p.display(), m);
                }
                if severe.len() > 20 {
                    eprintln!("  ... and {} more", severe.len() - 20);
                }
            }
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
