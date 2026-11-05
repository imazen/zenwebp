//! Fresh zeroed allocation: previous reserve/fill and vec paths vs bytemuck.
//! Run `just zeroed-alloc-perf`; results include allocation and deallocation.
//! The fill workload also writes every byte; allocation-only timings do not
//! measure first-touch costs. This isolates allocation, not codec throughput.

extern crate alloc;

use std::time::Duration;
use zenbench::{Suite, black_box};

// Exercise the production helper without exposing a public benchmarking API.
// The other helpers and the imported unit-test module are unused in this
// harness-free benchmark target.
#[allow(dead_code, unused_imports)]
#[path = "../src/decoder/alloc_util.rs"]
mod alloc_util;

fn reserve_zeroed(n: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(n).unwrap();
    buf.resize(n, 0);
    buf
}

fn consume(mut buf: Vec<u8>, fill: bool) {
    black_box(buf.as_mut_slice());
    if fill {
        buf.fill(black_box(0xAB));
        black_box(buf.as_mut_slice());
    }
    drop(buf);
}

fn allocations(suite: &mut Suite) {
    for n in [
        0,
        64 * 64 * 4,
        256 * 256 * 4,
        1024 * 1024 * 4,
        4096 * 4096 * 4,
    ] {
        for fill in [false, true] {
            let work = if fill { "fill" } else { "allocate" };
            suite.compare(format!("zeroed_{work}_{n}"), |g| {
                g.config().max_time(Duration::from_secs(2));
                g.bench("vec", move |b| {
                    b.iter(|| consume(vec![0u8; black_box(n)], fill));
                });
                g.bench("reserve_resize", move |b| {
                    b.iter(|| consume(reserve_zeroed(black_box(n)), fill));
                });
                for pref in [
                    alloc_util::AllocPreference::Fallible,
                    alloc_util::AllocPreference::Infallible,
                ] {
                    g.bench(format!("unified_{pref:?}"), move |b| {
                        b.iter(|| {
                            consume(
                                alloc_util::alloc_zeroed(black_box(pref), false, black_box(n))
                                    .unwrap(),
                                fill,
                            );
                        });
                    });
                }
            });
        }
    }
}

zenbench::main!(allocations);
