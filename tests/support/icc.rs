//! ICC checks using the tag table, PCS values and reference transfer functions.
//! This module is shared by the isolated consumer and integration tests.

pub fn tag<'a>(profile: &'a [u8], signature: &[u8; 4]) -> Option<&'a [u8]> {
    let count = u32_at(profile, 128) as usize;
    (0..count).find_map(|index| {
        let entry = 132 + index * 12;
        (&profile[entry..entry + 4] == signature).then(|| {
            let offset = u32_at(profile, entry + 4) as usize;
            let size = u32_at(profile, entry + 8) as usize;
            &profile[offset..offset + size]
        })
    })
}

fn u32_at(bytes: &[u8], i: usize) -> u32 {
    u32::from_be_bytes(bytes[i..i + 4].try_into().unwrap())
}

fn fixed(bytes: &[u8], i: usize) -> f64 {
    f64::from(u32_at(bytes, i) as i32) / 65536.0
}

pub fn assert_rgb_display(profile: &[u8]) {
    assert_eq!(u32_at(profile, 0) as usize, profile.len());
    assert_eq!(&profile[12..24], b"mntrRGB XYZ ");
    assert_eq!(&profile[36..40], b"acsp");
}

fn assert_colorants(profile: &[u8], columns: [[f64; 3]; 3]) {
    for (name, expected) in [b"rXYZ", b"gXYZ", b"bXYZ"].into_iter().zip(columns) {
        let xyz = tag(profile, name).expect("matrix colorant");
        assert_eq!(&xyz[..4], b"XYZ ");
        for (i, value) in expected.into_iter().enumerate() {
            let actual = fixed(xyz, 8 + 4 * i);
            assert!(
                (actual - value).abs() < 0.001,
                "{name:?} PCS XYZ component {i}: {actual} vs {value}"
            );
        }
    }
}

fn curve(tag: &[u8], x: f64) -> f64 {
    match &tag[..4] {
        b"para" => {
            let kind = u16::from_be_bytes(tag[8..10].try_into().unwrap());
            assert!(matches!(kind, 3 | 4), "unexpected parametric curve {kind}");
            let [g, a, b, c, d] = [12, 16, 20, 24, 28].map(|i| fixed(tag, i));
            let (e, f) = if kind == 4 {
                (fixed(tag, 32), fixed(tag, 36))
            } else {
                (0.0, 0.0)
            };
            if x < d {
                c * x + f
            } else {
                (a * x + b).powf(g) + e
            }
        }
        b"curv" => {
            let entries = u32_at(tag, 8) as usize;
            assert!(entries > 1, "sampled TRC");
            let x = x * (entries - 1) as f64;
            let i = x.floor() as usize;
            let read = |i| {
                let at = 12 + 2 * i;
                f64::from(u16::from_be_bytes(tag[at..at + 2].try_into().unwrap())) / 65535.0
            };
            let y = read(i);
            y + (read((i + 1).min(entries - 1)) - y) * x.fract()
        }
        kind => panic!("unexpected TRC {kind:?}"),
    }
}

fn assert_curves(profile: &[u8], reference: impl Fn(f64) -> f64, tolerance: f64) {
    for name in [b"rTRC", b"gTRC", b"bTRC"] {
        let trc = tag(profile, name).expect("tone reproduction curve");
        for x in [0.0, 0.018, 0.035, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let actual = curve(trc, x);
            let expected = reference(x);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{name:?} at {x}: {actual} vs {expected}"
            );
        }
    }
}

#[allow(dead_code)] // Each consumer tests the profiles it authors.
pub fn assert_p3(profile: &[u8], authored_compatible: bool) {
    assert_rgb_display(profile);
    // The authored compatibility variant redistributes red's negative Z.
    let columns = if authored_compatible {
        [
            [0.515121, 0.241196, 0.0],
            [0.291977, 0.692245, 0.041824],
            [0.157104, 0.066559, 0.783081],
        ]
    } else {
        [
            [0.515102, 0.241182, -0.001049],
            [0.291965, 0.692236, 0.041882],
            [0.157153, 0.066582, 0.784378],
        ]
    };
    assert_colorants(profile, columns);
    assert_curves(
        profile,
        |x| {
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        },
        0.00012,
    );
}

#[allow(dead_code)] // Each consumer tests the profiles it authors.
pub fn assert_pq(profile: &[u8]) {
    assert_rgb_display(profile);
    assert_colorants(
        profile,
        [
            [0.67348, 0.27904, -0.00194],
            [0.16566, 0.67534, 0.02998],
            [0.12505, 0.04561, 0.79684],
        ],
    );
    assert_curves(
        profile,
        |x| {
            let p = x.powf(32.0 / 2523.0);
            ((p - 3424.0 / 4096.0).max(0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * p))
                .powf(16384.0 / 2610.0)
        },
        0.00004,
    );
}
