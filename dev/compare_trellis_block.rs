//! Compare trellis quantization on a single block between zenwebp and libwebp.
//!
//! This creates a synthetic block with known coefficients and compares
//! the trellis output between both encoders.

use zenwebp::encoder::cost::{
    trellis_quantize_block, LevelCosts, MatrixType, PsyConfig, VP8Matrix,
};
use zenwebp::encoder::tables::VP8_ZIGZAG;

fn main() {
    // Simulate typical I4 block DCT coefficients (in natural order)
    // Large DC, decaying AC - common pattern
    let coeffs_natural: [i32; 16] = [256, 64, 32, 16, 48, 24, 12, 8, 16, 8, 4, 2, 8, 4, 2, 1];

    // Quality 75 quantizer values
    let q_dc = 24u16;
    let q_ac = 30u16;
    let y1_matrix = VP8Matrix::new(q_dc, q_ac, MatrixType::Y1);

    // Lambda for trellis (method 4+ uses lambda_trellis_i4)
    // From libwebp: lambda_trellis_i4 = (7 * q_i4 * q_i4) >> 3
    // where q_i4 = average of quantizers
    let q_i4 = y1_matrix.average_q();
    let lambda_trellis = (7 * q_i4 * q_i4) >> 3;

    println!("=== Trellis Block Comparison ===");
    println!("Q: DC={} AC={}", q_dc, q_ac);
    println!("q_i4={} lambda_trellis={}", q_i4, lambda_trellis);
    println!();

    // Print matrix parameters
    println!("Y1 Matrix:");
    println!("  q:       {:?}", &y1_matrix.q);
    println!("  iq:      {:?}", &y1_matrix.iq);
    println!(
        "  bias[0]: {} bias[1]: {}",
        y1_matrix.bias[0], y1_matrix.bias[1]
    );
    println!("  zthresh: {:?}", &y1_matrix.zthresh);
    println!("  sharpen: {:?}", &y1_matrix.sharpen);
    println!();

    // Print input coefficients
    println!("Input coefficients (natural order):");
    for row in 0..4 {
        print!("  ");
        for col in 0..4 {
            print!("{:4} ", coeffs_natural[row * 4 + col]);
        }
        println!();
    }
    println!();

    // Initialize level costs with default probabilities
    let level_costs = LevelCosts::new();

    // Run trellis quantization
    let mut coeffs = coeffs_natural;
    let mut out = [0i32; 16];
    let ctype = 1; // TYPE_I4_AC (Y_AC = 1 in our TokenType enum)
    let ctx0 = 0; // Initial context

    let has_nz = trellis_quantize_block(
        &mut coeffs,
        &mut out,
        &y1_matrix,
        lambda_trellis,
        0, // first
        &level_costs,
        ctype,
        ctx0,
        &PsyConfig::default(), // no psy-trellis for debugging
    );

    println!("Output (zigzag order): {:?}", out);
    println!("Has non-zero: {}", has_nz);
    println!();

    // Convert output to natural order for display
    let mut out_natural = [0i32; 16];
    for n in 0..16 {
        let j = VP8_ZIGZAG[n];
        out_natural[j] = out[n];
    }

    println!("Output (natural order):");
    for row in 0..4 {
        print!("  ");
        for col in 0..4 {
            print!("{:4} ", out_natural[row * 4 + col]);
        }
        println!();
    }
    println!();

    // Print dequantized (reconstructed) coefficients
    println!("Dequantized (reconstructed, natural order):");
    for row in 0..4 {
        print!("  ");
        for col in 0..4 {
            print!("{:4} ", coeffs[row * 4 + col]);
        }
        println!();
    }

    // Also test the simple quantization path for comparison
    println!("\n=== Simple Quantization (no trellis) ===");
    let mut simple_out = [0i32; 16];
    for n in 0..16 {
        let j = VP8_ZIGZAG[n];
        simple_out[n] = y1_matrix.quantize_coeff(coeffs_natural[j], j);
    }
    println!("Output (zigzag order): {:?}", simple_out);

    let mut simple_natural = [0i32; 16];
    for n in 0..16 {
        let j = VP8_ZIGZAG[n];
        simple_natural[j] = simple_out[n];
    }
    println!("Output (natural order):");
    for row in 0..4 {
        print!("  ");
        for col in 0..4 {
            print!("{:4} ", simple_natural[row * 4 + col]);
        }
        println!();
    }
}
