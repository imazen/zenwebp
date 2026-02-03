//! Color cache for VP8L encoding.
//!
//! The color cache stores recently used colors and allows referencing them
//! by index instead of encoding the full ARGB value.

/// Color cache hash multiplier (must match decoder exactly).
const COLOR_CACHE_MULT: u32 = 0x1e35a7bd;

/// Color cache for encoding.
#[derive(Debug, Clone)]
pub struct ColorCache {
    /// Cache entries (ARGB values).
    colors: Vec<u32>,
    /// Number of bits (1-11).
    bits: u8,
    /// Hash shift value (32 - bits).
    hash_shift: u32,
}

impl ColorCache {
    /// Create a new color cache with the given number of bits.
    /// `bits` must be in range 1-11.
    pub fn new(bits: u8) -> Self {
        debug_assert!((1..=11).contains(&bits));
        let size = 1 << bits;
        Self {
            colors: vec![0; size],
            bits,
            hash_shift: 32 - bits as u32,
        }
    }

    /// Get the number of cache bits.
    #[inline]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Get the cache size.
    #[inline]
    pub fn size(&self) -> usize {
        self.colors.len()
    }

    /// Compute hash index for a color.
    #[inline]
    fn hash(&self, argb: u32) -> usize {
        (COLOR_CACHE_MULT.wrapping_mul(argb) >> self.hash_shift) as usize
    }

    /// Insert a color into the cache.
    #[inline]
    pub fn insert(&mut self, argb: u32) {
        let idx = self.hash(argb);
        self.colors[idx] = argb;
    }

    /// Check if a color is in the cache and return its index.
    /// Returns `None` if not found.
    #[inline]
    pub fn lookup(&self, argb: u32) -> Option<u16> {
        let idx = self.hash(argb);
        if self.colors[idx] == argb {
            Some(idx as u16)
        } else {
            None
        }
    }

    /// Get the color at a given index.
    #[inline]
    pub fn get(&self, idx: u16) -> u32 {
        self.colors[idx as usize]
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.colors.fill(0);
    }
}

/// Determine optimal cache bits by estimating entropy for different sizes.
pub fn estimate_optimal_cache_bits(
    pixels: &[u32],
    _width: usize,
    quality: u8,
) -> u8 {
    if quality < 25 || pixels.len() < 100 {
        return 0; // Disable cache for low quality or tiny images
    }

    // Sample pixels to estimate cache hit rate
    let sample_size = (pixels.len() / 16).clamp(256, 4096);
    let step = pixels.len() / sample_size;

    let mut best_bits = 0u8;
    let mut best_savings = 0i32;

    // Try different cache sizes
    for bits in 1..=10u8 {
        let mut cache = ColorCache::new(bits);
        let mut hits = 0u32;
        let mut total = 0u32;

        for i in (0..pixels.len()).step_by(step) {
            let argb = pixels[i];
            if cache.lookup(argb).is_some() {
                hits += 1;
            }
            cache.insert(argb);
            total += 1;
        }

        // Estimate savings: hit rate * (32 - cache_bits) bits saved per hit
        // minus overhead of cache codes in alphabet
        let hit_rate = hits as f32 / total.max(1) as f32;
        let bits_saved_per_hit = 32 - bits as i32;
        let overhead_per_pixel = (bits as i32) / 4; // Rough approximation
        let savings = (hit_rate * bits_saved_per_hit as f32 - overhead_per_pixel as f32) as i32;

        if savings > best_savings {
            best_savings = savings;
            best_bits = bits;
        }
    }

    if best_savings <= 0 {
        0 // Cache not beneficial
    } else {
        best_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_cache_insert_lookup() {
        let mut cache = ColorCache::new(4);
        let color = 0xFF112233u32;

        // Initially not found
        assert!(cache.lookup(color).is_none());

        // After insert, found
        cache.insert(color);
        let idx = cache.lookup(color);
        assert!(idx.is_some());

        // Same index gives same color back
        assert_eq!(cache.get(idx.unwrap()), color);
    }

    #[test]
    fn test_color_cache_collision() {
        let mut cache = ColorCache::new(1); // Only 2 entries
        let c1 = 0xFF000000u32;
        let c2 = 0xFF000001u32;

        cache.insert(c1);
        cache.insert(c2);

        // One of them may have been evicted due to hash collision
        let found1 = cache.lookup(c1).is_some();
        let found2 = cache.lookup(c2).is_some();

        // At least one should be found (the last inserted if they collide)
        assert!(found1 || found2);
    }

    #[test]
    fn test_hash_determinism() {
        let cache = ColorCache::new(8);
        let color = 0xAABBCCDDu32;

        // Hash should be deterministic
        let h1 = cache.lookup(color);
        let h2 = cache.lookup(color);
        assert_eq!(h1, h2);
    }
}
