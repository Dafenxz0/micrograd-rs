//! Deterministic SplitMix64 PRNG so the crate requires zero external dependencies.

use std::f64::consts::TAU;

/// A small, reproducible pseudo-random number generator.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a new generator from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next pseudo-random `u64` using SplitMix64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Returns a uniform `f64` in `[0.0, 1.0)`.
    pub fn gen_unit(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    /// Returns a uniform `f64` in `[low, high)`.
    pub fn gen_range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.gen_unit()
    }

    /// Returns a standard normal `N(0, 1)` sample via Box-Muller transform.
    pub fn gen_normal(&mut self) -> f64 {
        let u1 = self.gen_unit().max(1e-12);
        let u2 = self.gen_unit();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}
