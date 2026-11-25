//! Stateless, counter-based random numbers: every draw is a hash of its
//! coordinates, so the GPU sim can reproduce the CPU draws exactly.

/// PCG hash (Jarzynski & Olano 2020).
#[inline]
pub fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Hashes a sequence of words.
#[inline]
pub fn hash(words: &[u32]) -> u32 {
    let mut h = 0x9E37_79B9u32;
    for &w in words {
        h = pcg(h ^ w);
    }
    h
}

/// Maps a hash to [0, 1) using its top 24 bits (exact in f32).
#[inline]
pub fn to_unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Where a random draw happens. Two draws with different keys are independent.
#[derive(Clone, Copy, Debug)]
pub struct RandKey {
    pub seed: u32,
    pub emitter: u32,
    /// Particle UniqueID, or `u32::MAX` for emitter-level draws.
    pub particle: u32,
    /// Step number for update stages; 0 for spawn stages (one draw per life).
    pub frame: u32,
    /// Stage (0..6) * 256 + module index in the stage.
    pub module: u32,
}

impl RandKey {
    /// Uniform [0, 1) for slot `slot` (input index * 8 + component, or 128+ for module-internal draws).
    pub fn unit(&self, slot: u32) -> f32 {
        to_unit(hash(&[
            self.seed,
            self.emitter,
            self.particle,
            self.frame,
            self.module,
            slot,
        ]))
    }
}
