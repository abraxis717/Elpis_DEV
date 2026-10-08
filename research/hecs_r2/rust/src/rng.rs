//! Deterministic pseudo-random numbers (SplitMix64 seeding, xoshiro256**). Bit-exact across platforms.

#[derive(Clone)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Rng {
    /// A stream identified by a base seed and a path of labels (world, seed, episode, ...).
    pub fn new(seed: u64, path: &[u64]) -> Rng {
        let mut x = seed;
        for &p in path {
            x = splitmix(&mut x) ^ p.wrapping_mul(0xd6e8_feb8_6659_fd93);
        }
        let mut s = [0u64; 4];
        for v in &mut s {
            *v = splitmix(&mut x);
        }
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Standard normal (Box-Muller, one value per call; deterministic).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn streams_are_deterministic_and_distinct() {
        let a: Vec<u64> = {
            let mut r = Rng::new(7, &[1, 2]);
            (0..4).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = Rng::new(7, &[1, 2]);
            (0..4).map(|_| r.next_u64()).collect()
        };
        let c: Vec<u64> = {
            let mut r = Rng::new(7, &[1, 3]);
            (0..4).map(|_| r.next_u64()).collect()
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut r = Rng::new(1, &[]);
        let n = 20000;
        let mean: f64 = (0..n).map(|_| r.normal()).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.03);
    }
}
