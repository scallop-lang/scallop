use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

const RANDOM_SEED: u64 = 1234;

#[derive(Clone, Debug)]
pub struct RuntimeEnvironment {
  pub random_seed: u64,
  pub rng: SmallRng,
}

impl Default for RuntimeEnvironment {
  fn default() -> Self {
    Self::new()
  }
}

impl RuntimeEnvironment {
  pub fn new() -> Self {
    Self {
      random_seed: RANDOM_SEED,
      rng: SmallRng::seed_from_u64(RANDOM_SEED),
    }
  }

  pub fn new_with_random_seed(seed: u64) -> Self {
    Self {
      random_seed: seed,
      rng: SmallRng::seed_from_u64(seed),
    }
  }

  pub fn random_u64(&mut self) -> u64 {
    self.rng.gen()
  }

  pub fn random_u32(&mut self) -> u32 {
    self.rng.gen()
  }

  pub fn random_f32(&mut self) -> f32 {
    self.rng.gen()
  }

  pub fn random_f64(&mut self) -> f64 {
    self.rng.gen()
  }
}
