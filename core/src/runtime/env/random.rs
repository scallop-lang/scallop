use std::sync::*;

use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct Random {
  pub rng: Arc<Mutex<SmallRng>>,
}

impl Random {
  /// Create a new random module
  pub fn new(seed: u64) -> Self {
    Self {
      rng: Arc::new(Mutex::new(SmallRng::seed_from_u64(seed))),
    }
  }

  /// Sample an element from a distribution using the rng
  pub fn sample_from<T, D: rand::distributions::Distribution<T>>(&self, dist: &D) -> T {
    dist.sample(&mut *self.rng.lock().unwrap())
  }

  /// Sample a number between 0 and `n` (exclusive)
  pub fn random_usize(&self, n: usize) -> usize {
    self.rng.lock().unwrap().gen_range(0..n)
  }
}
