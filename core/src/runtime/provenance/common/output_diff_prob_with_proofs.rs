#[derive(Clone)]
pub struct OutputDiffProbWithProofs {
  pub probability: f64,
  pub gradient: Vec<(usize, f64)>,
  pub proofs: Vec<Vec<(bool, usize)>>,
}

impl std::fmt::Debug for OutputDiffProbWithProofs {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("")
      .field(&self.probability)
      .field(
        &self
          .gradient
          .iter()
          .map(|(id, weight)| (id, weight))
          .collect::<Vec<_>>(),
      )
      .finish()
  }
}

impl std::fmt::Display for OutputDiffProbWithProofs {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("")
      .field(&self.probability)
      .field(
        &self
          .gradient
          .iter()
          .map(|(id, weight)| (id, weight))
          .collect::<Vec<_>>(),
      )
      .finish()
  }
}
