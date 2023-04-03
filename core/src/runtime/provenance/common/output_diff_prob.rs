#[derive(Clone)]
pub struct OutputDiffProb(pub f64, pub Vec<(usize, f64)>);

impl std::fmt::Debug for OutputDiffProb {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("")
      .field(&self.0)
      .field(&self.1.iter().map(|(id, weight)| (id, weight)).collect::<Vec<_>>())
      .finish()
  }
}

impl std::fmt::Display for OutputDiffProb {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("")
      .field(&self.0)
      .field(&self.1.iter().map(|(id, weight)| (id, weight)).collect::<Vec<_>>())
      .finish()
  }
}
