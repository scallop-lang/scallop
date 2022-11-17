#[derive(Clone)]
pub struct OutputDiffProb<T: Clone + 'static>(pub f64, pub Vec<(usize, f64, T)>);

impl<T: Clone + 'static> std::fmt::Debug for OutputDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("")
      .field(&self.0)
      .field(&self.1.iter().map(|(id, weight, _)| (id, weight)).collect::<Vec<_>>())
      .finish()
  }
}

impl<T: Clone + 'static> std::fmt::Display for OutputDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("")
      .field(&self.0)
      .field(&self.1.iter().map(|(id, weight, _)| (id, weight)).collect::<Vec<_>>())
      .finish()
  }
}
