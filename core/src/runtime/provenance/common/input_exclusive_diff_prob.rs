#[derive(Clone)]
pub struct InputExclusiveDiffProb<T: Clone + 'static> {
  /// The probability of the tag
  pub prob: f64,

  /// The external tag for differentiability
  pub tag: T,

  /// An optional identifier of the mutual exclusion
  pub exclusion: Option<usize>,
}

impl<T: Clone + 'static> InputExclusiveDiffProb<T> {
  pub fn new(prob: f64, tag: T, exclusion: Option<usize>) -> Self {
    Self { prob, tag, exclusion }
  }
}

impl<T: Clone + 'static> std::fmt::Debug for InputExclusiveDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.prob.fmt(f)
  }
}

impl<T: Clone + 'static> From<(f64, T, Option<usize>)> for InputExclusiveDiffProb<T> {
  fn from((prob, tag, exclusion): (f64, T, Option<usize>)) -> Self {
    Self { prob, tag, exclusion }
  }
}
