#[derive(Clone)]
pub struct InputDiffProb<T: Clone + 'static>(pub f64, pub T);

impl<T: Clone + 'static> std::fmt::Debug for InputDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    self.0.fmt(f)
  }
}

impl<T: Clone + 'static> From<(f64, T)> for InputDiffProb<T> {
  fn from((p, t): (f64, T)) -> Self {
    Self(p, t)
  }
}
