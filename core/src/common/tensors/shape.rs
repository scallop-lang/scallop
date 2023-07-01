/// The shape of a tensor
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorShape(Box<[i64]>);

impl TensorShape {
  pub fn scalar() -> Self {
    Self(Box::new([]))
  }

  /// Get the number of dimensions
  pub fn dim(&self) -> usize {
    self.0.len()
  }
}

impl From<Vec<i64>> for TensorShape {
  fn from(shape: Vec<i64>) -> Self {
    Self(shape.into_iter().collect())
  }
}

impl std::ops::Index<usize> for TensorShape {
  type Output = i64;

  fn index(&self, index: usize) -> &Self::Output {
    &self.0[index]
  }
}

impl std::fmt::Display for TensorShape {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("<")?;
    for i in 0..self.dim() {
      if i > 0 {
        f.write_str(", ")?;
      }
      self.0[i].fmt(f)?;
    }
    f.write_str(">")
  }
}
