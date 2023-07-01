use super::*;

/// A symbolic version of the tensor, storing its shape and ID under the shape.
/// Conceptually, this is a "pointer" to the actual tensor in the tensor registry.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorSymbol {
  pub shape: TensorShape,
  pub id: usize,
}

impl TensorSymbol {
  /// Create a new tensor symbol
  pub fn new(shape: TensorShape, id: usize) -> Self {
    Self { shape, id }
  }
}

impl std::fmt::Display for TensorSymbol {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("tensor{}(#{})", self.shape, self.id))
  }
}
