#[derive(Clone)]
pub struct RealSemiring;

impl RealSemiring {
  pub fn new() -> Self {
    Self
  }
}

impl sdd::Semiring for RealSemiring {
  type Element = f64;

  fn zero(&self) -> Self::Element {
    0.0
  }

  fn one(&self) -> Self::Element {
    1.0
  }

  fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
    a + b
  }

  fn mult(&self, a: Self::Element, b: Self::Element) -> Self::Element {
    a * b
  }

  fn negate(&self, a: Self::Element) -> Self::Element {
    1.0 - a
  }
}
