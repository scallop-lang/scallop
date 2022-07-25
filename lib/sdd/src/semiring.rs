pub trait Semiring {
  type Element: Clone;

  fn zero(&self) -> Self::Element;

  fn one(&self) -> Self::Element;

  fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element;

  fn mult(&self, a: Self::Element, b: Self::Element) -> Self::Element;

  fn negate(&self, a: Self::Element) -> Self::Element;
}

pub struct BooleanSemiring;

impl Semiring for BooleanSemiring {
  type Element = bool;

  fn zero(&self) -> Self::Element {
    false
  }

  fn one(&self) -> Self::Element {
    true
  }

  fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
    a || b
  }

  fn mult(&self, a: Self::Element, b: Self::Element) -> Self::Element {
    a && b
  }

  fn negate(&self, a: Self::Element) -> Self::Element {
    !a
  }
}
