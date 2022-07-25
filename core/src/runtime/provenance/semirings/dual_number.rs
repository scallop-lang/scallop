use sprs::CsVec;

#[derive(Clone)]
pub struct DualNumber {
  pub real: f64,
  pub deriv: CsVec<f64>,
}

#[derive(Clone)]
pub struct DualNumberSemiring {
  pub dim: usize,
}

impl DualNumberSemiring {
  pub fn new(dim: usize) -> Self {
    Self { dim }
  }

  pub fn singleton(&self, real: f64, id: usize) -> DualNumber {
    DualNumber {
      real,
      deriv: CsVec::new(self.dim, vec![id], vec![1.0]),
    }
  }
}

impl sdd::Semiring for DualNumberSemiring {
  type Element = DualNumber;

  fn zero(&self) -> Self::Element {
    DualNumber {
      real: 0.0,
      deriv: CsVec::empty(self.dim),
    }
  }

  fn one(&self) -> Self::Element {
    DualNumber {
      real: 1.0,
      deriv: CsVec::empty(self.dim),
    }
  }

  fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
    DualNumber {
      real: a.real + b.real,
      deriv: a.deriv + b.deriv,
    }
  }

  fn mult(&self, a: Self::Element, b: Self::Element) -> Self::Element {
    DualNumber {
      real: a.real * b.real,
      deriv: a.deriv.map(|v| v * b.real) + b.deriv.map(|v| v * a.real),
    }
  }

  fn negate(&self, a: Self::Element) -> Self::Element {
    DualNumber {
      real: 1.0 - a.real,
      deriv: -a.deriv,
    }
  }
}
