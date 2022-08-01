/// The production type
pub trait ProdType: Sized + std::ops::Mul<Self, Output = Self> {
  fn one() -> Self;

  fn prod<I>(i: I) -> Self
  where
    I: Iterator<Item = Self>,
  {
    let mut agg = Self::one();
    for x in i {
      agg = agg * x;
    }
    agg
  }
}

impl ProdType for u8 {
  fn one() -> Self {
    1
  }
}

impl ProdType for u16 {
  fn one() -> Self {
    1
  }
}

impl ProdType for u32 {
  fn one() -> Self {
    1
  }
}

impl ProdType for u64 {
  fn one() -> Self {
    1
  }
}

impl ProdType for u128 {
  fn one() -> Self {
    1
  }
}

impl ProdType for usize {
  fn one() -> Self {
    1
  }
}

impl ProdType for i8 {
  fn one() -> Self {
    1
  }
}

impl ProdType for i16 {
  fn one() -> Self {
    1
  }
}

impl ProdType for i32 {
  fn one() -> Self {
    1
  }
}

impl ProdType for i64 {
  fn one() -> Self {
    1
  }
}

impl ProdType for i128 {
  fn one() -> Self {
    1
  }
}

impl ProdType for isize {
  fn one() -> Self {
    1
  }
}

impl ProdType for f32 {
  fn one() -> Self {
    1.0
  }
}

impl ProdType for f64 {
  fn one() -> Self {
    1.0
  }
}
