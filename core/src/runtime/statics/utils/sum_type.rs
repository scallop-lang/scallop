/// The sum type
pub trait SumType: Sized + std::ops::Add<Self, Output = Self> {
  fn zero() -> Self;

  fn sum<I>(i: I) -> Self
  where
    I: Iterator<Item = Self>,
  {
    let mut agg = Self::zero();
    for x in i {
      agg = agg + x;
    }
    agg
  }
}

impl SumType for u8 {
  fn zero() -> Self {
    0
  }
}

impl SumType for u16 {
  fn zero() -> Self {
    0
  }
}

impl SumType for u32 {
  fn zero() -> Self {
    0
  }
}

impl SumType for u64 {
  fn zero() -> Self {
    0
  }
}

impl SumType for u128 {
  fn zero() -> Self {
    0
  }
}

impl SumType for usize {
  fn zero() -> Self {
    0
  }
}

impl SumType for i8 {
  fn zero() -> Self {
    0
  }
}

impl SumType for i16 {
  fn zero() -> Self {
    0
  }
}

impl SumType for i32 {
  fn zero() -> Self {
    0
  }
}

impl SumType for i64 {
  fn zero() -> Self {
    0
  }
}

impl SumType for i128 {
  fn zero() -> Self {
    0
  }
}

impl SumType for isize {
  fn zero() -> Self {
    0
  }
}

impl SumType for f32 {
  fn zero() -> Self {
    0.0
  }
}

impl SumType for f64 {
  fn zero() -> Self {
    0.0
  }
}
