/// Floating Point trait (f32, f64)
pub trait Float:
  Sized
  + Copy
  + Clone
  + PartialEq
  + PartialOrd
  + std::fmt::Debug
  + std::fmt::Display
  + std::ops::Add<Self, Output = Self>
  + std::ops::Sub<Self, Output = Self>
  + std::ops::Mul<Self, Output = Self>
  + std::ops::Div<Self, Output = Self>
  + std::convert::TryInto<f64>
{
  fn zero() -> Self;

  fn one() -> Self;
}

impl Float for f32 {
  fn zero() -> Self {
    0.0
  }

  fn one() -> Self {
    1.0
  }
}

impl Float for f64 {
  fn zero() -> Self {
    0.0
  }

  fn one() -> Self {
    1.0
  }
}
