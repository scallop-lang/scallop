/// Integer trait (i8 - i128, u8 - u128, isize, usize)
pub trait Integer:
  Sized +
  Copy +
  Clone +
  PartialEq +
  Eq +
  PartialOrd +
  Ord +
  std::fmt::Debug +
  std::fmt::Display +
  std::ops::Add<Self, Output=Self> +
  std::ops::Sub<Self, Output=Self> +
  std::ops::Mul<Self, Output=Self> +
  std::ops::Div<Self, Output=Self> +
  std::convert::TryInto<usize> +
  std::convert::TryInto<isize>
{
  fn zero() -> Self;

  fn one() -> Self;
}

macro_rules! impl_integer {
  ($type:ty) => {
    impl Integer for $type {
      fn zero() -> Self { 0 }
      fn one() -> Self { 1 }
    }
  };
}

impl_integer!(i8);
impl_integer!(i16);
impl_integer!(i32);
impl_integer!(i64);
impl_integer!(i128);
impl_integer!(isize);
impl_integer!(u8);
impl_integer!(u16);
impl_integer!(u32);
impl_integer!(u64);
impl_integer!(u128);
impl_integer!(usize);
