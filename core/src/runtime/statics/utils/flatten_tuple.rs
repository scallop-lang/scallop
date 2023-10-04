pub trait BaseType {}
impl BaseType for i8 {}
impl BaseType for i16 {}
impl BaseType for i32 {}
impl BaseType for i64 {}
impl BaseType for i128 {}
impl BaseType for isize {}
impl BaseType for u8 {}
impl BaseType for u16 {}
impl BaseType for u32 {}
impl BaseType for u64 {}
impl BaseType for u128 {}
impl BaseType for usize {}
impl BaseType for f32 {}
impl BaseType for f64 {}
impl BaseType for bool {}
impl BaseType for char {}
impl BaseType for String {}
impl BaseType for &'static str {}

pub trait TupleLength {
  fn len() -> usize;
}

impl<T: BaseType> TupleLength for T {
  fn len() -> usize {
    1
  }
}
impl TupleLength for () {
  fn len() -> usize {
    0
  }
}
impl<T> TupleLength for (T,) {
  fn len() -> usize {
    1
  }
}
impl<T1, T2> TupleLength for (T1, T2) {
  fn len() -> usize {
    2
  }
}
impl<T1, T2, T3> TupleLength for (T1, T2, T3) {
  fn len() -> usize {
    3
  }
}
impl<T1, T2, T3, T4> TupleLength for (T1, T2, T3, T4) {
  fn len() -> usize {
    4
  }
}
impl<T1, T2, T3, T4, T5> TupleLength for (T1, T2, T3, T4, T5) {
  fn len() -> usize {
    5
  }
}
impl<T1, T2, T3, T4, T5, T6> TupleLength for (T1, T2, T3, T4, T5, T6) {
  fn len() -> usize {
    6
  }
}
impl<T1, T2, T3, T4, T5, T6, T7> TupleLength for (T1, T2, T3, T4, T5, T6, T7) {
  fn len() -> usize {
    7
  }
}

pub trait FlattenTuple {
  type Output;

  fn flatten(self) -> Self::Output;

  fn unflatten(other: Self::Output) -> Self;
}

impl<T1: BaseType, T2: BaseType> FlattenTuple for (T1, T2) {
  type Output = (T1, T2);
  fn flatten(self) -> Self::Output {
    self
  }
  fn unflatten(other: Self::Output) -> Self {
    other
  }
}
