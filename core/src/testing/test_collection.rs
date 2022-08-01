use crate::common::tuple::Tuple;
use crate::common::value::Value;
use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub struct TestCollection {
  pub elements: Vec<Tuple>,
}

impl TestCollection {
  pub fn empty() -> Self {
    Self { elements: vec![] }
  }
}

impl<T: Into<Tuple>> From<Vec<T>> for TestCollection {
  fn from(v: Vec<T>) -> Self {
    Self {
      elements: v.into_iter().map(|t| t.into()).collect(),
    }
  }
}

pub struct TestCollectionWithTag<T: Tag> {
  pub elements: Vec<(OutputTagOf<T::Context>, Tuple)>,
}

impl<T: Tag> TestCollectionWithTag<T> {
  pub fn empty() -> Self {
    Self { elements: vec![] }
  }
}

impl<T: Tag, Tup: Into<Tuple>> From<Vec<(OutputTagOf<T::Context>, Tup)>> for TestCollectionWithTag<T> {
  fn from(v: Vec<(OutputTagOf<T::Context>, Tup)>) -> Self {
    Self {
      elements: v.into_iter().map(|(tag, tup)| (tag, tup.into())).collect(),
    }
  }
}

pub fn test_equals(t1: &Tuple, t2: &Tuple) -> bool {
  match (t1, t2) {
    (Tuple::Tuple(ts1), Tuple::Tuple(ts2)) => ts1.iter().zip(ts2.iter()).all(|(s1, s2)| test_equals(s1, s2)),
    (Tuple::Value(Value::F32(f1)), Tuple::Value(Value::F32(f2))) => (f1 - f2).abs() < 0.001,
    (Tuple::Value(Value::F64(f1)), Tuple::Value(Value::F64(f2))) => (f1 - f2).abs() < 0.001,
    _ => t1 == t2,
  }
}

pub fn expect_collection<T, C>(actual: &DynamicCollection<T>, expected: C)
where
  T: Tag + std::fmt::Debug,
  C: Into<TestCollection>,
{
  let expected = Into::<TestCollection>::into(expected);

  // First check everything in expected is in actual
  for e in &expected.elements {
    let te = e.clone().into();
    let pos = actual.iter().position(|elem| test_equals(&elem.tuple, &te));
    assert!(pos.is_some(), "Tuple {:?} not found in collection {:?}", te, actual)
  }

  // Then check everything in actual is in expected
  for elem in &actual.elements {
    let pos = expected
      .elements
      .iter()
      .position(|e| test_equals(&e.clone().into(), &elem.tuple));
    assert!(
      pos.is_some(),
      "Tuple {:?} is derived in collection but not found in expected set",
      elem
    )
  }
}

pub fn expect_output_collection<T, C>(actual: &DynamicOutputCollection<T>, expected: C)
where
  T: Tag + std::fmt::Debug,
  C: Into<TestCollection>,
{
  let expected = Into::<TestCollection>::into(expected);

  // First check everything in expected is in actual
  for e in &expected.elements {
    let te = e.clone().into();
    let pos = actual.iter().position(|(_, tuple)| test_equals(&tuple, &te));
    assert!(pos.is_some(), "Tuple {:?} not found in collection {:?}", te, actual)
  }

  // Then check everything in actual is in expected
  for elem in &actual.elements {
    let pos = expected
      .elements
      .iter()
      .position(|e| test_equals(&e.clone().into(), &elem.1));
    assert!(
      pos.is_some(),
      "Tuple {:?} is derived in collection but not found in expected set",
      elem
    )
  }
}

pub fn expect_output_collection_with_tag<T, C, F>(actual: &DynamicOutputCollection<T>, expected: C, cmp: F)
where
  T: Tag + std::fmt::Debug,
  C: Into<TestCollectionWithTag<T>>,
  F: Fn(&OutputTagOf<T::Context>, &OutputTagOf<T::Context>) -> bool,
{
  let expected = Into::<TestCollectionWithTag<T>>::into(expected);

  // First check everything in expected is in actual
  for e in &expected.elements {
    let (tage, te) = e.clone();
    let pos = actual
      .iter()
      .position(|(tag, tuple)| test_equals(&tuple, &te) && cmp(&tage, tag));
    assert!(
      pos.is_some(),
      "Tagged Tuple {:?} not found in collection {:?}",
      (tage, te),
      actual
    )
  }

  // Then check everything in actual is in expected
  for elem in &actual.elements {
    let pos = expected
      .elements
      .iter()
      .position(|(tag, tup)| test_equals(&tup.clone().into(), &elem.1) && cmp(tag, &elem.0));
    assert!(
      pos.is_some(),
      "Tagged Tuple {:?} is derived in collection but not found in expected set",
      elem
    )
  }
}

pub fn expect_static_collection<Tup, T>(actual: &StaticCollection<Tup, T>, expected: Vec<Tup>)
where
  Tup: StaticTupleTrait + StaticEquals,
  T: Tag + std::fmt::Debug,
{
  // First check everything in expected is in actual
  for e in &expected {
    let pos = actual
      .elements
      .iter()
      .position(|elem| StaticEquals::test_static_equals(elem.tuple.get(), e));
    assert!(pos.is_some(), "Tuple {:?} not found in collection {:?}", e, actual)
  }

  // Then check everything in actual is in expected
  for elem in &actual.elements {
    let pos = expected
      .iter()
      .position(|e| StaticEquals::test_static_equals(e, elem.tuple.get()));
    assert!(
      pos.is_some(),
      "Tuple {:?} is derived in collection but not found in expected set",
      elem
    )
  }
}

pub fn expect_static_output_collection<Tup, T>(actual: &StaticOutputCollection<Tup, T>, expected: Vec<Tup>)
where
  Tup: StaticTupleTrait + StaticEquals,
  T: Tag + std::fmt::Debug,
{
  // First check everything in expected is in actual
  for e in &expected {
    let pos = actual
      .elements
      .iter()
      .position(|elem| StaticEquals::test_static_equals(&elem.1, e));
    assert!(pos.is_some(), "Tuple {:?} not found in collection {:?}", e, actual)
  }

  // Then check everything in actual is in expected
  for elem in &actual.elements {
    let pos = expected
      .iter()
      .position(|e| StaticEquals::test_static_equals(e, &elem.1));
    assert!(
      pos.is_some(),
      "Tuple {:?} is derived in collection but not found in expected set",
      elem
    )
  }
}

pub trait StaticEquals {
  fn test_static_equals(t1: &Self, t2: &Self) -> bool;
}

macro_rules! impl_static_equals_for_value_type {
  ($ty:ty) => {
    impl StaticEquals for $ty {
      fn test_static_equals(t1: &Self, t2: &Self) -> bool {
        t1 == t2
      }
    }
  };
}

impl_static_equals_for_value_type!(u8);
impl_static_equals_for_value_type!(u16);
impl_static_equals_for_value_type!(u32);
impl_static_equals_for_value_type!(u64);
impl_static_equals_for_value_type!(u128);
impl_static_equals_for_value_type!(usize);
impl_static_equals_for_value_type!(i8);
impl_static_equals_for_value_type!(i16);
impl_static_equals_for_value_type!(i32);
impl_static_equals_for_value_type!(i64);
impl_static_equals_for_value_type!(i128);
impl_static_equals_for_value_type!(isize);
impl_static_equals_for_value_type!(bool);
impl_static_equals_for_value_type!(char);
impl_static_equals_for_value_type!(&str);
impl_static_equals_for_value_type!(String);

impl StaticEquals for f32 {
  fn test_static_equals(t1: &Self, t2: &Self) -> bool {
    (t1 - t2).abs() < 0.001
  }
}

impl StaticEquals for f64 {
  fn test_static_equals(t1: &Self, t2: &Self) -> bool {
    (t1 - t2).abs() < 0.001
  }
}

impl StaticEquals for () {
  fn test_static_equals(t1: &Self, t2: &Self) -> bool {
    t1 == t2
  }
}

macro_rules! impl_static_equals_for_tuple {
  ($($elem:ident),*) => {
    impl<$($elem,)*> StaticEquals for ($($elem,)*) where $($elem: StaticEquals,)* {
      fn test_static_equals(t1: &Self, t2: &Self) -> bool {
        paste::item! { let ($( [<$elem:lower 1>],)*) = t1; }
        paste::item! { let ($( [<$elem:lower 2>],)*) = t2; }
        $( paste::item! { if !StaticEquals::test_static_equals([<$elem:lower 1>], [<$elem:lower 2>]) { return false; } })*;
        return true;
      }
    }
  };
}

impl_static_equals_for_tuple!(A);
impl_static_equals_for_tuple!(A, B);
impl_static_equals_for_tuple!(A, B, C);
impl_static_equals_for_tuple!(A, B, C, D);
impl_static_equals_for_tuple!(A, B, C, D, E);
impl_static_equals_for_tuple!(A, B, C, D, E, F);
impl_static_equals_for_tuple!(A, B, C, D, E, F, G);
impl_static_equals_for_tuple!(A, B, C, D, E, F, G, H);
impl_static_equals_for_tuple!(A, B, C, D, E, F, G, H, I);
