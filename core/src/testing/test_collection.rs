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

pub struct TestCollectionWithTag<Prov: Provenance> {
  pub elements: Vec<(OutputTagOf<Prov>, Tuple)>,
}

impl<Prov: Provenance> TestCollectionWithTag<Prov> {
  pub fn empty() -> Self {
    Self { elements: vec![] }
  }
}

impl<Prov: Provenance, Tup: Into<Tuple>> From<Vec<(OutputTagOf<Prov>, Tup)>> for TestCollectionWithTag<Prov> {
  fn from(v: Vec<(OutputTagOf<Prov>, Tup)>) -> Self {
    Self {
      elements: v.into_iter().map(|(tag, tup)| (tag, tup.into())).collect(),
    }
  }
}

pub fn test_equals(t1: &Tuple, t2: &Tuple) -> bool {
  match (t1, t2) {
    (Tuple::Tuple(ts1), Tuple::Tuple(ts2)) => ts1.iter().zip(ts2.iter()).all(|(s1, s2)| test_equals(s1, s2)),
    (Tuple::Value(Value::F32(t1)), Tuple::Value(Value::F32(t2))) => {
      if t1.is_infinite() && t1.is_sign_positive() && t2.is_infinite() && t2.is_sign_positive() {
        true
      } else if t1.is_infinite() && t1.is_sign_negative() && t2.is_infinite() && t2.is_sign_negative() {
        true
      } else if t1.is_nan() || t2.is_nan() {
        false
      } else {
        (t1 - t2).abs() < 0.001
      }
    }
    (Tuple::Value(Value::F64(t1)), Tuple::Value(Value::F64(t2))) => {
      if t1.is_infinite() && t1.is_sign_positive() && t2.is_infinite() && t2.is_sign_positive() {
        true
      } else if t1.is_infinite() && t1.is_sign_negative() && t2.is_infinite() && t2.is_sign_negative() {
        true
      } else if t1.is_nan() || t2.is_nan() {
        false
      } else {
        (t1 - t2).abs() < 0.001
      }
    }
    _ => t1 == t2,
  }
}

pub fn expect_collection<Prov, C>(actual: &DynamicCollection<Prov>, expected: C)
where
  Prov: Provenance,
  Prov::Tag: std::fmt::Debug,
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

pub fn expect_output_collection<Prov, C>(name: &str, actual: &DynamicOutputCollection<Prov>, expected: C)
where
  Prov: Provenance,
  Prov::Tag: std::fmt::Debug,
  C: Into<TestCollection>,
{
  let expected = Into::<TestCollection>::into(expected);

  // First check everything in expected is in actual
  for e in &expected.elements {
    let te = e.clone().into();
    let pos = actual.iter().position(|(_, tuple)| test_equals(&tuple, &te));
    assert!(
      pos.is_some(),
      "Tuple {:?} not found in `{}` collection {:?}",
      te,
      name,
      actual
    )
  }

  // Then check everything in actual is in expected
  for elem in &actual.elements {
    let pos = expected
      .elements
      .iter()
      .position(|e| test_equals(&e.clone().into(), &elem.1));
    assert!(
      pos.is_some(),
      "Tuple {:?} is derived in collection `{}` but not found in expected set",
      elem,
      name,
    )
  }
}

pub fn expect_output_collection_with_tag<Prov, C, F>(
  name: &str,
  actual: &DynamicOutputCollection<Prov>,
  expected: C,
  cmp: F,
) where
  Prov: Provenance,
  Prov::Tag: std::fmt::Debug,
  C: Into<TestCollectionWithTag<Prov>>,
  F: Fn(&OutputTagOf<Prov>, &OutputTagOf<Prov>) -> bool,
{
  let expected = Into::<TestCollectionWithTag<Prov>>::into(expected);

  // First check everything in expected is in actual
  for e in &expected.elements {
    let (tage, te) = e.clone();
    let pos = actual
      .iter()
      .position(|(tag, tuple)| test_equals(&tuple, &te) && cmp(&tage, tag));
    assert!(
      pos.is_some(),
      "Tagged Tuple {:?} not found in `{}` collection {:?}",
      (tage, te),
      name,
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
      "Tagged Tuple {:?} is derived in `{}` collection but not found in expected set",
      elem,
      name,
    )
  }
}

pub fn expect_static_collection<Tup, Prov>(actual: &StaticCollection<Tup, Prov>, expected: Vec<Tup>)
where
  Tup: StaticTupleTrait + StaticEquals,
  Prov: Provenance,
  Prov::Tag: std::fmt::Debug,
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

pub fn expect_static_output_collection<Tup, Prov>(actual: &StaticOutputCollection<Tup, Prov>, expected: Vec<Tup>)
where
  Tup: StaticTupleTrait + StaticEquals,
  Prov: Provenance,
  Prov::Tag: std::fmt::Debug,
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
    if t1.is_infinite() && t1.is_sign_positive() && t2.is_infinite() && t2.is_sign_positive() {
      true
    } else if t1.is_infinite() && t1.is_sign_negative() && t2.is_infinite() && t2.is_sign_negative() {
      true
    } else if t1.is_nan() || t2.is_nan() {
      false
    } else {
      (t1 - t2).abs() < 0.001
    }
  }
}

impl StaticEquals for f64 {
  fn test_static_equals(t1: &Self, t2: &Self) -> bool {
    if t1.is_infinite() && t1.is_sign_positive() && t2.is_infinite() && t2.is_sign_positive() {
      true
    } else if t1.is_infinite() && t1.is_sign_negative() && t2.is_infinite() && t2.is_sign_negative() {
      true
    } else if t1.is_nan() || t2.is_nan() {
      false
    } else {
      (t1 - t2).abs() < 0.001
    }
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
