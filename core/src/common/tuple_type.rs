use super::generic_tuple::GenericTuple;
use super::tuple::Tuple;
use super::value_type::{FromType, ValueType};

pub type TupleType = GenericTuple<ValueType>;

impl std::fmt::Debug for TupleType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Tuple(ts) => f.write_fmt(format_args!(
        "({})",
        ts.iter().map(|t| format!("{t:?}")).collect::<Vec<_>>().join(", ")
      )),
      Self::Value(v) => std::fmt::Debug::fmt(v, f),
    }
  }
}

impl std::fmt::Display for TupleType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Tuple(ts) => f.write_fmt(format_args!(
        "({})",
        ts.iter().map(|t| format!("{}", t)).collect::<Vec<_>>().join(", ")
      )),
      Self::Value(v) => std::fmt::Debug::fmt(v, f),
    }
  }
}

impl TupleType {
  pub fn type_of(tuple: &Tuple) -> Self {
    match tuple {
      Tuple::Tuple(t) => Self::Tuple(t.iter().map(Self::type_of).collect()),
      Tuple::Value(p) => Self::Value(ValueType::type_of(p)),
    }
  }

  pub fn matches(&self, tuple: &Tuple) -> bool {
    match (self, tuple) {
      (TupleType::Tuple(tys), Tuple::Tuple(vs)) => {
        if tys.len() != vs.len() {
          false
        } else {
          tys.iter().zip(vs.iter()).all(|(ty, v)| ty.matches(v))
        }
      }
      (TupleType::Value(ty), Tuple::Value(v)) => &ValueType::type_of(v) == ty,
      _ => false,
    }
  }

  pub fn from_types(types: &[ValueType], can_be_singleton: bool) -> Self {
    if types.len() == 1 && can_be_singleton {
      Self::Value(types[0].clone())
    } else {
      Self::Tuple(types.iter().cloned().map(Self::Value).collect())
    }
  }
}

impl<A> FromType<A> for TupleType
where
  ValueType: FromType<A>,
{
  fn from_type() -> Self {
    Self::Value(<ValueType as FromType<A>>::from_type())
  }
}

impl FromType<()> for TupleType {
  fn from_type() -> Self {
    Self::Tuple(Box::new([]))
  }
}

impl<A> FromType<(A,)> for TupleType
where
  TupleType: FromType<A>,
{
  fn from_type() -> Self {
    Self::Tuple(Box::new([<TupleType as FromType<A>>::from_type()]))
  }
}

impl<A, B> FromType<(A, B)> for TupleType
where
  TupleType: FromType<A>,
  TupleType: FromType<B>,
{
  fn from_type() -> Self {
    Self::Tuple(Box::new([
      <TupleType as FromType<A>>::from_type(),
      <TupleType as FromType<B>>::from_type(),
    ]))
  }
}

impl<A, B, C> FromType<(A, B, C)> for TupleType
where
  TupleType: FromType<A>,
  TupleType: FromType<B>,
  TupleType: FromType<C>,
{
  fn from_type() -> Self {
    Self::Tuple(Box::new([
      <TupleType as FromType<A>>::from_type(),
      <TupleType as FromType<B>>::from_type(),
      <TupleType as FromType<C>>::from_type(),
    ]))
  }
}

impl<A, B, C, D> FromType<(A, B, C, D)> for TupleType
where
  TupleType: FromType<A>,
  TupleType: FromType<B>,
  TupleType: FromType<C>,
  TupleType: FromType<D>,
{
  fn from_type() -> Self {
    Self::Tuple(Box::new([
      <TupleType as FromType<A>>::from_type(),
      <TupleType as FromType<B>>::from_type(),
      <TupleType as FromType<C>>::from_type(),
      <TupleType as FromType<D>>::from_type(),
    ]))
  }
}

impl<A, B, C, D, E> FromType<(A, B, C, D, E)> for TupleType
where
  TupleType: FromType<A>,
  TupleType: FromType<B>,
  TupleType: FromType<C>,
  TupleType: FromType<D>,
  TupleType: FromType<E>,
{
  fn from_type() -> Self {
    Self::Tuple(Box::new([
      <TupleType as FromType<A>>::from_type(),
      <TupleType as FromType<B>>::from_type(),
      <TupleType as FromType<C>>::from_type(),
      <TupleType as FromType<D>>::from_type(),
      <TupleType as FromType<E>>::from_type(),
    ]))
  }
}

impl<A, B, C, D, E, F> FromType<(A, B, C, D, E, F)> for TupleType
where
  TupleType: FromType<A>,
  TupleType: FromType<B>,
  TupleType: FromType<C>,
  TupleType: FromType<D>,
  TupleType: FromType<E>,
  TupleType: FromType<F>,
{
  fn from_type() -> Self {
    Self::Tuple(Box::new([
      <TupleType as FromType<A>>::from_type(),
      <TupleType as FromType<B>>::from_type(),
      <TupleType as FromType<C>>::from_type(),
      <TupleType as FromType<D>>::from_type(),
      <TupleType as FromType<E>>::from_type(),
      <TupleType as FromType<F>>::from_type(),
    ]))
  }
}

impl<A, B, C, D, E, F, G> FromType<(A, B, C, D, E, F, G)> for TupleType
where
  TupleType: FromType<A>,
  TupleType: FromType<B>,
  TupleType: FromType<C>,
  TupleType: FromType<D>,
  TupleType: FromType<E>,
  TupleType: FromType<F>,
  TupleType: FromType<G>,
{
  fn from_type() -> Self {
    Self::Tuple(Box::new([
      <TupleType as FromType<A>>::from_type(),
      <TupleType as FromType<B>>::from_type(),
      <TupleType as FromType<C>>::from_type(),
      <TupleType as FromType<D>>::from_type(),
      <TupleType as FromType<E>>::from_type(),
      <TupleType as FromType<F>>::from_type(),
      <TupleType as FromType<G>>::from_type(),
    ]))
  }
}
