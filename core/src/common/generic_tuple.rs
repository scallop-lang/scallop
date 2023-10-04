use super::tuple_access::TupleAccessor;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GenericTuple<T> {
  Value(T),
  Tuple(Box<[GenericTuple<T>]>),
}

impl<T> GenericTuple<T> {
  pub fn unit() -> Self {
    Self::Tuple(Box::new([]))
  }

  pub fn empty() -> Self {
    Self::Tuple(Box::new([]))
  }

  pub fn is_empty(&self) -> bool {
    if let Self::Tuple(ts) = self {
      ts.is_empty()
    } else {
      false
    }
  }

  /// Return the size of the tuple; if is a value, then `None` is returned
  pub fn len(&self) -> Option<usize> {
    match self {
      Self::Value(_) => None,
      Self::Tuple(ts) => Some(ts.len()),
    }
  }

  pub fn get_value(&self) -> Option<&T> {
    match self {
      Self::Value(v) => Some(v),
      Self::Tuple(_) => None,
    }
  }

  pub fn drain(self, i: usize) -> Option<GenericTuple<T>> {
    match self {
      Self::Value(_) => None,
      Self::Tuple(vs) => vs.into_vec().into_iter().nth(i),
    }
  }
}

impl<T> std::ops::Index<usize> for GenericTuple<T> {
  type Output = Self;

  fn index(&self, i: usize) -> &Self::Output {
    match self {
      Self::Tuple(t) => &t[i],
      _ => panic!("Cannot access tuple value with `{:?}`", i),
    }
  }
}

impl<T> std::ops::Index<std::ops::Range<usize>> for GenericTuple<T> {
  type Output = [Self];

  fn index(&self, range: std::ops::Range<usize>) -> &Self::Output {
    match self {
      Self::Tuple(t) => &t[range],
      _ => panic!("Cannot access tuple value with `{:?}`", range),
    }
  }
}

impl<T> std::ops::Index<std::ops::RangeTo<usize>> for GenericTuple<T> {
  type Output = [Self];

  fn index(&self, range: std::ops::RangeTo<usize>) -> &Self::Output {
    match self {
      Self::Tuple(t) => &t[range],
      _ => panic!("Cannot access tuple value with `{:?}`", range),
    }
  }
}

impl<T> std::ops::Index<std::ops::RangeFrom<usize>> for GenericTuple<T> {
  type Output = [Self];

  fn index(&self, range: std::ops::RangeFrom<usize>) -> &Self::Output {
    match self {
      Self::Tuple(t) => &t[range],
      _ => panic!("Cannot access tuple value with `{:?}`", range),
    }
  }
}

impl<T> std::ops::Index<std::ops::RangeInclusive<usize>> for GenericTuple<T> {
  type Output = [Self];

  fn index(&self, range: std::ops::RangeInclusive<usize>) -> &Self::Output {
    match self {
      Self::Tuple(t) => &t[range],
      _ => panic!("Cannot access tuple value with `{:?}`", range),
    }
  }
}

impl<T> std::ops::Index<std::ops::RangeFull> for GenericTuple<T> {
  type Output = [Self];

  fn index(&self, range: std::ops::RangeFull) -> &Self::Output {
    match self {
      Self::Tuple(t) => &t[range],
      _ => panic!("Cannot access tuple value with `{:?}`", range),
    }
  }
}

impl<T> std::ops::Index<&TupleAccessor> for GenericTuple<T> {
  type Output = GenericTuple<T>;

  fn index(&self, acc: &TupleAccessor) -> &Self {
    match (self, acc.len) {
      (_, 0) => self,
      (Self::Tuple(t), _) => &t[acc.indices[0] as usize][&acc.shift()],
      _ => panic!("Cannot access tuple with `{:?}`", acc),
    }
  }
}

impl<T> std::iter::FromIterator<T> for GenericTuple<T> {
  fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
    Self::Tuple(iter.into_iter().map(|v| Self::Value(v)).collect())
  }
}

impl<T> From<()> for GenericTuple<T> {
  fn from(_: ()) -> Self {
    Self::Tuple(Box::new([]))
  }
}

impl<T, A> From<(A,)> for GenericTuple<T>
where
  A: Into<GenericTuple<T>>,
{
  fn from((a,): (A,)) -> Self {
    Self::Tuple(Box::new([a.into()]))
  }
}

impl<T, A, B> From<(A, B)> for GenericTuple<T>
where
  A: Into<GenericTuple<T>>,
  B: Into<GenericTuple<T>>,
{
  fn from((a, b): (A, B)) -> Self {
    Self::Tuple(Box::new([a.into(), b.into()]))
  }
}

impl<T, A, B, C> From<(A, B, C)> for GenericTuple<T>
where
  A: Into<GenericTuple<T>>,
  B: Into<GenericTuple<T>>,
  C: Into<GenericTuple<T>>,
{
  fn from((a, b, c): (A, B, C)) -> Self {
    Self::Tuple(Box::new([a.into(), b.into(), c.into()]))
  }
}

impl<T, A, B, C, D> From<(A, B, C, D)> for GenericTuple<T>
where
  A: Into<GenericTuple<T>>,
  B: Into<GenericTuple<T>>,
  C: Into<GenericTuple<T>>,
  D: Into<GenericTuple<T>>,
{
  fn from((a, b, c, d): (A, B, C, D)) -> Self {
    Self::Tuple(Box::new([a.into(), b.into(), c.into(), d.into()]))
  }
}

impl<T, A, B, C, D, E> From<(A, B, C, D, E)> for GenericTuple<T>
where
  A: Into<GenericTuple<T>>,
  B: Into<GenericTuple<T>>,
  C: Into<GenericTuple<T>>,
  D: Into<GenericTuple<T>>,
  E: Into<GenericTuple<T>>,
{
  fn from((a, b, c, d, e): (A, B, C, D, E)) -> Self {
    Self::Tuple(Box::new([a.into(), b.into(), c.into(), d.into(), e.into()]))
  }
}

impl<T, A, B, C, D, E, F> From<(A, B, C, D, E, F)> for GenericTuple<T>
where
  A: Into<GenericTuple<T>>,
  B: Into<GenericTuple<T>>,
  C: Into<GenericTuple<T>>,
  D: Into<GenericTuple<T>>,
  E: Into<GenericTuple<T>>,
  F: Into<GenericTuple<T>>,
{
  fn from((a, b, c, d, e, f): (A, B, C, D, E, F)) -> Self {
    Self::Tuple(Box::new([a.into(), b.into(), c.into(), d.into(), e.into(), f.into()]))
  }
}

impl<T, A, B, C, D, E, F, G> From<(A, B, C, D, E, F, G)> for GenericTuple<T>
where
  A: Into<GenericTuple<T>>,
  B: Into<GenericTuple<T>>,
  C: Into<GenericTuple<T>>,
  D: Into<GenericTuple<T>>,
  E: Into<GenericTuple<T>>,
  F: Into<GenericTuple<T>>,
  G: Into<GenericTuple<T>>,
{
  fn from((a, b, c, d, e, f, g): (A, B, C, D, E, F, G)) -> Self {
    Self::Tuple(Box::new([
      a.into(),
      b.into(),
      c.into(),
      d.into(),
      e.into(),
      f.into(),
      g.into(),
    ]))
  }
}
