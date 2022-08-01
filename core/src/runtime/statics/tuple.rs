pub trait StaticTupleTrait: 'static + Sized + Clone + std::fmt::Debug + std::cmp::PartialOrd {}

impl<T> StaticTupleTrait for T where T: 'static + Sized + Clone + std::fmt::Debug + std::cmp::PartialOrd {}

#[derive(Clone, PartialEq, PartialOrd)]
pub struct StaticTupleWrapper<T: StaticTupleTrait>(T);

impl<T: StaticTupleTrait> StaticTupleWrapper<T> {
  pub fn new(t: T) -> Self {
    Self(t)
  }

  pub fn get(&self) -> &T {
    &self.0
  }

  pub fn into(self) -> T {
    self.0
  }
}

impl<T: StaticTupleTrait> std::cmp::Eq for StaticTupleWrapper<T> {}

impl<T: StaticTupleTrait> std::cmp::Ord for StaticTupleWrapper<T> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    match self.0.partial_cmp(&other.0) {
      Some(ord) => ord,
      None => panic!("[Internal Error] Unable to find ordering"),
    }
  }
}

impl<T> std::ops::Deref for StaticTupleWrapper<T>
where
  T: StaticTupleTrait,
{
  type Target = T;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl<T: StaticTupleTrait> std::fmt::Debug for StaticTupleWrapper<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self.0, f)
  }
}

impl<T: StaticTupleTrait> std::fmt::Display for StaticTupleWrapper<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self.0, f)
  }
}
