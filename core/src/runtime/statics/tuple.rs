pub trait StaticTupleTrait: Sized + Clone + std::fmt::Debug + std::cmp::PartialOrd {}

impl<T> StaticTupleTrait for T where T: Sized + Clone + std::fmt::Debug + std::cmp::PartialOrd {}

#[derive(Clone, PartialEq, PartialOrd)]
pub struct StaticTuple<T: StaticTupleTrait>(pub T);

impl<T: StaticTupleTrait> std::cmp::Eq for StaticTuple<T> {}

impl<T: StaticTupleTrait> std::cmp::Ord for StaticTuple<T> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    match self.0.partial_cmp(&other.0) {
      Some(ord) => ord,
      None => panic!("[Internal Error] Unable to find ordering"),
    }
  }
}

impl<T: StaticTupleTrait> std::fmt::Debug for StaticTuple<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self.0, f)
  }
}

impl<T: StaticTupleTrait> std::fmt::Display for StaticTuple<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self.0, f)
  }
}
