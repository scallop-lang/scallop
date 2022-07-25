use super::*;

/// A vector of elements can form a dataflow
///
/// It will not be producing any stable batches. It will be producing
/// one single batch which contains the elements inside the vector.
impl<Tup, T> Dataflow<Tup, T> for Vec<StaticElement<Tup, T>>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  type Stable = EmptyBatches<std::iter::Empty<StaticElement<Tup, T>>>;

  type Recent = SingleBatch<std::vec::IntoIter<StaticElement<Tup, T>>>;

  fn iter_stable(&self) -> Self::Stable {
    Self::Stable::default()
  }

  fn iter_recent(self) -> Self::Recent {
    Self::Recent::singleton(self.into_iter())
  }
}
