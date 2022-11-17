use super::*;

/// A vector of elements can form a dataflow
///
/// It will not be producing any stable batches. It will be producing
/// one single batch which contains the elements inside the vector.
impl<Tup, Prov> Dataflow<Tup, Prov> for Vec<StaticElement<Tup, Prov>>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Stable = EmptyBatches<std::iter::Empty<StaticElement<Tup, Prov>>>;

  type Recent = SingleBatch<std::vec::IntoIter<StaticElement<Tup, Prov>>>;

  fn iter_stable(&self) -> Self::Stable {
    Self::Stable::default()
  }

  fn iter_recent(self) -> Self::Recent {
    Self::Recent::singleton(self.into_iter())
  }
}
