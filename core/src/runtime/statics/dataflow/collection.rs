use super::*;

pub fn collection<'a, Tup, Prov>(
  collection: &'a StaticCollection<Tup, Prov>,
  first_time: bool,
) -> StaticCollectionDataflow<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  StaticCollectionDataflow { collection, first_time }
}

pub struct StaticCollectionDataflow<'a, Tup: StaticTupleTrait, Prov: Provenance> {
  pub collection: &'a StaticCollection<Tup, Prov>,
  pub first_time: bool,
}

impl<'a, Tup, Prov> Clone for StaticCollectionDataflow<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  fn clone(&self) -> Self {
    Self {
      collection: self.collection,
      first_time: self.first_time,
    }
  }
}

impl<'a, Tup, Prov> Dataflow<Tup, Prov> for StaticCollectionDataflow<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Stable = SingleBatch<StaticCollectionBatch<'a, Tup, Prov>>;

  type Recent = SingleBatch<StaticCollectionBatch<'a, Tup, Prov>>;

  fn iter_recent(self) -> Self::Recent {
    if self.first_time {
      SingleBatch::singleton(StaticCollectionBatch {
        collection: self.collection,
        i: 0,
      })
    } else {
      SingleBatch::empty()
    }
  }

  fn iter_stable(&self) -> Self::Stable {
    if self.first_time {
      SingleBatch::empty()
    } else {
      SingleBatch::singleton(StaticCollectionBatch {
        collection: self.collection,
        i: 0,
      })
    }
  }
}

pub struct StaticCollectionBatch<'a, Tup: StaticTupleTrait, Prov: Provenance> {
  collection: &'a StaticCollection<Tup, Prov>,
  i: usize,
}

impl<'a, Tup: StaticTupleTrait, Prov: Provenance> Clone for StaticCollectionBatch<'a, Tup, Prov> {
  fn clone(&self) -> Self {
    Self {
      collection: self.collection,
      i: self.i,
    }
  }
}

impl<'a, Tup, Prov> Iterator for StaticCollectionBatch<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Item = StaticElement<Tup, Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    if let Some(e) = self.collection.ith(self.i) {
      self.i += 1;
      Some(e.clone())
    } else {
      None
    }
  }
}

impl<'a, Tup, Prov> Batch<Tup, Prov> for StaticCollectionBatch<'a, Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
}
