use super::*;

pub fn collection<'a, Tup, T>(
  collection: &'a StaticCollection<Tup, T>,
  first_time: bool,
) -> StaticCollectionDataflow<'a, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  StaticCollectionDataflow { collection, first_time }
}

pub struct StaticCollectionDataflow<'a, Tup: StaticTupleTrait, T: Tag> {
  pub collection: &'a StaticCollection<Tup, T>,
  pub first_time: bool,
}

impl<'a, Tup, T> Clone for StaticCollectionDataflow<'a, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  fn clone(&self) -> Self {
    Self {
      collection: self.collection,
      first_time: self.first_time,
    }
  }
}

impl<'a, Tup, T> Dataflow<Tup, T> for StaticCollectionDataflow<'a, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  type Stable = SingleBatch<StaticCollectionBatch<'a, Tup, T>>;

  type Recent = SingleBatch<StaticCollectionBatch<'a, Tup, T>>;

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

pub struct StaticCollectionBatch<'a, Tup: StaticTupleTrait, T: Tag> {
  collection: &'a StaticCollection<Tup, T>,
  i: usize,
}

impl<'a, Tup: StaticTupleTrait, T: Tag> Clone for StaticCollectionBatch<'a, Tup, T> {
  fn clone(&self) -> Self {
    Self {
      collection: self.collection,
      i: self.i,
    }
  }
}

impl<'a, Tup, T> Iterator for StaticCollectionBatch<'a, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  type Item = StaticElement<Tup, T>;

  fn next(&mut self) -> Option<Self::Item> {
    if let Some(e) = self.collection.ith(self.i) {
      self.i += 1;
      Some(e.clone())
    } else {
      None
    }
  }
}

impl<'a, Tup, T> Batch<Tup, T> for StaticCollectionBatch<'a, Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
}
