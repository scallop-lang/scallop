use crate::runtime::provenance::Tag;
use crate::runtime::statics::*;

pub trait Batch<Tup, T>: Iterator<Item = StaticElement<Tup, T>> + Clone
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  /// Step u steps
  fn step(&mut self, u: usize) {
    for _ in 0..u {
      self.next();
    }
  }

  /// Search until the given comparison function returns true on a given element
  fn search_ahead<F>(&mut self, _: F) -> Option<StaticElement<Tup, T>>
  where
    F: FnMut(&Tup) -> bool,
  {
    self.next()
  }
}

impl<Tup, T> Batch<Tup, T> for std::iter::Empty<StaticElement<Tup, T>>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
}

impl<Tup, T> Batch<Tup, T> for std::vec::IntoIter<StaticElement<Tup, T>>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
}
