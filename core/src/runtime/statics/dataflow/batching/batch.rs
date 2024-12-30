use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub trait Batch<Tup, Prov>: Iterator<Item = StaticElement<Tup, Prov>> + Clone
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  /// Step u steps
  fn step(&mut self, u: usize) {
    for _ in 0..u {
      self.next();
    }
  }

  /// Search until the given comparison function returns true on a given element
  fn search_ahead<F>(&mut self, _: F) -> Option<StaticElement<Tup, Prov>>
  where
    F: FnMut(&Tup) -> bool,
  {
    self.next()
  }

  fn collect_vec(&mut self) -> Vec<StaticElement<Tup, Prov>> {
    let mut result = vec![];
    while let Some(elem) = self.next() {
      result.push(elem);
    }
    result
  }
}

impl<Tup, Prov> Batch<Tup, Prov> for std::iter::Empty<StaticElement<Tup, Prov>>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
}

impl<Tup, Prov> Batch<Tup, Prov> for std::iter::Once<StaticElement<Tup, Prov>>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
}

impl<Tup, Prov> Batch<Tup, Prov> for std::vec::IntoIter<StaticElement<Tup, Prov>>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
}
