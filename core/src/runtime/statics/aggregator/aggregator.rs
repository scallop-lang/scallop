use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub trait Aggregator<Tup: StaticTupleTrait, T: Tag>: Clone {
  type Output: StaticTupleTrait;

  fn aggregate(&self, tuples: StaticElements<Tup, T>, ctx: &T::Context) -> StaticElements<Self::Output, T>;
}
