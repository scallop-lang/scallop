use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub trait Aggregator<Tup: StaticTupleTrait, Prov: Provenance>: Clone {
  type Output: StaticTupleTrait;

  fn aggregate(&self, tuples: StaticElements<Tup, Prov>, ctx: &Prov) -> StaticElements<Self::Output, Prov>;
}
