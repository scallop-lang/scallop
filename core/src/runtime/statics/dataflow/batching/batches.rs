use super::*;

use crate::runtime::provenance::*;
use crate::runtime::statics::*;

pub trait Batches<Tup, Prov>: Iterator<Item = Self::Batch> + Clone
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Batch: Batch<Tup, Prov>;
}
