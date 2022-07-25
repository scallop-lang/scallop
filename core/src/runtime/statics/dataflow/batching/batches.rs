use super::*;

use crate::runtime::provenance::Tag;
use crate::runtime::statics::StaticTupleTrait;

pub trait Batches<Tup, T>: Iterator<Item = Self::Batch> + Clone
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  type Batch: Batch<Tup, T>;
}
