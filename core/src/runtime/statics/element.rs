use crate::runtime::provenance::*;

use super::*;

pub type StaticElement<Tup, Tag> = Tagged<StaticTuple<Tup>, Tag>;

impl<A, B> StaticElement<A, B>
where
  A: StaticTupleTrait,
  B: Tag,
{
  pub fn new(tuple: A, tag: B) -> Self {
    Self {
      tuple: StaticTuple(tuple),
      tag,
    }
  }
}
