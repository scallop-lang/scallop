use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicUnique;

impl DynamicUnique {
  pub fn aggregate<T: Tag>(&self, batch: DynamicElements<T>, ctx: &T::Context) -> DynamicElements<T> {
    ctx.dynamic_unique(batch)
  }
}
