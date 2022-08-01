use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicMax;

impl DynamicMax {
  pub fn aggregate<T: Tag>(&self, batch: DynamicElements<T>, ctx: &T::Context) -> DynamicElements<T> {
    ctx.dynamic_max(batch)
  }
}
