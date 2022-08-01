use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicArgmin;

impl DynamicArgmin {
  pub fn aggregate<T: Tag>(&self, batch: DynamicElements<T>, ctx: &T::Context) -> DynamicElements<T> {
    ctx.dynamic_argmin(batch)
  }
}
