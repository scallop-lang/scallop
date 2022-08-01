use crate::common::value_type::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicSum(pub ValueType);

impl DynamicSum {
  pub fn aggregate<T: Tag>(&self, batch: DynamicElements<T>, ctx: &T::Context) -> DynamicElements<T> {
    ctx.dynamic_sum(&self.0, batch)
  }
}
