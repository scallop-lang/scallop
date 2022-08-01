use crate::common::value_type::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicProd(pub ValueType);

impl DynamicProd {
  pub fn aggregate<T: Tag>(&self, batch: DynamicElements<T>, ctx: &T::Context) -> DynamicElements<T> {
    ctx.dynamic_prod(&self.0, batch)
  }
}
