use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicCategoricalK(pub usize);

impl DynamicCategoricalK {
  pub fn aggregate<Prov: Provenance>(
    &self,
    batch: DynamicElements<Prov>,
    ctx: &Prov,
    rt: &RuntimeEnvironment,
  ) -> DynamicElements<Prov> {
    ctx.dynamic_categorical_k(self.0, batch, rt)
  }
}
