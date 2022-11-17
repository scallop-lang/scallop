use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicTopK(pub usize);

impl DynamicTopK {
  pub fn aggregate<Prov: Provenance>(&self, batch: DynamicElements<Prov>, ctx: &Prov) -> DynamicElements<Prov> {
    ctx.dynamic_top_k(self.0, batch)
  }
}
