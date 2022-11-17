use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicMax;

impl DynamicMax {
  pub fn aggregate<Prov: Provenance>(&self, batch: DynamicElements<Prov>, ctx: &Prov) -> DynamicElements<Prov> {
    ctx.dynamic_max(batch)
  }
}
