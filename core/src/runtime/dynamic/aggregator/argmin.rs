use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicArgmin;

impl DynamicArgmin {
  pub fn aggregate<Prov: Provenance>(&self, batch: DynamicElements<Prov>, ctx: &Prov) -> DynamicElements<Prov> {
    ctx.dynamic_argmin(batch)
  }
}
