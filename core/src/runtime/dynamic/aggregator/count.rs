use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicCount {
  pub discrete: bool,
}

impl DynamicCount {
  pub fn aggregate<Prov: Provenance>(&self, batch: DynamicElements<Prov>, ctx: &Prov) -> DynamicElements<Prov> {
    if self.discrete {
      ctx.dynamic_discrete_count(batch)
    } else {
      ctx.dynamic_count(batch)
    }
  }
}
