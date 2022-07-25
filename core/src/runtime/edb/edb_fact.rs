use crate::runtime::provenance::*;
use crate::common::tuple::*;

#[derive(Clone)]
pub struct EDBFact<C: ProvenanceContext> {
  pub tag: Option<C::InputTag>,
  pub tuple: Tuple,
}

impl<C: ProvenanceContext> EDBFact<C> {
  pub fn new(tag: Option<C::InputTag>, tuple: Tuple) -> Self {
    Self { tag, tuple }
  }
}
