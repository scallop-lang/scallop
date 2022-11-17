use crate::common::tuple::*;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub struct EDBFact<Prov: Provenance> {
  pub tag: Option<Prov::InputTag>,
  pub tuple: Tuple,
}

impl<Prov: Provenance> EDBFact<Prov> {
  pub fn new(tag: Option<Prov::InputTag>, tuple: Tuple) -> Self {
    Self { tag, tuple }
  }
}
