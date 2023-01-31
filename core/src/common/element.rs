//! # Element

use crate::runtime::provenance::*;

pub trait Element<Prov: Provenance> {
  fn tag(&self) -> &Prov::Tag;
}
