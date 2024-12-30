use std::collections::*;

use crate::common::tuple::*;

use super::*;

pub struct DynamicHashMapCollection<Prov: Provenance> {
  pub elements: HashMap<Tuple, DynamicElement<Prov>>,
}
