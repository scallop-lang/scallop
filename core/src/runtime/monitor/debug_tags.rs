use crate::common::tuple::Tuple;
use crate::runtime::provenance::Provenance;

use super::*;

pub struct DebugTagsMonitor;

impl<Prov: Provenance> Monitor<Prov> for DebugTagsMonitor {
  fn observe_loading_relation(&self, relation: &str) {
    println!("[Tagging Relation] {}", relation)
  }

  fn observe_tagging(&self, tup: &Tuple, input_tag: &Option<Prov::InputTag>, tag: &Prov::Tag) {
    println!("[Tagging] Tuple: {}, Input Tag: {:?} -> Tag: {:?}", tup, input_tag, tag)
  }

  fn observe_recovering_relation(&self, relation: &str) {
    println!("[Recover Relation] {}", relation)
  }

  fn observe_recover(&self, tup: &Tuple, tag: &Prov::Tag, output_tag: &Prov::OutputTag) {
    println!(
      "[Recover] Tuple: {}, Tag: {:?} -> Output Tag: {:?}",
      tup, tag, output_tag
    )
  }
}
