use crate::common::tuple::Tuple;
use crate::runtime::provenance::ProvenanceContext;

use super::*;

pub struct DebugTagsMonitor;

impl<C: ProvenanceContext> Monitor<C> for DebugTagsMonitor {
  fn observe_loading_relation(&self, relation: &str) {
    println!("[Tagging Relation] {}", relation)
  }

  fn observe_tagging(&self, tup: &Tuple, input_tag: &Option<C::InputTag>, tag: &C::Tag) {
    println!(
      "[Tagging] Tuple: {}, Input Tag: {:?} -> Tag: {:?}",
      tup, input_tag, tag
    )
  }

  fn observe_recovering_relation(&self, relation: &str) {
    println!("[Recover Relation] {}", relation)
  }

  fn observe_recover(&self, tup: &Tuple, tag: &C::Tag, output_tag: &C::OutputTag) {
    println!(
      "[Recover] Tuple: {}, Tag: {:?} -> Output Tag: {:?}",
      tup, tag, output_tag
    )
  }
}
