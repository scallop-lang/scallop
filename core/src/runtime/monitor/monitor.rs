use crate::common::tuple::Tuple;
use crate::runtime::provenance::*;

pub trait Monitor<Prov: Provenance> {
  /// Observe loading a relation
  fn observe_loading_relation(&self, _relation: &str) {}

  /// Observe a call on tagging function
  fn observe_tagging(&self, _tup: &Tuple, _input_tag: &Option<Prov::InputTag>, _tag: &Prov::Tag) {}

  /// Observe recovering output tags of a relation
  fn observe_recovering_relation(&self, _relation: &str) {}

  /// Observe a call on recover function
  fn observe_recover(&self, _tup: &Tuple, _tag: &Prov::Tag, _output_tag: &Prov::OutputTag) {}
}

impl<Prov: Provenance> Monitor<Prov> for () {}

impl<M1, Prov> Monitor<Prov> for (M1,)
where
  M1: Monitor<Prov>,
  Prov: Provenance,
{
  fn observe_loading_relation(&self, relation: &str) {
    self.0.observe_loading_relation(relation)
  }

  fn observe_tagging(&self, tup: &Tuple, input_tag: &Option<Prov::InputTag>, tag: &Prov::Tag) {
    self.0.observe_tagging(tup, input_tag, tag)
  }

  fn observe_recovering_relation(&self, relation: &str) {
    self.0.observe_recovering_relation(relation)
  }

  fn observe_recover(&self, tup: &Tuple, tag: &Prov::Tag, output_tag: &Prov::OutputTag) {
    self.0.observe_recover(tup, tag, output_tag)
  }
}

impl<M1, M2, Prov> Monitor<Prov> for (M1, M2)
where
  M1: Monitor<Prov>,
  M2: Monitor<Prov>,
  Prov: Provenance,
{
  fn observe_loading_relation(&self, relation: &str) {
    self.0.observe_loading_relation(relation);
    self.1.observe_loading_relation(relation);
  }

  fn observe_tagging(&self, tup: &Tuple, input_tag: &Option<Prov::InputTag>, tag: &Prov::Tag) {
    self.0.observe_tagging(tup, input_tag, tag);
    self.1.observe_tagging(tup, input_tag, tag);
  }

  fn observe_recovering_relation(&self, relation: &str) {
    self.0.observe_recovering_relation(relation);
    self.1.observe_recovering_relation(relation);
  }

  fn observe_recover(&self, tup: &Tuple, tag: &Prov::Tag, output_tag: &Prov::OutputTag) {
    self.0.observe_recover(tup, tag, output_tag);
    self.1.observe_recover(tup, tag, output_tag);
  }
}

impl<M1, M2, M3, Prov> Monitor<Prov> for (M1, M2, M3)
where
  M1: Monitor<Prov>,
  M2: Monitor<Prov>,
  M3: Monitor<Prov>,
  Prov: Provenance,
{
  fn observe_loading_relation(&self, relation: &str) {
    self.0.observe_loading_relation(relation);
    self.1.observe_loading_relation(relation);
    self.2.observe_loading_relation(relation);
  }

  fn observe_tagging(&self, tup: &Tuple, input_tag: &Option<Prov::InputTag>, tag: &Prov::Tag) {
    self.0.observe_tagging(tup, input_tag, tag);
    self.1.observe_tagging(tup, input_tag, tag);
    self.2.observe_tagging(tup, input_tag, tag);
  }

  fn observe_recovering_relation(&self, relation: &str) {
    self.0.observe_recovering_relation(relation);
    self.1.observe_recovering_relation(relation);
    self.2.observe_recovering_relation(relation);
  }

  fn observe_recover(&self, tup: &Tuple, tag: &Prov::Tag, output_tag: &Prov::OutputTag) {
    self.0.observe_recover(tup, tag, output_tag);
    self.1.observe_recover(tup, tag, output_tag);
    self.2.observe_recover(tup, tag, output_tag);
  }
}
