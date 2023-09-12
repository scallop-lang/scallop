use crate::common::foreign_predicate::*;
use crate::common::tuple::*;
use crate::common::value::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct ForeignPredicateGroundDataflow<'a, Prov: Provenance> {
  /// The foreign predicate
  pub foreign_predicate: String,

  /// The already bounded constants (in order to make this Dataflow free)
  pub bounded_constants: Vec<Value>,

  /// Whether this Dataflow is running on first iteration
  pub first_iteration: bool,

  /// Provenance context
  pub ctx: &'a Prov,

  /// Runtime environment
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> ForeignPredicateGroundDataflow<'a, Prov> {
  /// Generate a batch from the foreign predicate
  fn generate_batch(&self, runtime: &RuntimeEnvironment) -> ElementsBatch<Prov> {
    // Fetch the foreign predicate
    let foreign_predicate = runtime
      .predicate_registry
      .get(&self.foreign_predicate)
      .expect("Foreign predicate not found");

    // Evaluate the foreign predicate
    let elements = foreign_predicate
      .evaluate_with_env(runtime, &self.bounded_constants)
      .into_iter()
      .filter_map(|(input_tag, values)| {
        let input_tag = StaticInputTag::from_dynamic_input_tag(&input_tag);
        let tag = self.ctx.tagging_optional_fn(input_tag);
        let tuple = Tuple::from(values);
        let internal_tuple = runtime.internalize_tuple(&tuple)?;
        Some(DynamicElement::new(internal_tuple, tag))
      })
      .collect::<Vec<_>>();

    ElementsBatch::new(elements)
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for ForeignPredicateGroundDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    if self.first_iteration {
      DynamicBatches::empty()
    } else {
      DynamicBatches::single(self.generate_batch(self.runtime))
    }
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    if self.first_iteration {
      DynamicBatches::single(self.generate_batch(self.runtime))
    } else {
      DynamicBatches::empty()
    }
  }
}
