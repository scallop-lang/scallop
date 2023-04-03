use crate::common::foreign_predicate::*;
use crate::common::input_tag::*;
use crate::common::value::*;
use crate::common::tuple::*;
use crate::runtime::provenance::*;
use crate::runtime::env::*;

use super::*;

#[derive(Clone, Debug)]
pub struct ForeignPredicateGroundDataflow<'a, Prov: Provenance> {
  /// The foreign predicate
  pub foreign_predicate: String,

  /// The already bounded constants (in order to make this Dataflow free)
  pub bounded_constants: Vec<Value>,

  /// Whether this Dataflow is running on first iteration
  pub first_iteration: bool,

  /// Provenance context
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> ForeignPredicateGroundDataflow<'a, Prov> {
  /// Generate a batch from the foreign predicate
  fn generate_batch(&self, runtime: &RuntimeEnvironment) -> DynamicBatch<'a, Prov> {
    // Fetch the foreign predicate
    let foreign_predicate = runtime
      .predicate_registry
      .get(&self.foreign_predicate)
      .expect("Foreign predicate not found");

    // Evaluate the foreign predicate
    let elements = foreign_predicate
      .evaluate(&self.bounded_constants)
      .into_iter()
      .map(|(input_tag, values)| {
        let input_tag = StaticInputTag::from_dynamic_input_tag(&input_tag);
        let tag = self.ctx.tagging_optional_fn(input_tag);
        let tuple = Tuple::from(values);
        DynamicElement::new(tuple, tag)
      })
      .collect::<Vec<_>>();
    DynamicBatch::source_vec(elements)
  }
}

impl<'a, Prov: Provenance> ForeignPredicateGroundDataflow<'a, Prov> {
  pub fn iter_stable(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    if self.first_iteration {
      DynamicBatches::empty()
    } else {
      DynamicBatches::single(self.generate_batch(runtime))
    }
  }

  pub fn iter_recent(&self, runtime: &'a RuntimeEnvironment) -> DynamicBatches<'a, Prov> {
    if self.first_iteration {
      DynamicBatches::single(self.generate_batch(runtime))
    } else {
      DynamicBatches::empty()
    }
  }
}
