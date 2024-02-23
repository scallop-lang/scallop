use crate::common::input_tag::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct DisjunctAggregate;

impl DisjunctAggregate {
  pub fn new() -> Self {
    Self
  }
}

impl Into<DynamicAggregate> for DisjunctAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Disjunct(self)
  }
}

impl Aggregate for DisjunctAggregate {
  type Aggregator<P: Provenance> = DisjunctAggregator;

  fn name(&self) -> String {
    "disjunct".to_string()
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: vec![
        ("T1".to_string(), GenericTypeFamily::possibly_empty_tuple()),
        ("T2".to_string(), GenericTypeFamily::non_empty_tuple()),
      ]
      .into_iter()
      .collect(),
      arg_type: BindingTypes::generic("T1"),
      input_type: BindingTypes::generic("T2"),
      output_type: BindingTypes::tuple(vec![BindingType::generic("T1"), BindingType::generic("T2")]),
      ..Default::default()
    }
  }

  fn instantiate<P: Provenance>(&self, info: AggregateInfo) -> Self::Aggregator<P> {
    DisjunctAggregator {
      num_args: info.arg_var_types.len(),
    }
  }
}

#[derive(Clone)]
pub struct DisjunctAggregator {
  pub num_args: usize,
}

impl<Prov: Provenance> Aggregator<Prov> for DisjunctAggregator {
  default fn aggregate(
    &self,
    p: &Prov,
    env: &RuntimeEnvironment,
    batch: DynamicElements<Prov>,
  ) -> DynamicElements<Prov> {
    if batch.is_empty() {
      vec![]
    } else if self.num_args == 0 {
      let exclusion_id = env.allocate_new_exclusion_id();
      batch
        .into_iter()
        .map(|Tagged { tuple, tag }| {
          // Create new tag
          let exc_dyn_input_tag = DynamicInputTag::Exclusive(exclusion_id);
          let exc_sta_input_tag = <Prov::InputTag as StaticInputTag>::from_dynamic_input_tag(&exc_dyn_input_tag);
          let exc_tag = p.tagging_optional_fn(exc_sta_input_tag);
          let new_tag = p.mult(&tag, &exc_tag);

          // Create new element
          DynamicElement::new(tuple.clone(), new_tag)
        })
        .collect()
    } else {
      let mut result = vec![];
      let mut curr_args = &batch[0].tuple[..self.num_args];
      let mut curr_exclusion = env.allocate_new_exclusion_id();
      for Tagged { tuple, tag } in &batch {
        // Checking if we need to update the exclusion id
        if &tuple[..self.num_args] != curr_args {
          curr_args = &tuple[..self.num_args];
          curr_exclusion = env.allocate_new_exclusion_id();
        }

        // Create new tag
        let exc_dyn_input_tag = DynamicInputTag::Exclusive(curr_exclusion);
        let exc_sta_input_tag = <Prov::InputTag as StaticInputTag>::from_dynamic_input_tag(&exc_dyn_input_tag);
        let exc_tag = p.tagging_optional_fn(exc_sta_input_tag);
        let new_tag = p.mult(&tag, &exc_tag);

        // Create new element
        let new_elem = DynamicElement::new(tuple.clone(), new_tag);
        result.push(new_elem);
      }
      result
    }
  }
}
