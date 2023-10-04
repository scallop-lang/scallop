use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct ExistsAggregate;

impl ExistsAggregate {
  pub fn new() -> Self {
    Self
  }
}

impl Into<DynamicAggregate> for ExistsAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Exists(self)
  }
}

impl Aggregate for ExistsAggregate {
  type Aggregator<P: Provenance> = ExistsAggregator;

  fn name(&self) -> String {
    "exists".to_string()
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: vec![("T".to_string(), GenericTypeFamily::possibly_empty_tuple())]
        .into_iter()
        .collect(),
      param_types: vec![],
      arg_type: BindingTypes::empty_tuple(),
      input_type: BindingTypes::generic("T"),
      output_type: BindingTypes::value_type(ValueType::Bool),
      allow_exclamation_mark: true,
    }
  }

  fn instantiate<P: Provenance>(
    &self,
    _params: Vec<crate::common::value::Value>,
    has_exclamation_mark: bool,
    _arg_types: Vec<ValueType>,
    _input_types: Vec<ValueType>,
  ) -> Self::Aggregator<P> {
    ExistsAggregator {
      non_multi_world: has_exclamation_mark,
    }
  }
}

#[derive(Clone)]
pub struct ExistsAggregator {
  pub non_multi_world: bool,
}

impl ExistsAggregator {
  pub fn new(non_multi_world: bool) -> Self {
    Self { non_multi_world }
  }
}

impl<Prov: Provenance> Aggregator<Prov> for ExistsAggregator {
  default fn aggregate(
    &self,
    prov: &Prov,
    _env: &RuntimeEnvironment,
    batch: DynamicElements<Prov>,
  ) -> DynamicElements<Prov> {
    if self.non_multi_world {
      vec![DynamicElement::new(!batch.is_empty(), prov.one())]
    } else {
      let mut maybe_exists_tag = None;
      let mut maybe_not_exists_tag = Some(prov.one());
      for elem in batch {
        maybe_exists_tag = match maybe_exists_tag {
          Some(exists_tag) => Some(prov.add(&exists_tag, &elem.tag)),
          None => Some(elem.tag.clone()),
        };
        maybe_not_exists_tag = match maybe_not_exists_tag {
          Some(net) => prov
            .negate(&elem.tag)
            .map(|neg_elem_tag| prov.mult(&net, &neg_elem_tag)),
          None => None,
        };
      }

      maybe_exists_tag
        .into_iter()
        .map(|t| DynamicElement::new(true, t))
        .chain(maybe_not_exists_tag.into_iter().map(|t| DynamicElement::new(false, t)))
        .collect()
    }
  }
}
