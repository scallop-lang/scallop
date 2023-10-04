use itertools::Itertools;

use crate::common::value::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct CountAggregate;

impl Into<DynamicAggregate> for CountAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Count(self)
  }
}

impl Aggregate for CountAggregate {
  type Aggregator<P: Provenance> = CountAggregator;

  fn name(&self) -> String {
    "count".to_string()
  }

  /// `{T: non-empty-tuple} ==> usize := count(T)`
  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: std::iter::once(("T".to_string(), GenericTypeFamily::non_empty_tuple())).collect(),
      param_types: vec![],
      arg_type: BindingTypes::unit(),
      input_type: BindingTypes::generic("T"),
      output_type: BindingTypes::value_type(ValueType::USize),
      allow_exclamation_mark: true,
    }
  }

  fn instantiate<P: Provenance>(
    &self,
    _params: Vec<Value>,
    has_exlamation_mark: bool,
    _arg_types: Vec<ValueType>,
    _input_types: Vec<ValueType>,
  ) -> Self::Aggregator<P> {
    CountAggregator {
      non_multi_world: has_exlamation_mark,
    }
  }
}

#[derive(Clone)]
pub struct CountAggregator {
  pub non_multi_world: bool,
}

impl CountAggregator {
  pub fn new(non_multi_world: bool) -> Self {
    Self { non_multi_world }
  }
}

impl<Prov: Provenance> Aggregator<Prov> for CountAggregator {
  default fn aggregate(
    &self,
    prov: &Prov,
    _env: &RuntimeEnvironment,
    batch: DynamicElements<Prov>,
  ) -> DynamicElements<Prov> {
    if self.non_multi_world {
      vec![DynamicElement::new(batch.len(), prov.one())]
    } else {
      let mut result = vec![];
      if batch.is_empty() {
        result.push(DynamicElement::new(0usize, prov.one()));
      } else {
        for chosen_set in (0..batch.len()).powerset() {
          let count = chosen_set.len();
          let maybe_tag = batch.iter().enumerate().fold(Some(prov.one()), |maybe_acc, (i, elem)| {
            maybe_acc.and_then(|acc| {
              if chosen_set.contains(&i) {
                Some(prov.mult(&acc, &elem.tag))
              } else {
                prov.negate(&elem.tag).map(|neg_tag| prov.mult(&acc, &neg_tag))
              }
            })
          });
          if let Some(tag) = maybe_tag {
            result.push(DynamicElement::new(count, tag));
          }
        }
      }
      result
    }
  }
}
