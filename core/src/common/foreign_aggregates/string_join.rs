use itertools::Itertools;

use crate::common::tuple::*;
use crate::common::value::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct StringJoinAggregate;

impl StringJoinAggregate {
  pub fn new() -> Self {
    Self
  }
}

impl Into<DynamicAggregate> for StringJoinAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::StringJoin(self)
  }
}

impl Aggregate for StringJoinAggregate {
  type Aggregator<P: Provenance> = StringJoinAggregator;

  fn name(&self) -> String {
    "string_join".to_string()
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: vec![
        ("A".to_string(), GenericTypeFamily::possibly_empty_tuple()),
      ]
      .into_iter()
      .collect(),
      param_types: vec![ParamType::Optional(ValueType::String)],
      arg_type: BindingTypes::generic("A"),
      input_type: BindingTypes::value_type(ValueType::String),
      output_type: BindingTypes::value_type(ValueType::String),
      allow_exclamation_mark: true,
    }
  }

  fn instantiate<P: Provenance>(
    &self,
    params: Vec<Value>,
    has_exclamation_mark: bool,
    arg_types: Vec<ValueType>,
    _input_types: Vec<ValueType>,
  ) -> Self::Aggregator<P> {
    StringJoinAggregator {
      non_multi_world: has_exclamation_mark,
      num_args: arg_types.len(),
      separator: params.get(0).map(|v| v.as_str().to_string()).unwrap_or("".to_string()),
    }
  }
}

#[derive(Clone)]
pub struct StringJoinAggregator {
  non_multi_world: bool,
  num_args: usize,
  separator: String,
}

impl StringJoinAggregator {
  pub fn perform_string_join<'a, I: Iterator<Item = &'a Tuple>>(&self, i: I) -> Tuple {
    let strings: Vec<_> = if self.num_args > 0 {
      i.map(|t| t[self.num_args].as_string()).collect()
    } else {
      i.map(|t| t.as_string()).collect()
    };
    strings.join(&self.separator).into()
  }
}

impl<P: Provenance> Aggregator<P> for StringJoinAggregator {
  default fn aggregate(&self, prov: &P, _env: &RuntimeEnvironment, batch: DynamicElements<P>) -> DynamicElements<P> {
    if self.non_multi_world {
      let res = self.perform_string_join(batch.iter_tuples());
      vec![DynamicElement::new(res, prov.one())]
    } else {
      let mut result = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let res = self.perform_string_join(chosen_set.iter().map(|i| &batch[*i].tuple));
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
          result.push(DynamicElement::new(res, tag));
        }
      }
      result
    }
  }
}
