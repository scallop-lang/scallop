use itertools::Itertools;

use crate::common::tuple::*;
use crate::common::type_family::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct AvgAggregate;

impl AvgAggregate {
  pub fn new() -> Self {
    Self
  }
}

impl Into<DynamicAggregate> for AvgAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Avg(self)
  }
}

impl Aggregate for AvgAggregate {
  type Aggregator<P: Provenance> = AvgAggregator;

  fn name(&self) -> String {
    "avg".to_string()
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: vec![
        ("A".to_string(), GenericTypeFamily::possibly_empty_tuple()),
        ("T".to_string(), GenericTypeFamily::type_family(TypeFamily::Number)),
      ]
      .into_iter()
      .collect(),
      arg_type: BindingTypes::generic("A"),
      input_type: BindingTypes::generic("T"),
      output_type: BindingTypes::generic("T"),
      allow_exclamation_mark: true,
      ..Default::default()
    }
  }

  fn instantiate<P: Provenance>(&self, info: AggregateInfo) -> Self::Aggregator<P> {
    AvgAggregator {
      non_multi_world: info.has_exclamation_mark,
      num_args: info.arg_var_types.len(),
      value_type: info.input_var_types[0].clone(),
    }
  }
}

#[derive(Clone)]
pub struct AvgAggregator {
  non_multi_world: bool,
  num_args: usize,
  value_type: ValueType,
}

impl AvgAggregator {
  pub fn avg<T>(non_multi_world: bool) -> Self
  where
    ValueType: FromType<T>,
  {
    Self {
      non_multi_world,
      num_args: 0,
      value_type: ValueType::from_type(),
    }
  }
}

impl AvgAggregator {
  pub fn perform_avg<'a, I: Iterator<Item = &'a Tuple>>(&self, i: I) -> Tuple {
    if self.num_args > 0 {
      let iterator = i.map(|t| &t[self.num_args]);
      self.value_type.avg(iterator)
    } else {
      self.value_type.avg(i)
    }
  }
}

impl<P: Provenance> Aggregator<P> for AvgAggregator {
  default fn aggregate(&self, prov: &P, _env: &RuntimeEnvironment, batch: DynamicElements<P>) -> DynamicElements<P> {
    if self.non_multi_world {
      let res = self.perform_avg(batch.iter_tuples());
      vec![DynamicElement::new(res, prov.one())]
    } else {
      let mut result = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let res = self.perform_avg(chosen_set.iter().map(|i| &batch[*i].tuple));
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
