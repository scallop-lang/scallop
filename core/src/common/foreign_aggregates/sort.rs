use crate::common::tuple::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct SortAggregate {
  pub argsort: bool,
}

impl SortAggregate {
  pub fn sort() -> Self {
    Self { argsort: false }
  }

  pub fn argsort() -> Self {
    Self { argsort: true }
  }
}

impl Into<DynamicAggregate> for SortAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Sort(self)
  }
}

impl Aggregate for SortAggregate {
  type Aggregator<P: Provenance> = SortAggregator;

  fn name(&self) -> String {
    if self.argsort {
      "argsort".to_string()
    } else {
      "sort".to_string()
    }
  }

  fn aggregate_type(&self) -> AggregateType {
    let named_param_types = vec![
      ("asc".to_string(), ParamType::Optional(ValueType::Bool)),
      ("desc".to_string(), ParamType::Optional(ValueType::Bool)),
    ]
    .into_iter()
    .collect();

    if self.argsort {
      AggregateType {
        generics: vec![
          ("A".to_string(), GenericTypeFamily::non_empty_tuple()),
          ("T".to_string(), GenericTypeFamily::non_empty_tuple()),
        ]
        .into_iter()
        .collect(),
        named_param_types,
        input_type: BindingTypes::generic("T"),
        arg_type: BindingTypes::generic("A"),
        output_type: BindingTypes::tuple(vec![
          BindingType::value_type(ValueType::USize),
          BindingType::generic("A"),
        ]),
        ..Default::default()
      }
    } else {
      AggregateType {
        generics: vec![
          ("A".to_string(), GenericTypeFamily::possibly_empty_tuple()),
          ("T".to_string(), GenericTypeFamily::non_empty_tuple()),
        ]
        .into_iter()
        .collect(),
        named_param_types,
        input_type: BindingTypes::generic("T"),
        arg_type: BindingTypes::generic("A"),
        output_type: BindingTypes::tuple(vec![
          BindingType::value_type(ValueType::USize),
          BindingType::generic("A"),
          BindingType::generic("T"),
        ]),
        ..Default::default()
      }
    }
  }

  fn instantiate<P: Provenance>(&self, info: AggregateInfo) -> Self::Aggregator<P> {
    let asc = info.named_params.get("asc").map(|v| v.as_bool());
    let not_desc = info.named_params.get("desc").map(|v| !v.as_bool());
    SortAggregator {
      argsort: self.argsort,
      num_args: info.arg_var_types.len(),
      ascending: asc.unwrap_or(not_desc.unwrap_or(false)),
    }
  }
}

#[derive(Clone)]
pub struct SortAggregator {
  pub argsort: bool,
  pub num_args: usize,
  pub ascending: bool,
}

impl SortAggregator {
  pub fn compare(&self, t1: &Tuple, t2: &Tuple) -> std::cmp::Ordering {
    match (t1, t2) {
      (Tuple::Value(v1), Tuple::Value(v2)) => v1.cmp(v2),
      (Tuple::Tuple(ts1), Tuple::Tuple(ts2)) => ts1[self.num_args..].cmp(&ts2[self.num_args..]),
      _ => unreachable!("tuples are not of same type"),
    }
  }
}

impl<Prov: Provenance> Aggregator<Prov> for SortAggregator {
  default fn aggregate(
    &self,
    _: &Prov,
    _: &RuntimeEnvironment,
    mut batch: DynamicElements<Prov>,
  ) -> DynamicElements<Prov> {
    // Sort the batch
    if self.ascending {
      batch.sort_by(|e1, e2| self.compare(&e1.tuple, &e2.tuple));
    } else {
      batch.sort_by(|e1, e2| self.compare(&e2.tuple, &e1.tuple));
    }

    // Return
    batch
      .into_iter()
      .enumerate()
      .map(|(i, elem)| {
        let values = std::iter::once(i.into());
        let tup = if self.argsort {
          Tuple::from_values(values.chain(elem.tuple.to_values().into_iter().take(self.num_args)))
        } else {
          Tuple::from_values(values.chain(elem.tuple.to_values()))
        };
        DynamicElement::new(tup, elem.tag)
      })
      .collect()
  }
}
