use itertools::Itertools;

use crate::common::tuple::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;
use crate::utils;

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
      generics: vec![("A".to_string(), GenericTypeFamily::possibly_empty_tuple())]
        .into_iter()
        .collect(),
      param_types: vec![ParamType::Optional(ValueType::String)],
      named_param_types: vec![("all".to_string(), ParamType::Optional(ValueType::Bool))]
        .into_iter()
        .collect(),
      arg_type: BindingTypes::generic("A"),
      input_type: BindingTypes::value_type(ValueType::String),
      output_type: BindingTypes::value_type(ValueType::String),
      allow_exclamation_mark: true,
      ..Default::default()
    }
  }

  fn instantiate<P: Provenance>(&self, info: AggregateInfo) -> Self::Aggregator<P> {
    StringJoinAggregator {
      non_multi_world: info.has_exclamation_mark,
      num_args: info.arg_var_types.len(),
      use_all_args: info.named_params.get("all").map(|v| v.as_bool()).unwrap_or(false),
      separator: info
        .pos_params
        .get(0)
        .map(|v| v.as_str().to_string())
        .unwrap_or("".to_string()),
    }
  }
}

#[derive(Clone)]
pub struct StringJoinAggregator {
  non_multi_world: bool,
  num_args: usize,
  use_all_args: bool,
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

  pub fn aggregate_non_multi_world<P: Provenance>(&self, prov: &P, batch: DynamicElements<P>) -> DynamicElements<P> {
    let res = self.perform_string_join(batch.iter_tuples());
    vec![DynamicElement::new(res, prov.one())]
  }

  pub fn aggregate_all_multi_world<P: Provenance>(&self, prov: &P, batch: DynamicElements<P>) -> DynamicElements<P> {
    let res = self.perform_string_join(batch.iter_tuples());
    let tag = batch.iter().fold(prov.one(), |acc, elem| prov.mult(&acc, &elem.tag));
    vec![DynamicElement::new(res, tag)]
  }

  pub fn aggregate_all_arg_multi_world<P: Provenance>(
    &self,
    prov: &P,
    batch: DynamicElements<P>,
  ) -> DynamicElements<P> {
    let mut groups: Vec<(_, Vec<_>, Vec<_>)> = vec![];
    for elem in &batch {
      if let Some((prev_arg, values, tags)) = groups.last_mut() {
        if prev_arg == &&elem.tuple[..self.num_args] {
          values.push(&elem.tuple[self.num_args]);
          tags.push(&elem.tag);
        } else {
          groups.push((
            &elem.tuple[..self.num_args],
            vec![&elem.tuple[self.num_args]],
            vec![&elem.tag],
          ));
        }
      } else {
        groups.push((
          &elem.tuple[..self.num_args],
          vec![&elem.tuple[self.num_args]],
          vec![&elem.tag],
        ));
      }
    }

    let mut results = vec![];
    let group_sizes = groups.iter().map(|(_, vs, _)| vs.len()).collect::<Vec<_>>();
    for selected_indices in utils::cartesian(group_sizes) {
      let strings: Vec<_> = selected_indices
        .iter()
        .enumerate()
        .map(|(i, j)| groups[i].1[*j].as_string())
        .collect();
      let joined = strings.join(&self.separator);
      let tag = selected_indices
        .iter()
        .enumerate()
        .fold(prov.one(), |acc, (i, j)| prov.mult(&acc, groups[i].2[*j]));
      results.push(DynamicElement::new(joined, tag));
    }

    results
  }

  pub fn aggregate_multi_world<P: Provenance>(&self, prov: &P, batch: DynamicElements<P>) -> DynamicElements<P> {
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

impl<P: Provenance> Aggregator<P> for StringJoinAggregator {
  default fn aggregate(&self, prov: &P, _env: &RuntimeEnvironment, batch: DynamicElements<P>) -> DynamicElements<P> {
    if self.non_multi_world {
      self.aggregate_non_multi_world(prov, batch)
    } else if self.use_all_args {
      if self.num_args == 0 {
        self.aggregate_all_multi_world(prov, batch)
      } else {
        self.aggregate_all_arg_multi_world(prov, batch)
      }
    } else {
      self.aggregate_multi_world(prov, batch)
    }
  }
}
