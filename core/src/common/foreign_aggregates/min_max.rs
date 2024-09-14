use crate::common::tuple::*;
use crate::common::tuples::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct MinMaxAggregate {
  is_min: bool,
  arg_only: bool,
}

impl MinMaxAggregate {
  pub fn min() -> Self {
    Self {
      is_min: true,
      arg_only: false,
    }
  }

  pub fn max() -> Self {
    Self {
      is_min: false,
      arg_only: false,
    }
  }

  pub fn argmin() -> Self {
    Self {
      is_min: true,
      arg_only: true,
    }
  }

  pub fn argmax() -> Self {
    Self {
      is_min: false,
      arg_only: true,
    }
  }
}

impl Into<DynamicAggregate> for MinMaxAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::MinMax(self)
  }
}

impl Aggregate for MinMaxAggregate {
  type Aggregator<P: Provenance> = MinMaxAggregator;

  fn name(&self) -> String {
    if self.is_min {
      if self.arg_only {
        "argmin".to_string()
      } else {
        "min".to_string()
      }
    } else {
      if self.arg_only {
        "argmax".to_string()
      } else {
        "max".to_string()
      }
    }
  }

  /// `{A: Tuple?, T: Tuple} ==> (A, T) := min[A](T)`
  /// `{A: Tuple?, T: Tuple} ==> (A, T) := max[A](T)`
  /// `{A: Tuple, T: Tuple} ==> A := argmin[A](T)`
  /// `{A: Tuple, T: Tuple} ==> A := argmax[A](T)`
  fn aggregate_type(&self) -> AggregateType {
    if self.arg_only {
      AggregateType {
        generics: vec![
          ("A".to_string(), GenericTypeFamily::non_empty_tuple()),
          ("T".to_string(), GenericTypeFamily::non_empty_tuple()),
        ]
        .into_iter()
        .collect(),
        arg_type: BindingTypes::generic("A"),
        input_type: BindingTypes::generic("T"),
        output_type: BindingTypes::generic("A"),
        allow_exclamation_mark: true,
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
        arg_type: BindingTypes::generic("A"),
        input_type: BindingTypes::generic("T"),
        output_type: BindingTypes::if_not_unit(
          "A",
          BindingTypes::tuple(vec![BindingType::generic("A"), BindingType::generic("T")]),
          BindingTypes::generic("T"),
        ),
        allow_exclamation_mark: true,
        ..Default::default()
      }
    }
  }

  fn instantiate<P: Provenance>(&self, info: AggregateInfo) -> Self::Aggregator<P> {
    MinMaxAggregator {
      is_min: self.is_min,
      arg_only: self.arg_only,
      non_multi_world: info.has_exclamation_mark,
      num_args: info.arg_var_types.len(),
    }
  }
}

#[derive(Clone)]
pub struct MinMaxAggregator {
  pub is_min: bool,
  pub arg_only: bool,
  pub non_multi_world: bool,
  pub num_args: usize,
}

impl MinMaxAggregator {
  pub fn min() -> Self {
    Self {
      is_min: true,
      arg_only: false,
      non_multi_world: false,
      num_args: 0,
    }
  }

  pub fn max() -> Self {
    Self {
      is_min: false,
      arg_only: false,
      non_multi_world: false,
      num_args: 0,
    }
  }

  pub fn argmin(num_args: usize) -> Self {
    Self {
      is_min: true,
      arg_only: true,
      non_multi_world: false,
      num_args,
    }
  }

  pub fn argmax(num_args: usize) -> Self {
    Self {
      is_min: false,
      arg_only: true,
      non_multi_world: false,
      num_args,
    }
  }
}

impl MinMaxAggregator {
  pub fn post_process_arg<P: Provenance, I: Iterator<Item = DynamicElement<P>>>(&self, elems: I) -> DynamicElements<P> {
    elems
      .map(|e| {
        // Depending on whether we only need the argument, return only the arg part of the results
        if self.arg_only {
          let tuple = if self.num_args == 1 {
            e.tuple[0].clone()
          } else {
            Tuple::tuple(e.tuple[..self.num_args].iter().cloned())
          };
          DynamicElement::new(tuple, e.tag)
        } else {
          DynamicElement::new(e.tuple, e.tag)
        }
      })
      .collect()
  }

  pub fn discrete_min_max<'a, P: Provenance, I: Iterator<Item = &'a Tuple>>(
    &self,
    p: &P,
    tuple_iter: I,
  ) -> DynamicElements<P> {
    let min_max = if self.is_min {
      if self.num_args > 0 {
        tuple_iter.arg_minimum(self.num_args)
      } else {
        tuple_iter.minimum()
      }
    } else {
      if self.num_args > 0 {
        tuple_iter.arg_maximum(self.num_args)
      } else {
        tuple_iter.maximum()
      }
    };
    self.post_process_arg(min_max.into_iter().map(|t| DynamicElement::new(t.clone(), p.one())))
  }

  pub fn multi_world_min_max<P: Provenance>(&self, prov: &P, mut batch: DynamicElements<P>) -> DynamicElements<P> {
    // Check if there is argument variables
    if self.num_args > 0 {
      // First, we sort all the tuples
      batch.sort_by_key(|e| e.tuple[self.num_args..].iter().cloned().collect::<Vec<_>>());
      let tagged_tuples = batch;

      // Then we compute the strata
      //
      // For example, for the array [0, 0, 0, 1,  1,  2 , 3,  3],
      // We have 4 strata            ---1---  --2--  -3-  --4--
      // they are represented by their start index, [0, 3, 5, 6]
      let strata = {
        let (mut maybe_curr_elem, mut strata) = (None, vec![0]);
        for (i, tagged_tuple) in tagged_tuples.iter().enumerate() {
          if let Some(curr_elem) = maybe_curr_elem {
            if &tagged_tuple.tuple[self.num_args..] > curr_elem {
              strata.push(i);
            }
          } else {
            maybe_curr_elem = Some(&tagged_tuple.tuple[self.num_args..]);
          }
        }
        strata
      };

      // We now compute the tags
      //
      // Let's take `minimum` as an example. Suppose we are in a stratum.
      // For an element inside of this stratum to be the minimum of the whole batch,
      // It has to be the case that all the element before this stratum are FALSE (a.k.a. have a negated tag)
      // the elements in and after this stratum do not matter,
      let mut result = vec![];
      for (i, stratum_start) in strata.iter().copied().enumerate() {
        let stratum_end = if i + 1 < strata.len() {
          strata[i + 1]
        } else {
          tagged_tuples.len()
        };
        let false_range = if self.is_min {
          0..stratum_start
        } else {
          stratum_end..tagged_tuples.len()
        };
        let maybe_false_tag = false_range.fold(Some(prov.one()), |maybe_acc, j| {
          maybe_acc.and_then(|acc| prov.negate(&tagged_tuples[j].tag).map(|neg| prov.mult(&acc, &neg)))
        });
        if let Some(false_tag) = maybe_false_tag {
          for j in stratum_start..stratum_end {
            let and_true_tag: P::Tag = prov.mult(&false_tag, &tagged_tuples[j].tag);
            result.push(DynamicElement::<P>::new(tagged_tuples[j].tuple.clone(), and_true_tag));
          }
        }
      }

      // Depending on whether we only need the argument, return only the arg part of the results
      self.post_process_arg(result.into_iter())
    } else {
      // If there is no argument variable...
      let mut result = vec![];
      let mut accumulated_false_tag = prov.one();

      // Depending on minimum/maximum...
      for i in 0..batch.len() {
        let i = if self.is_min { i } else { batch.len() - 1 - i };
        let Tagged { tuple, tag } = &batch[i];
        let and_true_tag = prov.mult(&accumulated_false_tag, tag);
        result.push(DynamicElement::<P>::new(tuple.clone(), and_true_tag));
        if let Some(f) = prov.negate(tag).map(|neg| prov.mult(&accumulated_false_tag, &neg)) {
          accumulated_false_tag = f;
        }
      }

      // Depending on whether we only need the argument, return only the arg part of the results
      self.post_process_arg(result.into_iter())
    }
  }
}

impl<P: Provenance> Aggregator<P> for MinMaxAggregator {
  default fn aggregate(&self, p: &P, _env: &RuntimeEnvironment, batch: DynamicElements<P>) -> DynamicElements<P> {
    if self.non_multi_world {
      self.discrete_min_max(p, batch.iter_tuples())
    } else {
      self.multi_world_min_max(p, batch)
    }
  }
}
