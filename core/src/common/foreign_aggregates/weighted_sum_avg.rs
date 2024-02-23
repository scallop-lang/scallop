use itertools::Itertools;
use std::convert::*;

use crate::common::tuple::*;
use crate::common::type_family::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;
use crate::utils::*;

use super::*;

#[derive(Clone)]
pub struct WeightedSumAvgAggregate {
  is_sum: bool,
}

impl WeightedSumAvgAggregate {
  pub fn weighted_sum() -> Self {
    Self { is_sum: true }
  }

  pub fn weighted_avg() -> Self {
    Self { is_sum: false }
  }
}

impl Into<DynamicAggregate> for WeightedSumAvgAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::WeightedSumAvg(self)
  }
}

impl Aggregate for WeightedSumAvgAggregate {
  type Aggregator<P: Provenance> = WeightedSumAvgAggregator;

  fn name(&self) -> String {
    if self.is_sum {
      "weighted_sum".to_string()
    } else {
      "weighted_avg".to_string()
    }
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: vec![
        ("A".to_string(), GenericTypeFamily::possibly_empty_tuple()),
        ("T".to_string(), GenericTypeFamily::type_family(TypeFamily::Float)),
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
    WeightedSumAvgAggregator {
      is_sum: self.is_sum,
      non_multi_world: info.has_exclamation_mark,
      num_args: info.arg_var_types.len(),
      value_type: info.input_var_types[0].clone(),
    }
  }
}

#[derive(Clone)]
pub struct WeightedSumAvgAggregator {
  is_sum: bool,
  non_multi_world: bool,
  num_args: usize,
  value_type: ValueType,
}

impl WeightedSumAvgAggregator {
  pub fn weighted_sum<T>(non_multi_world: bool) -> Self
  where
    ValueType: FromType<T>,
  {
    Self {
      is_sum: true,
      non_multi_world,
      num_args: 0,
      value_type: ValueType::from_type(),
    }
  }

  pub fn weighted_avg<T>(non_multi_world: bool) -> Self
  where
    ValueType: FromType<T>,
  {
    Self {
      is_sum: false,
      non_multi_world,
      num_args: 0,
      value_type: ValueType::from_type(),
    }
  }
}

impl WeightedSumAvgAggregator {
  pub fn perform_avg<'a, I: Iterator<Item = (f64, &'a Tuple)>, F: Float + Into<Tuple>>(&self, i: I) -> Tuple
  where
    Tuple: AsTuple<F>,
  {
    let (sum, sum_of_weight): (F, F) = if self.num_args > 0 {
      i.fold((F::zero(), F::zero()), |(sum, sum_of_weight), (weight, tuple)| {
        let mult = F::from_f64(weight) * AsTuple::<F>::as_tuple(&tuple[self.num_args]);
        (sum + mult, sum_of_weight + F::from_f64(weight))
      })
    } else {
      i.fold((F::zero(), F::zero()), |(sum, sum_of_weight), (weight, tuple)| {
        let mult = F::from_f64(weight) * AsTuple::<F>::as_tuple(tuple);
        (sum + mult, sum_of_weight + F::from_f64(weight))
      })
    };
    (sum / sum_of_weight).into()
  }

  pub fn perform_sum<'a, I: Iterator<Item = (f64, &'a Tuple)>, F: Float + Into<Tuple>>(&self, i: I) -> Tuple
  where
    Tuple: AsTuple<F>,
  {
    let sum: F = if self.num_args > 0 {
      i.fold(F::zero(), |sum, (weight, tuple)| {
        sum + F::from_f64(weight) * AsTuple::<F>::as_tuple(&tuple[self.num_args])
      })
    } else {
      i.fold(F::zero(), |sum, (weight, tuple)| {
        sum + F::from_f64(weight) * AsTuple::<F>::as_tuple(tuple)
      })
    };
    sum.into()
  }

  pub fn perform_sum_avg_typed<'a, I: Iterator<Item = (f64, &'a Tuple)>, F: Float + Into<Tuple>>(&self, i: I) -> Tuple
  where
    Tuple: AsTuple<F>,
  {
    if self.is_sum {
      self.perform_sum(i)
    } else {
      self.perform_avg(i)
    }
  }

  pub fn perform_sum_avg<'a, I: Iterator<Item = (f64, &'a Tuple)>>(&self, i: I) -> Tuple {
    match self.value_type {
      ValueType::F32 => self.perform_sum_avg_typed::<_, f32>(i),
      ValueType::F64 => self.perform_sum_avg_typed::<_, f64>(i),
      _ => unreachable!("type checking should have confirmed that the value type is f32 or f64"),
    }
  }
}

impl<P: Provenance> Aggregator<P> for WeightedSumAvgAggregator {
  default fn aggregate(&self, prov: &P, _env: &RuntimeEnvironment, batch: DynamicElements<P>) -> DynamicElements<P> {
    if self.non_multi_world {
      let res = self.perform_sum_avg(batch.iter().map(|e| (prov.weight(&e.tag), &e.tuple)));
      vec![DynamicElement::new(res, prov.one())]
    } else {
      let mut result = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let res = self.perform_sum_avg(
          chosen_set
            .iter()
            .map(|i| (prov.weight(&batch[*i].tag), &batch[*i].tuple)),
        );
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
