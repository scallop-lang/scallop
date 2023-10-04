use itertools::Itertools;

use crate::common::tuple::*;
use crate::common::type_family::*;
use crate::common::value::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct SumProdAggregate {
  is_sum: bool,
}

impl SumProdAggregate {
  pub fn sum() -> Self {
    Self { is_sum: true }
  }

  pub fn prod() -> Self {
    Self { is_sum: false }
  }
}

impl Into<DynamicAggregate> for SumProdAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::SumProd(self)
  }
}

impl Aggregate for SumProdAggregate {
  type Aggregator<P: Provenance> = SumProdAggregator;

  fn name(&self) -> String {
    if self.is_sum {
      "sum".to_string()
    } else {
      "prod".to_string()
    }
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: vec![
        ("A".to_string(), GenericTypeFamily::possibly_empty_tuple()),
        ("T".to_string(), GenericTypeFamily::type_family(TypeFamily::Number)),
      ]
      .into_iter()
      .collect(),
      param_types: vec![],
      arg_type: BindingTypes::generic("A"),
      input_type: BindingTypes::generic("T"),
      output_type: BindingTypes::generic("T"),
      allow_exclamation_mark: true,
    }
  }

  fn instantiate<P: Provenance>(
    &self,
    _params: Vec<Value>,
    has_exclamation_mark: bool,
    arg_types: Vec<ValueType>,
    input_types: Vec<ValueType>,
  ) -> Self::Aggregator<P> {
    assert!(
      input_types.len() == 1,
      "sum/prod aggregate should take in argument of only size 1"
    );
    SumProdAggregator {
      is_sum: self.is_sum,
      non_multi_world: has_exclamation_mark,
      num_args: arg_types.len(),
      value_type: input_types[0].clone(),
    }
  }
}

#[derive(Clone)]
pub struct SumProdAggregator {
  is_sum: bool,
  non_multi_world: bool,
  num_args: usize,
  value_type: ValueType,
}

impl SumProdAggregator {
  pub fn sum<T>(non_multi_world: bool) -> Self
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

  pub fn prod<T>(non_multi_world: bool) -> Self
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

impl SumProdAggregator {
  pub fn perform_sum_prod<'a, I: Iterator<Item = &'a Tuple>>(&self, i: I) -> Tuple {
    if self.num_args > 0 {
      let iterator = i.map(|t| &t[self.num_args]);
      if self.is_sum {
        self.value_type.sum(iterator)
      } else {
        self.value_type.prod(iterator)
      }
    } else {
      if self.is_sum {
        self.value_type.sum(i)
      } else {
        self.value_type.prod(i)
      }
    }
  }
}

impl<P: Provenance> Aggregator<P> for SumProdAggregator {
  default fn aggregate(&self, prov: &P, _env: &RuntimeEnvironment, batch: DynamicElements<P>) -> DynamicElements<P> {
    if self.non_multi_world {
      let res = self.perform_sum_prod(batch.iter_tuples());
      vec![DynamicElement::new(res, prov.one())]
    } else {
      let mut result = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let res = self.perform_sum_prod(chosen_set.iter().map(|i| &batch[*i].tuple));
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
