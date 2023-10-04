use std::collections::*;

use rand::distributions::WeightedIndex;

use crate::common::value::*;
use crate::common::value_type::*;
use crate::runtime::env::*;

use super::*;

#[derive(Clone)]
pub struct CategoricalAggregate;

impl CategoricalAggregate {
  pub fn new() -> Self {
    Self
  }
}

impl Into<DynamicAggregate> for CategoricalAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Sampler(DynamicSampleAggregate::new(self))
  }
}

impl SampleAggregate for CategoricalAggregate {
  fn name(&self) -> String {
    "uniform".to_string()
  }

  fn param_types(&self) -> Vec<ParamType> {
    vec![ParamType::Mandatory(ValueType::USize)]
  }

  fn instantiate(
    &self,
    params: Vec<Value>,
    _has_exclamation_mark: bool,
    _arg_types: Vec<ValueType>,
    _input_types: Vec<ValueType>,
  ) -> DynamicSampler {
    CategoricalSampler {
      k: params[0].as_usize(),
    }
    .into()
  }
}

#[derive(Clone)]
pub struct CategoricalSampler {
  k: usize,
}

impl Into<DynamicSampler> for CategoricalSampler {
  fn into(self) -> DynamicSampler {
    DynamicSampler::new(self)
  }
}

impl Sampler for CategoricalSampler {
  fn sampler_type(&self) -> SamplerType {
    SamplerType::WeightOnly
  }

  fn sample_weight_only(&self, rt: &RuntimeEnvironment, weights: Vec<f64>) -> Vec<usize> {
    if weights.len() <= self.k {
      (0..weights.len()).collect()
    } else {
      let dist = WeightedIndex::new(&weights).unwrap();
      (0..self.k)
        .map(|_| rt.random.sample_from(&dist))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
    }
  }
}
