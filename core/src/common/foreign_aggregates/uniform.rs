use std::collections::*;

use crate::common::value_type::*;
use crate::runtime::env::*;

use super::*;

#[derive(Clone)]
pub struct UniformAggregate;

impl UniformAggregate {
  pub fn new() -> Self {
    Self
  }
}

impl Into<DynamicAggregate> for UniformAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Sampler(DynamicSampleAggregate::new(self))
  }
}

impl SampleAggregate for UniformAggregate {
  fn name(&self) -> String {
    "uniform".to_string()
  }

  fn param_types(&self) -> Vec<ParamType> {
    vec![ParamType::Mandatory(ValueType::USize)]
  }

  fn instantiate(&self, info: AggregateInfo) -> DynamicSampler {
    UniformSampler {
      k: info.pos_params[0].as_usize(),
    }
    .into()
  }
}

#[derive(Clone)]
pub struct UniformSampler {
  k: usize,
}

impl Into<DynamicSampler> for UniformSampler {
  fn into(self) -> DynamicSampler {
    DynamicSampler::new(self)
  }
}

impl Sampler for UniformSampler {
  fn sampler_type(&self) -> SamplerType {
    SamplerType::LengthOnly
  }

  fn sample_length_only(&self, rt: &RuntimeEnvironment, len: usize) -> Vec<usize> {
    if len <= self.k {
      (0..len).collect()
    } else {
      (0..self.k)
        .map(|_| rt.random.random_usize(len))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
    }
  }
}
