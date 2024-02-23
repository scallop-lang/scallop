use crate::common::tuple::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

pub trait SampleAggregate: dyn_clone::DynClone + 'static {
  fn name(&self) -> String;

  fn param_types(&self) -> Vec<ParamType>;

  fn instantiate(&self, info: AggregateInfo) -> DynamicSampler;
}

pub struct DynamicSampleAggregate(Box<dyn SampleAggregate>);

unsafe impl Send for DynamicSampleAggregate {}
unsafe impl Sync for DynamicSampleAggregate {}

impl DynamicSampleAggregate {
  pub fn new<T: SampleAggregate>(t: T) -> Self {
    Self(Box::new(t))
  }
}

impl Clone for DynamicSampleAggregate {
  fn clone(&self) -> Self {
    Self(dyn_clone::clone_box(&*self.0))
  }
}

impl Into<DynamicAggregate> for DynamicSampleAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Sampler(self)
  }
}

impl Aggregate for DynamicSampleAggregate {
  type Aggregator<P: Provenance> = DynamicSampler;

  fn name(&self) -> String {
    SampleAggregate::name(&*self.0)
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: vec![("T".to_string(), GenericTypeFamily::non_empty_tuple())]
        .into_iter()
        .collect(),
      param_types: SampleAggregate::param_types(&*self.0),
      input_type: BindingTypes::generic("T"),
      output_type: BindingTypes::generic("T"),
      ..Default::default()
    }
  }

  fn instantiate<P: Provenance>(&self, info: AggregateInfo) -> Self::Aggregator<P> {
    SampleAggregate::instantiate(&*self.0, info)
  }
}

pub enum SamplerType {
  LengthOnly,
  WeightOnly,
  TupleOnly,
  WeightAndTuple,
}

#[allow(unused)]
pub trait Sampler: dyn_clone::DynClone + 'static {
  fn sampler_type(&self) -> SamplerType;

  fn sample_length_only(&self, env: &RuntimeEnvironment, len: usize) -> Vec<usize> {
    vec![]
  }

  fn sample_weight_only(&self, env: &RuntimeEnvironment, elems: Vec<f64>) -> Vec<usize> {
    vec![]
  }

  fn sample_tuple_only(&self, env: &RuntimeEnvironment, elems: Vec<&Tuple>) -> Vec<usize> {
    vec![]
  }

  fn sample_weight_and_tuple(&self, env: &RuntimeEnvironment, elems: Vec<(f64, &Tuple)>) -> Vec<usize> {
    vec![]
  }
}

pub struct DynamicSampler(Box<dyn Sampler>);

impl DynamicSampler {
  pub fn new<T: Sampler>(t: T) -> Self {
    Self(Box::new(t))
  }
}

impl Clone for DynamicSampler {
  fn clone(&self) -> Self {
    Self(dyn_clone::clone_box(&*self.0))
  }
}

impl<P: Provenance> Aggregator<P> for DynamicSampler {
  fn aggregate(&self, p: &P, env: &RuntimeEnvironment, elems: DynamicElements<P>) -> DynamicElements<P> {
    let indices = match self.0.sampler_type() {
      SamplerType::LengthOnly => self.0.sample_length_only(env, elems.len()),
      SamplerType::TupleOnly => {
        let to_sample = elems.iter().map(|e| &e.tuple).collect();
        self.0.sample_tuple_only(env, to_sample)
      }
      SamplerType::WeightOnly => {
        let to_sample = elems.iter().map(|e| p.weight(&e.tag)).collect();
        self.0.sample_weight_only(env, to_sample)
      }
      SamplerType::WeightAndTuple => {
        let to_sample = elems.iter().map(|e| (p.weight(&e.tag), &e.tuple)).collect();
        self.0.sample_weight_and_tuple(env, to_sample)
      }
    };
    indices.into_iter().map(|i| elems[i].clone()).collect()
  }
}
