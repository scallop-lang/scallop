use std::collections::*;

use crate::common::value::*;
use crate::common::value_type::*;
use crate::runtime::env::*;

use super::*;

#[derive(Clone)]
pub struct TopKSamplerAggregate {
  is_unique: bool,
}

impl TopKSamplerAggregate {
  pub fn top() -> Self {
    Self { is_unique: false }
  }

  pub fn unique() -> Self {
    Self { is_unique: true }
  }
}

impl Into<DynamicAggregate> for TopKSamplerAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Sampler(DynamicSampleAggregate::new(self))
  }
}

impl SampleAggregate for TopKSamplerAggregate {
  fn name(&self) -> String {
    if self.is_unique {
      "unique".to_string()
    } else {
      "top".to_string()
    }
  }

  fn param_types(&self) -> Vec<ParamType> {
    if self.is_unique {
      vec![]
    } else {
      vec![ParamType::Optional(ValueType::USize)]
    }
  }

  fn instantiate(&self, params: Vec<Value>, _: bool, _: Vec<ValueType>, _: Vec<ValueType>) -> DynamicSampler {
    if self.is_unique {
      TopKSampler { k: 1 }.into()
    } else {
      TopKSampler {
        k: params.get(0).map(|v| v.as_usize()).unwrap_or(1),
      }
      .into()
    }
  }
}

#[derive(Clone)]
pub struct TopKSampler {
  k: usize,
}

impl TopKSampler {
  pub fn new(k: usize) -> Self {
    Self { k }
  }
}

impl Into<DynamicSampler> for TopKSampler {
  fn into(self) -> DynamicSampler {
    DynamicSampler::new(self)
  }
}

impl Sampler for TopKSampler {
  fn sampler_type(&self) -> SamplerType {
    SamplerType::WeightOnly
  }

  fn sample_weight_only(&self, _: &RuntimeEnvironment, elems: Vec<f64>) -> Vec<usize> {
    aggregate_top_k_helper(elems.len(), self.k, |i| elems[i])
  }
}

pub fn aggregate_top_k_helper<F>(num_elements: usize, k: usize, weight_fn: F) -> Vec<usize>
where
  F: Fn(usize) -> f64,
{
  #[derive(Clone, Debug)]
  struct Element {
    id: usize,
    weight: f64,
  }

  impl std::cmp::PartialEq for Element {
    fn eq(&self, other: &Self) -> bool {
      self.id == other.id
    }
  }

  impl std::cmp::Eq for Element {}

  impl std::cmp::PartialOrd for Element {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
      other.weight.partial_cmp(&self.weight)
    }
  }

  impl std::cmp::Ord for Element {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
      if let Some(ord) = other.weight.partial_cmp(&self.weight) {
        ord
      } else {
        other.id.cmp(&self.id)
      }
    }
  }

  // Create a min-heap
  let mut heap = BinaryHeap::new();

  // First insert k elements into the heap
  let size = k.min(num_elements);
  for id in 0..size {
    let elem = Element {
      id,
      weight: weight_fn(id),
    };
    heap.push(elem);
  }

  // Then iterate through all other elements
  if heap.len() > 0 {
    for id in size..num_elements {
      let elem = Element {
        id,
        weight: weight_fn(id),
      };
      let min_elem_in_heap = heap.peek().unwrap();
      if &elem < min_elem_in_heap {
        heap.pop();
        heap.push(elem);
      }
    }
  }

  // Return the list of ids in the heap
  heap.into_iter().map(|elem| elem.id).collect()
}

pub fn unweighted_aggregate_top_k_helper<T>(elements: Vec<T>, k: usize) -> Vec<T> {
  if elements.len() <= k {
    elements
  } else {
    elements.into_iter().take(k).collect()
  }
}
