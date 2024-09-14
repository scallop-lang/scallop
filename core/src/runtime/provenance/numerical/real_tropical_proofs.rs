use std::collections::*;
use itertools::Itertools;

use crate::{common::input_tag::DynamicInputTag, utils::IdAllocator2};

use super::*;

pub type VariableID = usize;

#[derive(Clone, Debug)]
pub enum RealTropicalProofsInputTag {
  Float(f64),
  NewVariable,
}

impl StaticInputTag for RealTropicalProofsInputTag {
  fn from_dynamic_input_tag(t: &DynamicInputTag) -> Option<Self> {
    match t {
      DynamicInputTag::None => None,
      DynamicInputTag::NewVariable => Some(Self::NewVariable),
      DynamicInputTag::Exclusive(_) => None,
      DynamicInputTag::Bool(_) => None,
      DynamicInputTag::Natural(_) => None,
      DynamicInputTag::Float(f) => Some(RealTropicalProofsInputTag::Float(*f)),
      DynamicInputTag::ExclusiveFloat(f, _) => Some(RealTropicalProofsInputTag::Float(*f)),
      DynamicInputTag::FloatWithID(_, f) => Some(RealTropicalProofsInputTag::Float(*f)),
      DynamicInputTag::ExclusiveFloatWithID(_, f, _) => Some(RealTropicalProofsInputTag::Float(*f)),
      _ => None,
    }
  }
}

#[derive(Clone, Debug)]
pub struct RealTropicalProofsTag {
  /// The real part of the tag
  pub real: f64,

  /// The proof part of the tag, which is a ordered set of variable IDs
  pub proof: BTreeSet<VariableID>,
}

impl Tag for RealTropicalProofsTag {}

impl std::fmt::Display for RealTropicalProofsTag {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let proof = self.proof.iter().map(|u| u.to_string()).join(", ");
    f.write_fmt(format_args!("[{}]{{{}}}", self.real, proof))
  }
}

#[derive(Clone, Debug, Default)]
pub struct RealTropicalProofsProvenance {
  id_allocator: IdAllocator2,
}

impl RealTropicalProofsProvenance {
  pub fn new() -> Self {
    Self {
      id_allocator: IdAllocator2::new(),
    }
  }
}

impl Provenance for RealTropicalProofsProvenance {
  type Tag = RealTropicalProofsTag;

  type InputTag = RealTropicalProofsInputTag;

  type OutputTag = RealTropicalProofsTag;

  fn name(&self) -> String {
    format!("realtropicalproofs")
  }

  fn tagging_fn(&self, ext_tag: Self::InputTag) -> Self::Tag {
    match ext_tag {
      Self::InputTag::Float(f) => RealTropicalProofsTag {
        real: f,
        proof: BTreeSet::new(),
      },
      Self::InputTag::NewVariable => RealTropicalProofsTag {
        real: 0.0,
        proof: std::iter::once(self.id_allocator.alloc()).collect(),
      }
    }
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    t.clone()
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    if t.real.is_infinite() {
      return true;
    }
    return t.proof.is_empty();
  }

  fn weight(&self, tag: &Self::Tag) -> f64 {
    (-tag.real).exp()
  }

  fn zero(&self) -> Self::Tag {
    Self::Tag {
      real: std::f64::INFINITY,
      proof: BTreeSet::new(),
    }
  }

  fn one(&self) -> Self::Tag {
    Self::Tag {
      real: 0.0,
      proof: BTreeSet::new(),
    }
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    if t1.real < t2.real {
      t1.clone()
    } else {
      t2.clone()
    }
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    Self::Tag {
      real: t1.real + t2.real,
      proof: t1.proof.iter().cloned().chain(t2.proof.iter().cloned()).collect(),
    }
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    (t_old.real - t_new.real).abs() < 0.01
  }
}
