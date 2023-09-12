use pyo3::class::iter::IterNextOutput;
use pyo3::prelude::*;

use scallop_core::common;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::*;

use crate::runtime::PythonRuntimeEnvironment;

use super::custom_tag;
use super::external_tag::*;
use super::provenance::*;
use super::tuple::*;

#[derive(Clone)]
pub enum CollectionEnum<P: PointerFamily> {
  Unit {
    collection: P::Rc<DynamicOutputCollection<unit::UnitProvenance>>,
  },
  Proofs {
    collection: P::Rc<DynamicOutputCollection<proofs::ProofsProvenance<P>>>,
  },
  MinMaxProb {
    collection: P::Rc<DynamicOutputCollection<min_max_prob::MinMaxProbProvenance>>,
  },
  AddMultProb {
    collection: P::Rc<DynamicOutputCollection<add_mult_prob::AddMultProbProvenance>>,
  },
  TopKProofs {
    collection: P::Rc<DynamicOutputCollection<top_k_proofs::TopKProofsProvenance<P>>>,
  },
  TopBottomKClauses {
    collection: P::Rc<DynamicOutputCollection<top_bottom_k_clauses::TopBottomKClausesProvenance<P>>>,
  },
  DiffMinMaxProb {
    collection: P::Rc<DynamicOutputCollection<diff_min_max_prob::DiffMinMaxProbProvenance<ExtTag, P>>>,
    tags: P::RcCell<Vec<ExtTag>>,
  },
  DiffAddMultProb {
    collection: P::Rc<DynamicOutputCollection<diff_add_mult_prob::DiffAddMultProbProvenance<ExtTag, P>>>,
    tags: P::RcCell<Vec<ExtTag>>,
  },
  DiffNandMultProb {
    collection: P::Rc<DynamicOutputCollection<diff_nand_mult_prob::DiffNandMultProbProvenance<ExtTag, P>>>,
    tags: P::RcCell<Vec<ExtTag>>,
  },
  DiffMaxMultProb {
    collection: P::Rc<DynamicOutputCollection<diff_max_mult_prob::DiffMaxMultProbProvenance<ExtTag, P>>>,
    tags: P::RcCell<Vec<ExtTag>>,
  },
  DiffNandMinProb {
    collection: P::Rc<DynamicOutputCollection<diff_nand_min_prob::DiffNandMinProbProvenance<ExtTag, P>>>,
    tags: P::RcCell<Vec<ExtTag>>,
  },
  DiffSampleKProofs {
    collection: P::Rc<DynamicOutputCollection<diff_sample_k_proofs::DiffSampleKProofsProvenance<ExtTag, P>>>,
    tags: DiffProbStorage<ExtTag, P>,
  },
  DiffTopKProofs {
    collection: P::Rc<DynamicOutputCollection<diff_top_k_proofs::DiffTopKProofsProvenance<ExtTag, P>>>,
    tags: DiffProbStorage<ExtTag, P>,
  },
  DiffTopBottomKClauses {
    collection: P::Rc<DynamicOutputCollection<diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance<ExtTag, P>>>,
    tags: DiffProbStorage<ExtTag, P>,
  },
  Custom {
    collection: P::Rc<DynamicOutputCollection<custom_tag::CustomProvenance>>,
  },
}

macro_rules! match_collection {
  ($ctx:expr, $v:ident, $e:expr) => {
    match $ctx {
      CollectionEnum::Unit { collection: $v } => $e,
      CollectionEnum::Proofs { collection: $v } => $e,
      CollectionEnum::MinMaxProb { collection: $v } => $e,
      CollectionEnum::AddMultProb { collection: $v } => $e,
      CollectionEnum::TopKProofs { collection: $v, .. } => $e,
      CollectionEnum::TopBottomKClauses { collection: $v, .. } => $e,
      CollectionEnum::DiffMinMaxProb { collection: $v, .. } => $e,
      CollectionEnum::DiffAddMultProb { collection: $v, .. } => $e,
      CollectionEnum::DiffNandMultProb { collection: $v, .. } => $e,
      CollectionEnum::DiffMaxMultProb { collection: $v, .. } => $e,
      CollectionEnum::DiffNandMinProb { collection: $v, .. } => $e,
      CollectionEnum::DiffSampleKProofs { collection: $v, .. } => $e,
      CollectionEnum::DiffTopKProofs { collection: $v, .. } => $e,
      CollectionEnum::DiffTopBottomKClauses { collection: $v, .. } => $e,
      CollectionEnum::Custom { collection: $v } => $e,
    }
  };
}

impl CollectionEnum<ArcFamily> {
  pub fn num_input_facts(&self) -> Option<usize> {
    match self {
      Self::Unit { .. } => None,
      Self::Proofs { .. } => None,
      Self::MinMaxProb { .. } => None,
      Self::AddMultProb { .. } => None,
      Self::TopKProofs { .. } => None,
      Self::TopBottomKClauses { .. } => None,
      Self::DiffMinMaxProb { tags, .. } => Some(ArcFamily::get_rc_cell(tags, |t| t.len())),
      Self::DiffAddMultProb { tags, .. } => Some(ArcFamily::get_rc_cell(tags, |t| t.len())),
      Self::DiffNandMultProb { tags, .. } => Some(ArcFamily::get_rc_cell(tags, |t| t.len())),
      Self::DiffMaxMultProb { tags, .. } => Some(ArcFamily::get_rc_cell(tags, |t| t.len())),
      Self::DiffNandMinProb { tags, .. } => Some(ArcFamily::get_rc_cell(tags, |t| t.len())),
      Self::DiffSampleKProofs { tags, .. } => Some(tags.num_input_tags()),
      Self::DiffTopKProofs { tags, .. } => Some(tags.num_input_tags()),
      Self::DiffTopBottomKClauses { tags, .. } => Some(tags.num_input_tags()),
      Self::Custom { .. } => None,
    }
  }

  pub fn input_tags(&self) -> Option<Vec<Py<PyAny>>> {
    match self {
      Self::Unit { .. } => None,
      Self::Proofs { .. } => None,
      Self::MinMaxProb { .. } => None,
      Self::AddMultProb { .. } => None,
      Self::TopKProofs { .. } => None,
      Self::TopBottomKClauses { .. } => None,
      Self::DiffMinMaxProb { .. } => None,
      Self::DiffAddMultProb { tags, .. } => Some(ArcFamily::get_rc_cell(tags, |t| t.clone()).into_vec()),
      Self::DiffNandMultProb { tags, .. } => Some(ArcFamily::clone_rc_cell_internal(tags).into_vec()),
      Self::DiffMaxMultProb { tags, .. } => Some(ArcFamily::clone_rc_cell_internal(tags).into_vec()),
      Self::DiffNandMinProb { tags, .. } => Some(ArcFamily::clone_rc_cell_internal(tags).into_vec()),
      Self::DiffSampleKProofs { tags, .. } => Some(tags.input_tags().into_vec()),
      Self::DiffTopKProofs { tags, .. } => Some(tags.input_tags().into_vec()),
      Self::DiffTopBottomKClauses { tags, .. } => Some(tags.input_tags().into_vec()),
      Self::Custom { .. } => None,
    }
  }

  fn len(&self) -> usize {
    match_collection!(self, c, c.len())
  }

  fn has_empty_tag(&self) -> bool {
    match self {
      Self::Unit { .. } => true,
      Self::Proofs { .. } => true,
      _ => false,
    }
  }

  fn ith_tuple(&self, i: usize) -> &common::tuple::Tuple {
    match_collection!(self, c, &c.ith_tuple(i).unwrap())
  }

  fn ith_tag(&self, i: usize) -> Py<PyAny> {
    fn ith_tag_helper<Prov>(c: &DynamicOutputCollection<Prov>, i: usize) -> Py<PyAny>
    where
      Prov: PythonProvenance,
    {
      Prov::to_output_py_tag(c.ith_tag(i).unwrap())
    }

    match_collection!(self, c, ith_tag_helper(ArcFamily::get_rc(c), i))
  }

  pub fn to_collection(self, env: &RuntimeEnvironment) -> Collection {
    Collection {
      env: env.into(),
      collection: self,
    }
  }
}

#[pyclass(unsendable, name = "InternalScallopCollection")]
pub struct Collection {
  pub env: PythonRuntimeEnvironment,
  pub collection: CollectionEnum<ArcFamily>,
}

#[pymethods]
impl Collection {
  fn num_input_facts(&self) -> Option<usize> {
    self.collection.num_input_facts()
  }

  fn input_tags(&self) -> Option<Vec<Py<PyAny>>> {
    self.collection.input_tags()
  }

  fn len(&self) -> usize {
    self.collection.len()
  }

  fn __iter__(slf: PyRef<Self>) -> CollectionIterator {
    CollectionIterator {
      env: slf.env.clone(),
      collection: slf.collection.clone(),
      current_index: 0,
    }
  }
}

#[pyclass(unsendable, name = "InternalScallopCollectionIterator")]
pub struct CollectionIterator {
  env: PythonRuntimeEnvironment,
  collection: CollectionEnum<ArcFamily>,
  current_index: usize,
}

#[pymethods]
impl CollectionIterator {
  fn __next__(mut slf: PyRefMut<Self>) -> IterNextOutput<Py<PyAny>, &'static str> {
    while slf.current_index < slf.collection.len() {
      let i = slf.current_index;
      slf.current_index += 1;
      if slf.collection.has_empty_tag() {
        if let Some(tuple) = to_python_tuple(slf.collection.ith_tuple(i), &slf.env) {
          return IterNextOutput::Yield(tuple)
        }
      } else {
        let tuple = to_python_tuple(slf.collection.ith_tuple(i), &slf.env);
        let tag = slf.collection.ith_tag(i);
        let elem = Python::with_gil(|py| (tag, tuple).to_object(py));
        return IterNextOutput::Yield(elem)
      }
    }
    IterNextOutput::Return("Ended")
  }
}
