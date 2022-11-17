use std::sync::Arc;

use pyo3::class::iter::IterNextOutput;
use pyo3::prelude::*;

use scallop_core::common;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::ArcFamily;

use super::custom_tag;
use super::provenance::*;
use super::tuple::*;

#[derive(Clone)]
pub enum CollectionEnum {
  Unit {
    collection: Arc<DynamicOutputCollection<unit::UnitProvenance>>,
  },
  Proofs {
    collection: Arc<DynamicOutputCollection<proofs::ProofsProvenance>>,
  },
  MinMaxProb {
    collection: Arc<DynamicOutputCollection<min_max_prob::MinMaxProbProvenance>>,
  },
  AddMultProb {
    collection: Arc<DynamicOutputCollection<add_mult_prob::AddMultProbContext>>,
  },
  TopKProofs {
    collection: Arc<DynamicOutputCollection<top_k_proofs::TopKProofsContext<ArcFamily>>>,
    tags: Arc<Vec<f64>>,
  },
  TopBottomKClauses {
    collection: Arc<DynamicOutputCollection<top_bottom_k_clauses::TopBottomKClausesContext<ArcFamily>>>,
    tags: Arc<Vec<f64>>,
  },
  DiffMinMaxProb {
    collection: Arc<DynamicOutputCollection<diff_min_max_prob::DiffMinMaxProbContext<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, diff_min_max_prob::Derivative<Py<PyAny>>)>>,
  },
  DiffAddMultProb {
    collection: Arc<DynamicOutputCollection<diff_add_mult_prob::DiffAddMultProbContext<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<Py<PyAny>>>,
  },
  DiffNandMultProb {
    collection: Arc<DynamicOutputCollection<diff_nand_mult_prob::DiffNandMultProbContext<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<Py<PyAny>>>,
  },
  DiffMaxMultProb {
    collection: Arc<DynamicOutputCollection<diff_max_mult_prob::DiffMaxMultProbContext<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<Py<PyAny>>>,
  },
  DiffNandMinProb {
    collection: Arc<DynamicOutputCollection<diff_nand_min_prob::DiffNandMinProbContext<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<Py<PyAny>>>,
  },
  DiffSampleKProofs {
    collection: Arc<DynamicOutputCollection<diff_sample_k_proofs::DiffSampleKProofsContext<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, Py<PyAny>)>>,
  },
  DiffTopKProofs {
    collection: Arc<DynamicOutputCollection<diff_top_k_proofs::DiffTopKProofsContext<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, Py<PyAny>)>>,
  },
  DiffTopKProofsIndiv {
    collection: Arc<DynamicOutputCollection<diff_top_k_proofs_indiv::DiffTopKProofsIndivContext<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, Py<PyAny>)>>,
  },
  DiffTopBottomKClauses {
    collection: Arc<DynamicOutputCollection<diff_top_bottom_k_clauses::DiffTopBottomKClausesContext<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, Py<PyAny>)>>,
  },
  Custom {
    collection: Arc<DynamicOutputCollection<custom_tag::CustomTagContext>>,
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
      CollectionEnum::DiffTopKProofsIndiv { collection: $v, .. } => $e,
      CollectionEnum::DiffTopBottomKClauses { collection: $v, .. } => $e,
      CollectionEnum::Custom { collection: $v } => $e,
    }
  };
}

impl CollectionEnum {
  pub fn num_input_facts(&self) -> Option<usize> {
    match self {
      Self::Unit { .. } => None,
      Self::Proofs { .. } => None,
      Self::MinMaxProb { .. } => None,
      Self::AddMultProb { .. } => None,
      Self::TopKProofs { tags, .. } => Some(tags.len()),
      Self::TopBottomKClauses { tags, .. } => Some(tags.len()),
      Self::DiffMinMaxProb { tags, .. } => Some(tags.len()),
      Self::DiffAddMultProb { tags, .. } => Some(tags.len()),
      Self::DiffNandMultProb { tags, .. } => Some(tags.len()),
      Self::DiffMaxMultProb { tags, .. } => Some(tags.len()),
      Self::DiffNandMinProb { tags, .. } => Some(tags.len()),
      Self::DiffSampleKProofs { tags, .. } => Some(tags.len()),
      Self::DiffTopKProofs { tags, .. } => Some(tags.len()),
      Self::DiffTopKProofsIndiv { tags, .. } => Some(tags.len()),
      Self::DiffTopBottomKClauses { tags, .. } => Some(tags.len()),
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
      Self::DiffAddMultProb { tags, .. } => Some(tags.iter().cloned().collect()),
      Self::DiffNandMultProb { tags, .. } => Some(tags.iter().cloned().collect()),
      Self::DiffMaxMultProb { tags, .. } => Some(tags.iter().cloned().collect()),
      Self::DiffNandMinProb { tags, .. } => Some(tags.iter().cloned().collect()),
      Self::DiffSampleKProofs { tags, .. } => Some(tags.iter().map(|(_, t)| t.clone()).collect()),
      Self::DiffTopKProofs { tags, .. } => Some(tags.iter().map(|(_, t)| t.clone()).collect()),
      Self::DiffTopKProofsIndiv { tags, .. } => Some(tags.iter().map(|(_, t)| t.clone()).collect()),
      Self::DiffTopBottomKClauses { tags, .. } => Some(tags.iter().map(|(_, t)| t.clone()).collect()),
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
    fn ith_tag_helper<Prov>(c: &Arc<DynamicOutputCollection<Prov>>, i: usize) -> Py<PyAny>
    where
      Prov: PythonProvenance,
    {
      Prov::to_output_py_tag(c.ith_tag(i).unwrap())
    }

    match_collection!(self, c, ith_tag_helper(c, i))
  }
}

#[pyclass(unsendable, name = "InternalScallopCollection")]
pub struct Collection {
  pub collection: CollectionEnum,
}

#[pymethods]
impl Collection {
  fn num_input_facts(&self) -> Option<usize> {
    self.collection.num_input_facts()
  }

  fn input_tags(&self) -> Option<Vec<Py<PyAny>>> {
    self.collection.input_tags()
  }

  fn __iter__(slf: PyRef<Self>) -> CollectionIterator {
    CollectionIterator {
      collection: slf.collection.clone(),
      current_index: 0,
    }
  }
}

impl From<CollectionEnum> for Collection {
  fn from(collection: CollectionEnum) -> Self {
    Self { collection }
  }
}

#[pyclass(unsendable, name = "InternalScallopCollectionIterator")]
pub struct CollectionIterator {
  collection: CollectionEnum,
  current_index: usize,
}

#[pymethods]
impl CollectionIterator {
  fn __next__(mut slf: PyRefMut<Self>) -> IterNextOutput<Py<PyAny>, &'static str> {
    if slf.current_index < slf.collection.len() {
      let i = slf.current_index;
      slf.current_index += 1;
      if slf.collection.has_empty_tag() {
        let tuple = to_python_tuple(slf.collection.ith_tuple(i));
        IterNextOutput::Yield(tuple)
      } else {
        let tuple = to_python_tuple(slf.collection.ith_tuple(i));
        let tag = slf.collection.ith_tag(i);
        let elem = Python::with_gil(|py| (tag, tuple).to_object(py));
        IterNextOutput::Yield(elem)
      }
    } else {
      IterNextOutput::Return("Ended")
    }
  }
}
