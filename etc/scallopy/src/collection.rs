use std::sync::Arc;

use pyo3::class::iter::IterNextOutput;
use pyo3::prelude::*;

use scallop_core::common;
use scallop_core::runtime::dynamic;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::ArcFamily;

use super::custom_tag;
use super::tuple::*;

#[derive(Clone)]
pub enum CollectionEnum {
  Unit {
    collection: Arc<dynamic::DynamicOutputCollection<unit::Unit>>,
  },
  Proofs {
    collection: Arc<dynamic::DynamicOutputCollection<proofs::Proofs>>,
  },
  MinMaxProb {
    collection: Arc<dynamic::DynamicOutputCollection<min_max_prob::Prob>>,
  },
  AddMultProb {
    collection: Arc<dynamic::DynamicOutputCollection<add_mult_prob::Prob>>,
  },
  TopKProofs {
    collection: Arc<dynamic::DynamicOutputCollection<top_k_proofs::Proofs<ArcFamily>>>,
    tags: Arc<Vec<f64>>,
  },
  TopBottomKClauses {
    collection: Arc<dynamic::DynamicOutputCollection<top_bottom_k_clauses::Formula<ArcFamily>>>,
    tags: Arc<Vec<f64>>,
  },
  DiffMinMaxProb {
    collection: Arc<dynamic::DynamicOutputCollection<diff_min_max_prob::Prob<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, diff_min_max_prob::Derivative<Py<PyAny>>)>>,
  },
  DiffAddMultProb {
    collection: Arc<dynamic::DynamicOutputCollection<diff_add_mult_prob::DiffProb<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<Py<PyAny>>>,
  },
  DiffSampleKProofs {
    collection: Arc<dynamic::DynamicOutputCollection<diff_sample_k_proofs::Proofs<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, Py<PyAny>)>>,
  },
  DiffTopKProofs {
    collection: Arc<dynamic::DynamicOutputCollection<diff_top_k_proofs::Proofs<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, Py<PyAny>)>>,
  },
  DiffTopBottomKClauses {
    collection: Arc<dynamic::DynamicOutputCollection<diff_top_bottom_k_clauses::Formula<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, Py<PyAny>)>>,
  },
  Custom {
    collection: Arc<dynamic::DynamicOutputCollection<custom_tag::CustomTag>>,
  },
}

impl CollectionEnum {
  pub fn unit(collection: Arc<dynamic::DynamicOutputCollection<unit::Unit>>) -> Self {
    Self::Unit { collection }
  }

  pub fn proofs(collection: Arc<dynamic::DynamicOutputCollection<proofs::Proofs>>) -> Self {
    Self::Proofs { collection }
  }

  pub fn min_max_prob(collection: Arc<dynamic::DynamicOutputCollection<min_max_prob::Prob>>) -> Self {
    Self::MinMaxProb { collection }
  }

  pub fn add_mult_prob(collection: Arc<dynamic::DynamicOutputCollection<add_mult_prob::Prob>>) -> Self {
    Self::AddMultProb { collection }
  }

  pub fn top_k_proofs(
    collection: Arc<dynamic::DynamicOutputCollection<top_k_proofs::Proofs<ArcFamily>>>,
    tags: Arc<Vec<f64>>,
  ) -> Self {
    Self::TopKProofs { collection, tags }
  }

  pub fn top_bottom_k_clauses(
    collection: Arc<dynamic::DynamicOutputCollection<top_bottom_k_clauses::Formula<ArcFamily>>>,
    tags: Arc<Vec<f64>>,
  ) -> Self {
    Self::TopBottomKClauses { collection, tags }
  }

  pub fn diff_min_max_prob(
    collection: Arc<dynamic::DynamicOutputCollection<diff_min_max_prob::Prob<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, diff_min_max_prob::Derivative<Py<PyAny>>)>>,
  ) -> Self {
    Self::DiffMinMaxProb { collection, tags }
  }

  pub fn diff_add_mult_prob(
    collection: Arc<dynamic::DynamicOutputCollection<diff_add_mult_prob::DiffProb<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<Py<PyAny>>>,
  ) -> Self {
    Self::DiffAddMultProb { collection, tags }
  }

  pub fn diff_sample_k_proofs(
    collection: Arc<dynamic::DynamicOutputCollection<diff_sample_k_proofs::Proofs<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, Py<PyAny>)>>,
  ) -> Self {
    Self::DiffSampleKProofs { collection, tags }
  }

  pub fn diff_top_k_proofs(
    collection: Arc<dynamic::DynamicOutputCollection<diff_top_k_proofs::Proofs<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, Py<PyAny>)>>,
  ) -> Self {
    Self::DiffTopKProofs { collection, tags }
  }

  pub fn diff_top_bottom_k_clauses(
    collection: Arc<dynamic::DynamicOutputCollection<diff_top_bottom_k_clauses::Formula<Py<PyAny>, ArcFamily>>>,
    tags: Arc<Vec<(f64, Py<PyAny>)>>,
  ) -> Self {
    Self::DiffTopBottomKClauses { collection, tags }
  }

  pub fn custom(collection: Arc<dynamic::DynamicOutputCollection<custom_tag::CustomTag>>) -> Self {
    Self::Custom { collection }
  }

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
      Self::DiffSampleKProofs { tags, .. } => Some(tags.len()),
      Self::DiffTopKProofs { tags, .. } => Some(tags.len()),
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
      Self::DiffSampleKProofs { tags, .. } => Some(tags.iter().map(|(_, t)| t.clone()).collect()),
      Self::DiffTopKProofs { tags, .. } => Some(tags.iter().map(|(_, t)| t.clone()).collect()),
      Self::DiffTopBottomKClauses { tags, .. } => Some(tags.iter().map(|(_, t)| t.clone()).collect()),
      Self::Custom { .. } => None,
    }
  }

  fn len(&self) -> usize {
    match self {
      Self::Unit { collection } => collection.len(),
      Self::Proofs { collection } => collection.len(),
      Self::MinMaxProb { collection } => collection.len(),
      Self::AddMultProb { collection } => collection.len(),
      Self::TopKProofs { collection, .. } => collection.len(),
      Self::TopBottomKClauses { collection, .. } => collection.len(),
      Self::DiffMinMaxProb { collection, .. } => collection.len(),
      Self::DiffAddMultProb { collection, .. } => collection.len(),
      Self::DiffSampleKProofs { collection, .. } => collection.len(),
      Self::DiffTopKProofs { collection, .. } => collection.len(),
      Self::DiffTopBottomKClauses { collection, .. } => collection.len(),
      Self::Custom { collection } => collection.len(),
    }
  }

  fn has_empty_tag(&self) -> bool {
    match self {
      Self::Unit { .. } => true,
      Self::Proofs { .. } => true,
      _ => false,
    }
  }

  fn ith_tuple(&self, i: usize) -> &common::tuple::Tuple {
    match self {
      Self::Unit { collection } => &collection.ith_tuple(i).unwrap(),
      Self::Proofs { collection } => &collection.ith_tuple(i).unwrap(),
      Self::MinMaxProb { collection } => &collection.ith_tuple(i).unwrap(),
      Self::AddMultProb { collection } => &collection.ith_tuple(i).unwrap(),
      Self::TopKProofs { collection, .. } => &collection.ith_tuple(i).unwrap(),
      Self::TopBottomKClauses { collection, .. } => &collection.ith_tuple(i).unwrap(),
      Self::DiffMinMaxProb { collection, .. } => &collection.ith_tuple(i).unwrap(),
      Self::DiffAddMultProb { collection, .. } => &collection.ith_tuple(i).unwrap(),
      Self::DiffSampleKProofs { collection, .. } => &collection.ith_tuple(i).unwrap(),
      Self::DiffTopKProofs { collection, .. } => &collection.ith_tuple(i).unwrap(),
      Self::DiffTopBottomKClauses { collection, .. } => &collection.ith_tuple(i).unwrap(),
      Self::Custom { collection, .. } => &collection.ith_tuple(i).unwrap(),
    }
  }

  fn ith_tag(&self, i: usize) -> Py<PyAny> {
    match self {
      Self::Unit { .. } => Python::with_gil(|py| ().to_object(py)),
      Self::Proofs { .. } => Python::with_gil(|py| ().to_object(py)),
      Self::MinMaxProb { collection } => Python::with_gil(|py| collection.ith_tag(i).unwrap().to_object(py)),
      Self::AddMultProb { collection } => Python::with_gil(|py| collection.ith_tag(i).unwrap().to_object(py)),
      Self::TopKProofs { collection, .. } => {
        let output_tag: f64 = *collection.ith_tag(i).unwrap();
        Python::with_gil(|py| output_tag.to_object(py))
      }
      Self::TopBottomKClauses { collection, .. } => {
        let output_tag: f64 = *collection.ith_tag(i).unwrap();
        Python::with_gil(|py| output_tag.to_object(py))
      }
      Self::DiffMinMaxProb { collection, .. } => {
        let output_tag = collection.ith_tag(i).unwrap();
        match output_tag.2 {
          1 => Python::with_gil(|py| (1, output_tag.3.clone().unwrap()).to_object(py)),
          0 => Python::with_gil(|py| (0, output_tag.0).to_object(py)),
          _ => Python::with_gil(|py| (-1, output_tag.3.clone().unwrap()).to_object(py)),
        }
      }
      Self::DiffAddMultProb { collection, .. } => {
        let tag = collection.ith_tag(i).unwrap();
        Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
      }
      Self::DiffSampleKProofs { collection, .. } => {
        let tag = &collection.ith_tag(i).unwrap();
        Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
      }
      Self::DiffTopKProofs { collection, .. } => {
        let output_tag = &collection.ith_tag(i).unwrap();
        Python::with_gil(|py| (output_tag.0, output_tag.1.clone()).to_object(py))
      }
      Self::DiffTopBottomKClauses { collection, .. } => {
        let output_tag = &collection.ith_tag(i).unwrap();
        Python::with_gil(|py| (output_tag.0, output_tag.1.clone()).to_object(py))
      }
      Self::Custom { collection } => collection.ith_tag(i).unwrap().clone(),
    }
  }

  fn ith_tag_to_string(&self, i: usize) -> String {
    match self {
      Self::Unit { .. } => "()".to_string(),
      Self::Proofs { collection } => format!("{}", collection.ith_tag(i).unwrap()),
      Self::MinMaxProb { collection } => format!("{}", collection.ith_tag(i).unwrap()),
      Self::AddMultProb { collection } => format!("{}", collection.ith_tag(i).unwrap()),
      Self::TopKProofs { collection, .. } => {
        let proofs = &collection.ith_tag(i).unwrap();
        format!("{}", proofs)
      }
      Self::TopBottomKClauses { collection, .. } => {
        let formula = &collection.ith_tag(i).unwrap();
        format!("{}", formula)
      }
      Self::DiffMinMaxProb { collection, .. } => {
        let id = collection.ith_tag(i).unwrap();
        format!("{}", id)
      }
      Self::DiffAddMultProb { collection, .. } => {
        let tag = &collection.ith_tag(i).unwrap();
        format!("{}", tag)
      }
      Self::DiffSampleKProofs { collection, .. } => {
        let tag = &collection.ith_tag(i).unwrap();
        format!("{}", tag)
      }
      Self::DiffTopKProofs { collection, .. } => {
        let tag = &collection.ith_tag(i).unwrap();
        format!("{}", tag)
      }
      Self::DiffTopBottomKClauses { collection, .. } => {
        let tag = &collection.ith_tag(i).unwrap();
        format!("{}", tag)
      }
      Self::Custom { collection } => format!("{}", collection.ith_tag(i).unwrap()),
    }
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

  fn debug_provenance(&self) {
    for i in 0..self.collection.len() {
      println!(
        "{}: {}",
        self.collection.ith_tuple(i),
        self.collection.ith_tag_to_string(i)
      );
    }
  }
}

impl From<CollectionEnum> for Collection {
  fn from(collection: CollectionEnum) -> Self {
    Self { collection }
  }
}

#[pyproto]
impl pyo3::PyIterProtocol for Collection {
  fn __iter__(slf: PyRef<Self>) -> CollectionIterator {
    CollectionIterator {
      collection: slf.collection.clone(),
      current_index: 0,
    }
  }
}

#[pyclass(unsendable, name = "InternalScallopCollectionIterator")]
pub struct CollectionIterator {
  collection: CollectionEnum,
  current_index: usize,
}

#[pyproto]
impl pyo3::PyIterProtocol for CollectionIterator {
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
