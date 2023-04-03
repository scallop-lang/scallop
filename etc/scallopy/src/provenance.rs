use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::*;

use scallop_core::common::tuple::*;
use scallop_core::common::tuple_type::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::*;

use super::collection::*;
use super::custom_tag;
use super::tuple::*;

/// The trait which all provenance contexts used in `scallopy` should implement
///
/// Contains the functions and default implementations for type conversion from and to python objects
pub trait PythonProvenance: Provenance {
  /// Process a list of python facts while the tuples are typed with `tuple_type`
  fn process_typed_py_facts(facts: &PyList, tuple_type: &TupleType) -> PyResult<Vec<(Option<Self::InputTag>, Tuple)>> {
    let facts: Vec<&PyAny> = facts.extract()?;
    facts
      .into_iter()
      .map(|fact| {
        let (maybe_py_tag, py_tup) = Self::split_py_fact(fact)?;
        let tag = Self::process_optional_py_tag(maybe_py_tag)?;
        let tup = from_python_tuple(py_tup, tuple_type)?;
        Ok((tag, tup))
      })
      .collect::<PyResult<Vec<_>>>()
  }

  /// Split a python object into (Option<PyTag>, PyTup) pair
  fn split_py_fact(fact: &PyAny) -> PyResult<(Option<&PyAny>, &PyAny)> {
    let (py_tag, py_tup) = fact.extract()?;
    Ok((py_tag, py_tup))
  }

  /// Convert a python object into an optional input tag for this provenance context
  fn process_optional_py_tag(maybe_py_tag: Option<&PyAny>) -> PyResult<Option<Self::InputTag>> {
    if let Some(py_tag) = maybe_py_tag {
      Ok(Self::process_py_tag(py_tag)?)
    } else {
      Ok(None)
    }
  }

  /// Convert a python object into an optional input tag for this provenance context
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>>;

  /// Convert an output collection into a python collection enum
  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily>;

  /// Convert an output tag into a python object
  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny>;
}

impl PythonProvenance for unit::UnitProvenance {
  fn split_py_fact(fact: &PyAny) -> PyResult<(Option<&PyAny>, &PyAny)> {
    Ok((None, fact))
  }

  fn process_py_tag(_: &PyAny) -> PyResult<Option<Self::InputTag>> {
    Ok(None)
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::Unit { collection: col }
  }

  fn to_output_py_tag(_: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| ().to_object(py))
  }
}

impl PythonProvenance for proofs::ProofsProvenance<ArcFamily> {
  fn process_py_tag(disj_id: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let disj_id: usize = disj_id.extract()?;
    Ok(Some(proofs::ProofsInputTag::Exclusive(disj_id)))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::Proofs { collection: col }
  }

  fn to_output_py_tag(proofs: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| {
      proofs
        .proofs
        .iter()
        .map(|proof| proof.facts.iter().collect::<Vec<_>>())
        .collect::<Vec<_>>()
        .to_object(py)
    })
  }
}

impl PythonProvenance for min_max_prob::MinMaxProbProvenance {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    tag.extract().map(Some)
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::MinMaxProb { collection: col }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| tag.to_object(py))
  }
}

impl PythonProvenance for add_mult_prob::AddMultProbProvenance {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    tag.extract().map(Some)
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::AddMultProb { collection: col }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| tag.to_object(py))
  }
}

impl PythonProvenance for top_k_proofs::TopKProofsProvenance<ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let (prob, disj_id): (Option<f64>, Option<usize>) = tag.extract()?;
    if let Some(prob) = prob {
      Ok(Some((prob, disj_id).into()))
    } else {
      Ok(None)
    }
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::TopKProofs {
      collection: col.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| tag.to_object(py))
  }
}

impl PythonProvenance for top_bottom_k_clauses::TopBottomKClausesProvenance<ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let (prob, disj_id): (Option<f64>, Option<usize>) = tag.extract()?;
    if let Some(prob) = prob {
      Ok(Some((prob, disj_id).into()))
    } else {
      Ok(None)
    }
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::TopBottomKClauses {
      collection: col.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| tag.to_object(py))
  }
}

impl PythonProvenance for diff_min_max_prob::DiffMinMaxProbProvenance<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok(Some((prob, Some(tag)).into()))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffMinMaxProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    match tag.2 {
      1 => Python::with_gil(|py| (1, tag.3.clone().unwrap()).to_object(py)),
      0 => Python::with_gil(|py| (0, tag.0).to_object(py)),
      _ => Python::with_gil(|py| (-1, tag.3.clone().unwrap()).to_object(py)),
    }
  }
}

impl PythonProvenance for diff_add_mult_prob::DiffAddMultProbProvenance<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok(Some((prob, Some(tag)).into()))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffAddMultProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_nand_mult_prob::DiffNandMultProbProvenance<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok(Some((prob, Some(tag)).into()))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffNandMultProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_max_mult_prob::DiffMaxMultProbProvenance<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok(Some((prob, Some(tag)).into()))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffMaxMultProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_nand_min_prob::DiffNandMinProbProvenance<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok(Some((prob, Some(tag)).into()))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffNandMinProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_sample_k_proofs::DiffSampleKProofsProvenance<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let tag_disj_id: (&PyAny, Option<usize>) = tag.extract()?;
    if let Some(prob) = tag_disj_id.0.extract()? {
      let tag: Py<PyAny> = tag_disj_id.0.into();
      Ok(Some((prob, tag, tag_disj_id.1).into()))
    } else {
      Ok(None)
    }
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffSampleKProofs {
      collection: col,
      tags: ctx.storage.clone_rc(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_top_k_proofs::DiffTopKProofsProvenance<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let tag_disj_id: (&PyAny, Option<usize>) = tag.extract()?;
    if let Some(prob) = tag_disj_id.0.extract()? {
      let tag: Py<PyAny> = tag_disj_id.0.into();
      Ok(Some((prob, tag, tag_disj_id.1).into()))
    } else {
      Ok(None)
    }
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffTopKProofs {
      collection: col,
      tags: ctx.storage.clone_rc(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_top_k_proofs_indiv::DiffTopKProofsIndivProvenance<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let tag_disj_id: (&PyAny, Option<usize>) = tag.extract()?;
    if let Some(prob) = tag_disj_id.0.extract()? {
      let tag: Py<PyAny> = tag_disj_id.0.into();
      Ok(Some((prob, tag, tag_disj_id.1).into()))
    } else {
      Ok(None)
    }
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffTopKProofsIndiv {
      collection: col,
      tags: ctx.storage.clone_rc(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.k, tag.proofs.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let tag_disj_id: (&PyAny, Option<usize>) = tag.extract()?;
    if let Some(prob) = tag_disj_id.0.extract()? {
      let tag: Py<PyAny> = tag_disj_id.0.into();
      Ok(Some((prob, tag, tag_disj_id.1).into()))
    } else {
      Ok(None)
    }
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffTopBottomKClauses {
      collection: col,
      tags: ctx.storage.clone_rc(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for custom_tag::CustomProvenance {
  fn process_py_tag(tag: &PyAny) -> PyResult<Option<Self::InputTag>> {
    let tag: Py<PyAny> = tag.into();
    Ok(Some(tag))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::Custom { collection: col }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    tag.clone()
  }
}
