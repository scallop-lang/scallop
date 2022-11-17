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
      Ok(Some(Self::process_py_tag(py_tag)?))
    } else {
      Ok(None)
    }
  }

  /// Convert a python object into an optional input tag for this provenance context
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag>;

  /// Convert an output collection into a python collection enum
  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum;

  /// Convert an output tag into a python object
  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny>;
}

impl PythonProvenance for unit::UnitProvenance {
  fn split_py_fact(fact: &PyAny) -> PyResult<(Option<&PyAny>, &PyAny)> {
    Ok((None, fact))
  }

  fn process_py_tag(_: &PyAny) -> PyResult<Self::InputTag> {
    Ok(())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum {
    CollectionEnum::Unit { collection: col }
  }

  fn to_output_py_tag(_: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| ().to_object(py))
  }
}

impl PythonProvenance for proofs::ProofsProvenance {
  fn split_py_fact(fact: &PyAny) -> PyResult<(Option<&PyAny>, &PyAny)> {
    Ok((None, fact))
  }

  fn process_py_tag(_: &PyAny) -> PyResult<Self::InputTag> {
    Ok(())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum {
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
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    tag.extract()
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum {
    CollectionEnum::MinMaxProb { collection: col }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| tag.to_object(py))
  }
}

impl PythonProvenance for add_mult_prob::AddMultProbContext {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    tag.extract()
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum {
    CollectionEnum::AddMultProb { collection: col }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| tag.to_object(py))
  }
}

impl PythonProvenance for top_k_proofs::TopKProofsContext<ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    tag.extract()
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::TopKProofs {
      collection: col.clone(),
      tags: ctx.probs.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| tag.to_object(py))
  }
}

impl PythonProvenance for top_bottom_k_clauses::TopBottomKClausesContext<ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    tag.extract()
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::TopBottomKClauses {
      collection: col.clone(),
      tags: ctx.probs.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| tag.to_object(py))
  }
}

impl PythonProvenance for diff_min_max_prob::DiffMinMaxProbContext<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok((prob, tag).into())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::DiffMinMaxProb {
      collection: col,
      tags: ctx.diff_probs.clone(),
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

impl PythonProvenance for diff_add_mult_prob::DiffAddMultProbContext<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok((prob, tag).into())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::DiffAddMultProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_nand_mult_prob::DiffNandMultProbContext<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok((prob, tag).into())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::DiffNandMultProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_max_mult_prob::DiffMaxMultProbContext<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok((prob, tag).into())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::DiffMaxMultProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_nand_min_prob::DiffNandMinProbContext<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok((prob, tag).into())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::DiffNandMinProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_sample_k_proofs::DiffSampleKProofsContext<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok((prob, tag).into())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::DiffSampleKProofs {
      collection: col,
      tags: ctx.diff_probs.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_top_k_proofs::DiffTopKProofsContext<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok((prob, tag).into())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::DiffTopKProofs {
      collection: col,
      tags: ctx.diff_probs.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_top_k_proofs_indiv::DiffTopKProofsIndivContext<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok((prob, tag).into())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::DiffTopKProofsIndiv {
      collection: col,
      tags: ctx.diff_probs.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.k, tag.proofs.clone()).to_object(py))
  }
}

impl PythonProvenance for diff_top_bottom_k_clauses::DiffTopBottomKClausesContext<Py<PyAny>, ArcFamily> {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    let prob: f64 = tag.extract()?;
    let tag: Py<PyAny> = tag.into();
    Ok((prob, tag).into())
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum {
    CollectionEnum::DiffTopBottomKClauses {
      collection: col,
      tags: ctx.diff_probs.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
  }
}

impl PythonProvenance for custom_tag::CustomTagContext {
  fn process_py_tag(tag: &PyAny) -> PyResult<Self::InputTag> {
    let tag: Py<PyAny> = tag.into();
    Ok(tag)
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum {
    CollectionEnum::Custom { collection: col }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    tag.clone()
  }
}
