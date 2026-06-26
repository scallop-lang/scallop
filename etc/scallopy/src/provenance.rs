use std::sync::Arc;

use pyo3::exceptions::*;
use pyo3::types::*;
use pyo3::{prelude::*, Bound, IntoPyObjectExt};

use scallop_core::common::tuple::*;
use scallop_core::common::tuple_type::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::*;

use super::collection::*;
use super::custom_tag;
use super::external_tag::*;
use super::tuple::*;

/// The trait which all provenance contexts used in `scallopy` should implement
///
/// Contains the functions and default implementations for type conversion from and to python objects
pub trait PythonProvenance: Provenance {
  /// Process a list of python facts while the tuples are typed with `tuple_type`
  fn process_typed_py_facts(
    facts: &Bound<'_, PyList>,
    tuple_type: &TupleType,
    env: &RuntimeEnvironment,
  ) -> PyResult<Vec<(Option<Self::InputTag>, Tuple)>> {
    let facts: Vec<Bound<'_, PyAny>> = facts.extract()?;
    facts
      .into_iter()
      .map(|fact| {
        let (maybe_py_tag, py_tup) = Self::split_py_fact(&fact)?;
        let tag = Self::process_optional_py_tag(maybe_py_tag)?;
        let tup = from_python_tuple(&py_tup, tuple_type, &env.into())?;
        Ok((tag, tup))
      })
      .collect::<PyResult<Vec<_>>>()
  }

  /// Split a python object into (Option<PyTag>, PyTup) pair
  fn split_py_fact<'py>(fact: &Bound<'py, PyAny>) -> PyResult<(Option<Bound<'py, PyAny>>, Bound<'py, PyAny>)> {
    let (py_tag, py_tup) = fact.extract()?;
    Ok((py_tag, py_tup))
  }

  /// Convert a python object into an optional input tag for this provenance context
  fn process_optional_py_tag(maybe_py_tag: Option<Bound<'_, PyAny>>) -> PyResult<Option<Self::InputTag>> {
    if let Some(py_tag) = maybe_py_tag {
      Ok(Self::process_py_tag(&py_tag)?)
    } else {
      Ok(None)
    }
  }

  /// Convert a python object into an optional input tag for this provenance context
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>>;

  /// Convert an output collection into a python collection enum
  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily>;

  /// Convert an output tag into a python object
  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny>;
}

impl PythonProvenance for unit::UnitProvenance {
  fn split_py_fact<'py>(fact: &Bound<'py, PyAny>) -> PyResult<(Option<Bound<'py, PyAny>>, Bound<'py, PyAny>)> {
    Ok((None, fact.clone()))
  }

  fn process_py_tag(_: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    Ok(None)
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::Unit { collection: col }
  }

  fn to_output_py_tag(_: &Self::OutputTag) -> Py<PyAny> {
    Python::attach(|py| ().into_py_any(py).unwrap())
  }
}

impl PythonProvenance for proofs::ProofsProvenance<ArcFamily> {
  fn process_py_tag(disj_id: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let disj_id: usize = disj_id.extract()?;
    Ok(Some(Exclusion::Exclusive(disj_id)))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::Proofs { collection: col }
  }

  fn to_output_py_tag(proofs: &Self::OutputTag) -> Py<PyAny> {
    Python::attach(|py| {
      proofs
        .proofs
        .iter()
        .map(|proof| proof.facts.iter().collect::<Vec<_>>())
        .collect::<Vec<_>>()
        .into_py_any(py)
        .unwrap()
    })
  }
}

impl PythonProvenance for tropical::TropicalProvenance {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    tag.extract().map(Some)
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::Tropical { collection: col }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::attach(|py| tag.into_py_any(py).unwrap())
  }
}

impl PythonProvenance for min_max_prob::MinMaxProbProvenance {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    tag.extract().map(Some)
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::MinMaxProb { collection: col }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::attach(|py| tag.into_py_any(py).unwrap())
  }
}

impl PythonProvenance for add_mult_prob::AddMultProbProvenance {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    tag.extract().map(Some)
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::AddMultProb { collection: col }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::attach(|py| tag.into_py_any(py).unwrap())
  }
}

impl PythonProvenance for top_k_proofs::TopKProofsProvenance<ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
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
    Python::attach(|py| tag.into_py_any(py).unwrap())
  }
}

impl PythonProvenance for top_bottom_k_clauses::TopBottomKClausesProvenance<ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
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
    Python::attach(|py| tag.into_py_any(py).unwrap())
  }
}

impl PythonProvenance for diff_min_max_prob::DiffMinMaxProbProvenance<ExtTag, ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let prob: f64 = tag.extract()?;
    let tag: ExtTag = tag.into();
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
      1 => Python::attach(|py| (1, tag.3.as_ref().into_option().unwrap()).into_py_any(py).unwrap()),
      0 => Python::attach(|py| (0, tag.0).into_py_any(py).unwrap()),
      _ => Python::attach(|py| (-1, tag.3.as_ref().into_option().unwrap()).into_py_any(py).unwrap()),
    }
  }
}

impl PythonProvenance for diff_add_mult_prob::DiffAddMultProbProvenance<ExtTag, ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let prob: f64 = tag.extract()?;
    let tag: ExtTag = tag.into();
    Ok(Some((prob, Some(tag)).into()))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffAddMultProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::attach(|py| (tag.0, tag.1.clone()).into_py_any(py).unwrap())
  }
}

impl PythonProvenance for diff_nand_mult_prob::DiffNandMultProbProvenance<ExtTag, ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let prob: f64 = tag.extract()?;
    let tag: ExtTag = tag.into();
    Ok(Some((prob, Some(tag)).into()))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffNandMultProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::attach(|py| (tag.0, tag.1.clone()).into_py_any(py).unwrap())
  }
}

impl PythonProvenance for diff_max_mult_prob::DiffMaxMultProbProvenance<ExtTag, ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let prob: f64 = tag.extract()?;
    let tag: ExtTag = tag.into();
    Ok(Some((prob, Some(tag)).into()))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffMaxMultProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::attach(|py| (tag.0, tag.1.clone()).into_py_any(py).unwrap())
  }
}

impl PythonProvenance for diff_nand_min_prob::DiffNandMinProbProvenance<ExtTag, ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let prob: f64 = tag.extract()?;
    let tag: ExtTag = tag.into();
    Ok(Some((prob, Some(tag)).into()))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffNandMinProb {
      collection: col,
      tags: ctx.storage.clone(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::attach(|py| (tag.0, tag.1.clone()).into_py_any(py).unwrap())
  }
}

impl PythonProvenance for diff_sample_k_proofs::DiffSampleKProofsProvenance<ExtTag, ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let tag_disj_id: (Bound<'_, PyAny>, Option<usize>) = tag.extract()?;
    if let Some(prob) = tag_disj_id.0.extract()? {
      let tag: ExtTag = (&tag_disj_id.0).into();
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
    Python::attach(|py| (tag.0, tag.1.clone()).into_py_any(py).unwrap())
  }
}

impl PythonProvenance for diff_top_k_proofs::DiffTopKProofsProvenance<ExtTag, ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let tag_disj_id: (Bound<'_, PyAny>, Option<usize>) = tag.extract()?;
    if let Some(prob) = tag_disj_id.0.extract()? {
      let tag: ExtTag = (&tag_disj_id.0).into();
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
    Python::attach(|py| (tag.0, tag.1.clone()).into_py_any(py).unwrap())
  }
}

impl PythonProvenance for diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance<ExtTag, ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let tag_disj_id: (Bound<'_, PyAny>, Option<usize>) = tag.extract()?;
    if let Some(prob) = tag_disj_id.0.extract()? {
      let tag: ExtTag = (&tag_disj_id.0).into();
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
    Python::attach(|py| (tag.0, tag.1.clone()).into_py_any(py).unwrap())
  }
}

impl PythonProvenance for diff_top_k_proofs_debug::DiffTopKProofsDebugProvenance<ExtTag, ArcFamily> {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let tag_tuple = tag.cast::<PyTuple>()?;
    let prob = tag_tuple.get_item(0)?;
    if let Some(prob) = prob.extract()? {
      let tag_disj_id: (Bound<'_, PyAny>, usize, Option<usize>) = tag.extract()?;
      let tag: ExtTag = (&tag_disj_id.0).into();
      let id: usize = tag_disj_id.1.into();
      if id == 0 {
        Err(PyErr::new::<PyValueError, _>("The input ID to the diff-top-k-proofs-debug provenance cannot be 0; consider changing it to starting from 1."))
      } else {
        Ok(Some(Self::InputTag {
          prob,
          id,
          external_tag: Some(tag),
          exclusion: tag_disj_id.2,
        }))
      }
    } else {
      Ok(None)
    }
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, ctx: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::DiffTopKProofsDebug {
      collection: col,
      tags: ctx.storage.clone_rc(),
    }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    Python::attach(|py| {
      (tag.probability, tag.gradient.clone(), tag.proofs.clone())
        .into_py_any(py)
        .unwrap()
    })
  }
}

impl PythonProvenance for custom_tag::CustomProvenance {
  fn process_py_tag(tag: &Bound<'_, PyAny>) -> PyResult<Option<Self::InputTag>> {
    let tag: Py<PyAny> = tag.clone().unbind();
    Ok(Some(tag))
  }

  fn to_collection_enum(col: Arc<DynamicOutputCollection<Self>>, _: &Self) -> CollectionEnum<ArcFamily> {
    CollectionEnum::Custom { collection: col }
  }

  fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
    tag.clone()
  }
}
