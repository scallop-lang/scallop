use std::collections::*;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyTuple;

use rayon::prelude::*;

use scallop_core::common::tuple::Tuple;
use scallop_core::common::tuple_type::TupleType;
use scallop_core::integrate::IntegrateContext;
use scallop_core::integrate::{Attribute, AttributeArgument};
use scallop_core::runtime::dynamic::DynamicOutputCollection;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::ArcFamily;

use crate::custom_tag;
use crate::wmc::WMCType;

use super::collection::*;
use super::error::*;
use super::io::*;
use super::tuple::*;

#[pyclass(unsendable, name = "InternalScallopContext")]
pub struct Context {
  pub ctx: ContextEnum,
}

#[pymethods]
impl Context {
  #[new]
  #[args(
    provenance = "\"unit\"",
    k = "3",
    custom_provenance = "None",
    wmc_type = "\"bottom-up\""
  )]
  fn new(
    provenance: &str,
    k: usize,
    custom_provenance: Option<Py<PyAny>>,
    wmc_type: &str,
  ) -> Result<Self, BindingError> {
    // Check provenance type
    match provenance {
      "unit" => Ok(Self {
        ctx: ContextEnum::Unit(IntegrateContext::new(unit::UnitContext::default())),
      }),
      "proofs" => Ok(Self {
        ctx: ContextEnum::Proofs(IntegrateContext::new(proofs::ProofsContext::default())),
      }),
      "minmaxprob" => Ok(Self {
        ctx: ContextEnum::MinMaxProb(IntegrateContext::new(
          min_max_prob::MinMaxProbContext::default(),
        )),
      }),
      "addmultprob" => Ok(Self {
        ctx: ContextEnum::AddMultProb(IntegrateContext::new(
          add_mult_prob::AddMultProbContext::default(),
        )),
      }),
      "topkproofs" => Ok(Self {
        ctx: ContextEnum::TopKProofs(IntegrateContext::new(top_k_proofs::TopKProofsContext::new(
          k,
        ))),
      }),
      "topbottomkclauses" => Ok(Self {
        ctx: ContextEnum::TopBottomKClauses(IntegrateContext::new(
          top_bottom_k_clauses::TopBottomKClausesContext::new(k),
        )),
      }),
      "diffminmaxprob" => Ok(Self {
        ctx: ContextEnum::DiffMinMaxProb(IntegrateContext::new(
          diff_min_max_prob::DiffMinMaxProbContext::default(),
        )),
      }),
      "diffaddmultprob" => Ok(Self {
        ctx: ContextEnum::DiffAddMultProb(IntegrateContext::new(
          diff_add_mult_prob::DiffAddMultProbContext::default(),
        )),
      }),
      "diffsamplekproofs" => Ok(Self {
        ctx: ContextEnum::DiffSampleKProofs(IntegrateContext::new(
          diff_sample_k_proofs::DiffSampleKProofsContext::new(k),
        )),
      }),
      "difftopkproofs" => Ok(Self {
        ctx: ContextEnum::DiffTopKProofs(
          IntegrateContext::new(diff_top_k_proofs::DiffTopKProofsContext::new(k)),
          WMCType::from(wmc_type),
        ),
      }),
      "difftopbottomkclauses" => Ok(Self {
        ctx: ContextEnum::DiffTopBottomKClauses(IntegrateContext::new(
          diff_top_bottom_k_clauses::DiffTopBottomKClausesContext::new(k),
        )),
      }),
      "custom" => {
        if let Some(py_ctx) = custom_provenance {
          Ok(Self {
            ctx: ContextEnum::Custom(IntegrateContext::new(custom_tag::CustomTagContext(py_ctx))),
          })
        } else {
          Err(BindingError::InvalidCustomProvenance)
        }
      }
      p => Err(BindingError::UnknownProvenance(p.to_string())),
    }
  }

  fn clone(&self) -> Self {
    Self {
      ctx: self.ctx.clone(),
    }
  }

  fn compile(&mut self) -> Result<(), BindingError> {
    match &mut self.ctx {
      ContextEnum::Unit(c) => c.compile().map_err(BindingError::from),
      ContextEnum::Proofs(c) => c.compile().map_err(BindingError::from),
      ContextEnum::MinMaxProb(c) => c.compile().map_err(BindingError::from),
      ContextEnum::AddMultProb(c) => c.compile().map_err(BindingError::from),
      ContextEnum::TopKProofs(c) => c.compile().map_err(BindingError::from),
      ContextEnum::TopBottomKClauses(c) => c.compile().map_err(BindingError::from),
      ContextEnum::DiffMinMaxProb(c) => c.compile().map_err(BindingError::from),
      ContextEnum::DiffAddMultProb(c) => c.compile().map_err(BindingError::from),
      ContextEnum::DiffSampleKProofs(c) => c.compile().map_err(BindingError::from),
      ContextEnum::DiffTopKProofs(c, _) => c.compile().map_err(BindingError::from),
      ContextEnum::DiffTopBottomKClauses(c) => c.compile().map_err(BindingError::from),
      ContextEnum::Custom(c) => c.compile().map_err(BindingError::from),
    }
  }

  fn import_file(&mut self, file_name: &str) -> Result<(), BindingError> {
    match &mut self.ctx {
      ContextEnum::Unit(c) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::Proofs(c) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::MinMaxProb(c) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::AddMultProb(c) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::TopKProofs(c) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::TopBottomKClauses(c) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::DiffMinMaxProb(c) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::DiffAddMultProb(c) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::DiffSampleKProofs(c) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::DiffTopKProofs(c, _) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::DiffTopBottomKClauses(c) => c.import_file(file_name).map_err(BindingError::from),
      ContextEnum::Custom(c) => c.import_file(file_name).map_err(BindingError::from),
    }
  }

  fn dump_front_ir(&self) {
    match &self.ctx {
      ContextEnum::Unit(c) => c.dump_front_ir(),
      ContextEnum::Proofs(c) => c.dump_front_ir(),
      ContextEnum::MinMaxProb(c) => c.dump_front_ir(),
      ContextEnum::AddMultProb(c) => c.dump_front_ir(),
      ContextEnum::TopKProofs(c) => c.dump_front_ir(),
      ContextEnum::TopBottomKClauses(c) => c.dump_front_ir(),
      ContextEnum::DiffMinMaxProb(c) => c.dump_front_ir(),
      ContextEnum::DiffAddMultProb(c) => c.dump_front_ir(),
      ContextEnum::DiffSampleKProofs(c) => c.dump_front_ir(),
      ContextEnum::DiffTopKProofs(c, _) => c.dump_front_ir(),
      ContextEnum::DiffTopBottomKClauses(c) => c.dump_front_ir(),
      ContextEnum::Custom(c) => c.dump_front_ir(),
    }
  }

  fn input_tags(&self) -> Option<Vec<Py<PyAny>>> {
    match &self.ctx {
      ContextEnum::Unit(_) => None,
      ContextEnum::Proofs(_) => None,
      ContextEnum::MinMaxProb(_) => None,
      ContextEnum::AddMultProb(_) => None,
      ContextEnum::TopKProofs(_) => None,
      ContextEnum::TopBottomKClauses(_) => None,
      ContextEnum::DiffMinMaxProb(_) => None,
      ContextEnum::DiffAddMultProb(c) => Some(c.provenance_context().input_tags()),
      ContextEnum::DiffSampleKProofs(c) => Some(c.provenance_context().input_tags()),
      ContextEnum::DiffTopKProofs(c, _) => Some(c.provenance_context().input_tags()),
      ContextEnum::DiffTopBottomKClauses(c) => Some(c.provenance_context().input_tags()),
      ContextEnum::Custom(_) => None,
    }
  }

  fn set_k(&mut self, k: usize) {
    match &mut self.ctx {
      ContextEnum::Unit(_) => (),
      ContextEnum::Proofs(_) => (),
      ContextEnum::MinMaxProb(_) => (),
      ContextEnum::AddMultProb(_) => (),
      ContextEnum::TopKProofs(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::TopBottomKClauses(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::DiffMinMaxProb(_) => (),
      ContextEnum::DiffAddMultProb(_) => (),
      ContextEnum::DiffSampleKProofs(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::DiffTopKProofs(c, _) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::DiffTopBottomKClauses(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::Custom(_) => (),
    }
  }

  #[args(load_csv = "None", demand = "None")]
  fn add_relation(
    &mut self,
    relation: &str,
    load_csv: Option<&PyAny>,
    demand: Option<String>,
  ) -> Result<String, BindingError> {
    // Get the attributes
    let mut attrs = Vec::new();

    // Load the load csv file
    if let Some(obj) = load_csv {
      if let Ok(csv_file) = obj.extract::<CSVFileOptions>() {
        attrs.push(csv_file.into());
      } else if let Ok(path) = obj.extract::<String>() {
        attrs.push(CSVFileOptions::new(path).into());
      } else {
        return Err(BindingError::InvalidLoadCSVArg);
      }
    }

    // Demand attribute
    if let Some(d) = demand {
      attrs.push(Attribute {
        name: "demand".to_string(),
        positional_arguments: vec![AttributeArgument::String(d)],
        keyword_arguments: HashMap::new(),
      })
    }

    // Add relation
    let rtd = match &mut self.ctx {
      ContextEnum::Unit(c) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::Proofs(c) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::MinMaxProb(c) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::AddMultProb(c) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::TopKProofs(c) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::TopBottomKClauses(c) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::DiffMinMaxProb(c) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::DiffAddMultProb(c) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::DiffSampleKProofs(c) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::DiffTopKProofs(c, _) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::DiffTopBottomKClauses(c) => c.add_relation_with_attributes(relation, attrs)?,
      ContextEnum::Custom(c) => c.add_relation_with_attributes(relation, attrs)?,
    };

    // Return
    Ok(rtd.predicate().to_string())
  }

  fn add_facts(
    &mut self,
    relation: &str,
    elems: &PyList,
    disjunctions: Option<Vec<Vec<usize>>>,
  ) -> Result<(), BindingError> {
    match &mut self.ctx {
      ContextEnum::Unit(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_unit_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::Proofs(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_unit_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::MinMaxProb(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_prob_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::AddMultProb(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_prob_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::TopKProofs(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_prob_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::TopBottomKClauses(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_prob_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::DiffMinMaxProb(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_diff_prob_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::DiffAddMultProb(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_diff_prob_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::DiffSampleKProofs(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_diff_prob_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::DiffTopKProofs(c, _) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_diff_prob_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::DiffTopBottomKClauses(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_diff_prob_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
      ContextEnum::Custom(c) => {
        if let Some(tuple_type) = c.relation_type(relation) {
          let tuples = process_custom_facts(elems, tuple_type)?;
          c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
          Ok(())
        } else {
          Err(BindingError::UnknownRelation(relation.to_string()).into())
        }
      }
    }
  }

  #[args(tag = "None", demand = "None")]
  fn add_rule(
    &mut self,
    rule: &str,
    tag: Option<&PyAny>,
    demand: Option<String>,
  ) -> Result<(), BindingError> {
    // Attributes
    let mut attrs = Vec::new();

    // Demand attribute
    if let Some(d) = demand {
      attrs.push(Attribute {
        name: "demand".to_string(),
        positional_arguments: vec![AttributeArgument::String(d)],
        keyword_arguments: HashMap::new(),
      })
    }

    match &mut self.ctx {
      ContextEnum::Unit(c) => c
        .add_rule_with_attributes(rule, attrs)
        .map(|_| ())
        .map_err(BindingError::IntegrateError),
      ContextEnum::Proofs(c) => c
        .add_rule_with_attributes(rule, attrs)
        .map(|_| ())
        .map_err(BindingError::IntegrateError),
      ContextEnum::MinMaxProb(c) => {
        // Get the tag
        let tag = if let Some(prov) = tag {
          let prob: Option<f64> = prov.extract()?;
          prob
        } else {
          None
        };

        // Add the rule
        c.add_rule_with_options(rule, tag, attrs)?;

        Ok(())
      }
      ContextEnum::AddMultProb(c) => {
        // Get the tag
        let tag = if let Some(prov) = tag {
          let prob: Option<f64> = prov.extract()?;
          prob
        } else {
          None
        };

        // Add the rule
        c.add_rule_with_options(rule, tag, attrs)?;

        Ok(())
      }
      ContextEnum::TopKProofs(c) => {
        // Get the tag
        let tag = if let Some(prov) = tag {
          let prob: Option<f64> = prov.extract()?;
          prob
        } else {
          None
        };

        // Add the rule
        c.add_rule_with_options(rule, tag, attrs)?;

        Ok(())
      }
      ContextEnum::TopBottomKClauses(c) => {
        // Get the tag
        let tag = if let Some(prov) = tag {
          let prob: Option<f64> = prov.extract()?;
          prob
        } else {
          None
        };

        // Add the rule
        c.add_rule_with_options(rule, tag, attrs)?;

        Ok(())
      }
      ContextEnum::DiffMinMaxProb(c) => {
        // Get the tag
        let tag = if let Some(prov) = tag {
          let tag: Py<PyAny> = prov.into();
          let prob: f64 = prov.extract()?;
          Some((prob, tag).into())
        } else {
          None
        };

        // Add the rule
        c.add_rule_with_options(rule, tag, attrs)?;

        Ok(())
      }
      ContextEnum::DiffAddMultProb(c) => {
        // Get the tag
        let tag = if let Some(prov) = tag {
          let tag: Py<PyAny> = prov.into();
          let prob: f64 = prov.extract()?;
          Some((prob, tag).into())
        } else {
          None
        };

        // Add the rule
        c.add_rule_with_options(rule, tag, attrs)?;

        Ok(())
      }
      ContextEnum::DiffSampleKProofs(c) => {
        // Get the tag
        let tag = if let Some(prov) = tag {
          let tag: Py<PyAny> = prov.into();
          let prob: f64 = prov.extract()?;
          Some((prob, tag).into())
        } else {
          None
        };

        // Add the rule
        c.add_rule_with_options(rule, tag, attrs)?;

        Ok(())
      }
      ContextEnum::DiffTopKProofs(c, _) => {
        // Get the tag
        let tag = if let Some(prov) = tag {
          let tag: Py<PyAny> = prov.into();
          let prob: f64 = prov.extract()?;
          Some((prob, tag).into())
        } else {
          None
        };

        // Add the rule
        c.add_rule_with_options(rule, tag, attrs)?;

        Ok(())
      }
      ContextEnum::DiffTopBottomKClauses(c) => {
        // Get the tag
        let tag = if let Some(prov) = tag {
          let tag: Py<PyAny> = prov.into();
          let prob: f64 = prov.extract()?;
          Some((prob, tag).into())
        } else {
          None
        };

        // Add the rule
        c.add_rule_with_options(rule, tag, attrs)?;

        Ok(())
      }
      ContextEnum::Custom(c) => {
        // Get the tag
        let tag = if let Some(prov) = tag {
          let tag: Py<PyAny> = prov.into();
          Some(tag)
        } else {
          None
        };

        // Add the rule
        c.add_rule_with_options(rule, tag, attrs)?;

        Ok(())
      }
    }
  }

  fn run(&mut self, iter_limit: Option<usize>) -> Result<(), BindingError> {
    match &mut self.ctx {
      ContextEnum::Unit(c) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::Proofs(c) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::MinMaxProb(c) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::AddMultProb(c) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::TopKProofs(c) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::TopBottomKClauses(c) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::DiffMinMaxProb(c) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::DiffAddMultProb(c) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::DiffSampleKProofs(c) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::DiffTopKProofs(c, _) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::DiffTopBottomKClauses(c) => c.run_with_iter_limit(iter_limit)?,
      ContextEnum::Custom(c) => c.run_with_iter_limit(iter_limit)?,
    }
    Ok(())
  }

  fn run_batch(
    &self,
    iter_limit: Option<usize>,
    output_relations: Vec<&str>,
    inputs: HashMap<String, Vec<(&PyList, Option<Vec<Vec<usize>>>)>>,
    parallel: bool,
  ) -> Result<Vec<Collection>, BindingError> {
    // Sanity check: has input
    if inputs.is_empty() {
      return Err(BindingError::EmptyBatchInput);
    }

    // Sanity check: all input facts share the same batch size
    if !is_all_equal(inputs.iter().map(|(_, batch)| batch.len())) {
      return Err(BindingError::InvalidBatchSize);
    }

    // Sanity check: non-empty batch size
    if inputs.iter().next().unwrap().1.is_empty() {
      return Err(BindingError::InvalidBatchSize);
    }

    // Depending on whether we want parallelism, choose the subsequent function to call
    if parallel {
      self.run_batch_parallel(iter_limit, output_relations, inputs)
    } else {
      self.run_batch_serial(iter_limit, output_relations, inputs)
    }
  }

  fn relation(&mut self, r: &str) -> Result<Collection, BindingError> {
    match &mut self.ctx {
      ContextEnum::Unit(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(CollectionEnum::unit(collection).into())
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::Proofs(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(CollectionEnum::proofs(collection).into())
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::MinMaxProb(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(CollectionEnum::min_max_prob(collection).into())
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::AddMultProb(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(CollectionEnum::add_mult_prob(collection).into())
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::TopKProofs(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(
              CollectionEnum::TopKProofs {
                collection,
                tags: c.provenance_context().probs.clone(),
              }
              .into(),
            )
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::TopBottomKClauses(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(
              CollectionEnum::TopBottomKClauses {
                collection,
                tags: c.provenance_context().probs.clone(),
              }
              .into(),
            )
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::DiffMinMaxProb(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(
              CollectionEnum::DiffMinMaxProb {
                collection,
                tags: c.provenance_context().diff_probs.clone(),
              }
              .into(),
            )
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::DiffAddMultProb(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(
              CollectionEnum::DiffAddMultProb {
                collection,
                tags: c.provenance_context().storage.clone(),
              }
              .into(),
            )
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::DiffSampleKProofs(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(
              CollectionEnum::DiffSampleKProofs {
                collection,
                tags: c.provenance_context().diff_probs.clone(),
              }
              .into(),
            )
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::DiffTopKProofs(c, wmc_type) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(
              CollectionEnum::DiffTopKProofs {
                collection,
                tags: c.provenance_context().diff_probs.clone(),
                wmc_type: wmc_type.clone(),
              }
              .into(),
            )
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::DiffTopBottomKClauses(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(
              CollectionEnum::DiffTopBottomKClauses {
                collection,
                tags: c.provenance_context().diff_probs.clone(),
              }
              .into(),
            )
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
      ContextEnum::Custom(c) => {
        if c.has_relation(r) {
          if let Some(collection) = c.computed_rc_relation(r) {
            Ok(CollectionEnum::Custom { collection }.into())
          } else {
            Err(BindingError::RelationNotComputed(r.to_string()))
          }
        } else {
          Err(BindingError::UnknownRelation(r.to_string()))
        }
      }
    }
  }

  fn has_relation(&self, r: &str) -> bool {
    match &self.ctx {
      ContextEnum::Unit(c) => c.has_relation(r),
      ContextEnum::Proofs(c) => c.has_relation(r),
      ContextEnum::MinMaxProb(c) => c.has_relation(r),
      ContextEnum::AddMultProb(c) => c.has_relation(r),
      ContextEnum::TopKProofs(c) => c.has_relation(r),
      ContextEnum::TopBottomKClauses(c) => c.has_relation(r),
      ContextEnum::DiffMinMaxProb(c) => c.has_relation(r),
      ContextEnum::DiffAddMultProb(c) => c.has_relation(r),
      ContextEnum::DiffSampleKProofs(c) => c.has_relation(r),
      ContextEnum::DiffTopKProofs(c, _) => c.has_relation(r),
      ContextEnum::DiffTopBottomKClauses(c) => c.has_relation(r),
      ContextEnum::Custom(c) => c.has_relation(r),
    }
  }

  fn relation_is_computed(&self, r: &str) -> bool {
    match &self.ctx {
      ContextEnum::Unit(c) => c.is_computed(r),
      ContextEnum::Proofs(c) => c.is_computed(r),
      ContextEnum::MinMaxProb(c) => c.is_computed(r),
      ContextEnum::AddMultProb(c) => c.is_computed(r),
      ContextEnum::TopKProofs(c) => c.is_computed(r),
      ContextEnum::TopBottomKClauses(c) => c.is_computed(r),
      ContextEnum::DiffMinMaxProb(c) => c.is_computed(r),
      ContextEnum::DiffAddMultProb(c) => c.is_computed(r),
      ContextEnum::DiffSampleKProofs(c) => c.is_computed(r),
      ContextEnum::DiffTopKProofs(c, _) => c.is_computed(r),
      ContextEnum::DiffTopBottomKClauses(c) => c.is_computed(r),
      ContextEnum::Custom(c) => c.is_computed(r),
    }
  }

  #[args(include_hidden = false)]
  fn num_relations(&self, include_hidden: bool) -> usize {
    if include_hidden {
      match &self.ctx {
        ContextEnum::Unit(c) => c.num_all_relations(),
        ContextEnum::Proofs(c) => c.num_all_relations(),
        ContextEnum::MinMaxProb(c) => c.num_all_relations(),
        ContextEnum::AddMultProb(c) => c.num_all_relations(),
        ContextEnum::TopKProofs(c) => c.num_all_relations(),
        ContextEnum::TopBottomKClauses(c) => c.num_all_relations(),
        ContextEnum::DiffMinMaxProb(c) => c.num_all_relations(),
        ContextEnum::DiffAddMultProb(c) => c.num_all_relations(),
        ContextEnum::DiffSampleKProofs(c) => c.num_all_relations(),
        ContextEnum::DiffTopKProofs(c, _) => c.num_all_relations(),
        ContextEnum::DiffTopBottomKClauses(c) => c.num_all_relations(),
        ContextEnum::Custom(c) => c.num_all_relations(),
      }
    } else {
      match &self.ctx {
        ContextEnum::Unit(c) => c.num_relations(),
        ContextEnum::Proofs(c) => c.num_relations(),
        ContextEnum::MinMaxProb(c) => c.num_relations(),
        ContextEnum::AddMultProb(c) => c.num_relations(),
        ContextEnum::TopKProofs(c) => c.num_relations(),
        ContextEnum::TopBottomKClauses(c) => c.num_relations(),
        ContextEnum::DiffMinMaxProb(c) => c.num_relations(),
        ContextEnum::DiffAddMultProb(c) => c.num_relations(),
        ContextEnum::DiffSampleKProofs(c) => c.num_relations(),
        ContextEnum::DiffTopKProofs(c, _) => c.num_relations(),
        ContextEnum::DiffTopBottomKClauses(c) => c.num_relations(),
        ContextEnum::Custom(c) => c.num_relations(),
      }
    }
  }

  #[args(include_hidden = false)]
  fn relations(&self, include_hidden: bool) -> Vec<String> {
    if include_hidden {
      match &self.ctx {
        ContextEnum::Unit(c) => c.all_relations(),
        ContextEnum::Proofs(c) => c.all_relations(),
        ContextEnum::MinMaxProb(c) => c.all_relations(),
        ContextEnum::AddMultProb(c) => c.all_relations(),
        ContextEnum::TopKProofs(c) => c.all_relations(),
        ContextEnum::TopBottomKClauses(c) => c.all_relations(),
        ContextEnum::DiffMinMaxProb(c) => c.all_relations(),
        ContextEnum::DiffAddMultProb(c) => c.all_relations(),
        ContextEnum::DiffSampleKProofs(c) => c.all_relations(),
        ContextEnum::DiffTopKProofs(c, _) => c.all_relations(),
        ContextEnum::DiffTopBottomKClauses(c) => c.all_relations(),
        ContextEnum::Custom(c) => c.all_relations(),
      }
    } else {
      match &self.ctx {
        ContextEnum::Unit(c) => c.relations(),
        ContextEnum::Proofs(c) => c.relations(),
        ContextEnum::MinMaxProb(c) => c.relations(),
        ContextEnum::AddMultProb(c) => c.relations(),
        ContextEnum::TopKProofs(c) => c.relations(),
        ContextEnum::TopBottomKClauses(c) => c.relations(),
        ContextEnum::DiffMinMaxProb(c) => c.relations(),
        ContextEnum::DiffAddMultProb(c) => c.relations(),
        ContextEnum::DiffSampleKProofs(c) => c.relations(),
        ContextEnum::DiffTopKProofs(c, _) => c.relations(),
        ContextEnum::DiffTopBottomKClauses(c) => c.relations(),
        ContextEnum::Custom(c) => c.relations(),
      }
    }
  }
}

impl Context {
  fn run_batch_parallel(
    &self,
    iter_limit: Option<usize>,
    output_relations: Vec<&str>,
    inputs: HashMap<String, Vec<(&PyList, Option<Vec<Vec<usize>>>)>>,
  ) -> Result<Vec<Collection>, BindingError> {
    let batch_size = inputs.iter().next().unwrap().1.len();
    match &self.ctx {
      ContextEnum::Unit(c) => {
        let inputs = process_batched_inputs(inputs, process_unit_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, _| CollectionEnum::unit(c),
        )
      }
      ContextEnum::Proofs(c) => {
        let inputs = process_batched_inputs(inputs, process_unit_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, _| CollectionEnum::proofs(c),
        )
      }
      ContextEnum::MinMaxProb(c) => {
        let inputs = process_batched_inputs(inputs, process_prob_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, _| CollectionEnum::min_max_prob(c),
        )
      }
      ContextEnum::AddMultProb(c) => {
        let inputs = process_batched_inputs(inputs, process_prob_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, _| CollectionEnum::add_mult_prob(c),
        )
      }
      ContextEnum::TopKProofs(c) => {
        let inputs = process_batched_inputs(inputs, process_prob_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, prov_ctx| CollectionEnum::TopKProofs {
            collection: c,
            tags: Arc::clone(&prov_ctx.probs),
          },
        )
      }
      ContextEnum::TopBottomKClauses(c) => {
        let inputs = process_batched_inputs(inputs, process_prob_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, prov_ctx| CollectionEnum::TopBottomKClauses {
            collection: c,
            tags: Arc::clone(&prov_ctx.probs),
          },
        )
      }
      ContextEnum::DiffMinMaxProb(c) => {
        let inputs =
          process_batched_inputs(inputs, process_diff_prob_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, prov_ctx| CollectionEnum::DiffMinMaxProb {
            collection: c,
            tags: Arc::clone(&prov_ctx.diff_probs),
          },
        )
      }
      ContextEnum::DiffAddMultProb(c) => {
        let inputs =
          process_batched_inputs(inputs, process_diff_prob_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, prov_ctx| CollectionEnum::DiffAddMultProb {
            collection: c,
            tags: Arc::clone(&prov_ctx.storage),
          },
        )
      }
      ContextEnum::DiffSampleKProofs(c) => {
        let inputs =
          process_batched_inputs(inputs, process_diff_prob_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, prov_ctx| CollectionEnum::DiffSampleKProofs {
            collection: c,
            tags: Arc::clone(&prov_ctx.diff_probs),
          },
        )
      }
      ContextEnum::DiffTopKProofs(c, wmc_type) => {
        let inputs =
          process_batched_inputs(inputs, process_diff_prob_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, prov_ctx| CollectionEnum::DiffTopKProofs {
            collection: c,
            tags: Arc::clone(&prov_ctx.diff_probs),
            wmc_type: wmc_type.clone(),
          },
        )
      }
      ContextEnum::DiffTopBottomKClauses(c) => {
        let inputs =
          process_batched_inputs(inputs, process_diff_prob_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, prov_ctx| CollectionEnum::DiffTopBottomKClauses {
            collection: c,
            tags: Arc::clone(&prov_ctx.diff_probs),
          },
        )
      }
      ContextEnum::Custom(c) => {
        let inputs = process_batched_inputs(inputs, process_custom_facts, |r| c.relation_type(r))?;
        run_batch_parallel(
          c,
          batch_size,
          inputs,
          output_relations,
          iter_limit,
          |c, _| CollectionEnum::Custom { collection: c },
        )
      }
    }
  }

  fn run_batch_serial(
    &self,
    iter_limit: Option<usize>,
    output_relation: Vec<&str>,
    inputs: HashMap<String, Vec<(&PyList, Option<Vec<Vec<usize>>>)>>,
  ) -> Result<Vec<Collection>, BindingError> {
    let batch_size = inputs.iter().next().unwrap().1.len();
    (0..batch_size)
      .map(|i| {
        let mut temp_ctx = self.clone();
        for (relation, relation_inputs) in &inputs {
          temp_ctx.add_facts(relation, relation_inputs[i].0, relation_inputs[i].1.clone())?;
        }
        temp_ctx.run(iter_limit)?;
        temp_ctx.relation(&output_relation[i])
      })
      .collect()
  }
}

#[derive(Clone)]
pub enum ContextEnum {
  Unit(IntegrateContext<unit::UnitContext, ArcFamily>),
  Proofs(IntegrateContext<proofs::ProofsContext, ArcFamily>),
  MinMaxProb(IntegrateContext<min_max_prob::MinMaxProbContext, ArcFamily>),
  AddMultProb(IntegrateContext<add_mult_prob::AddMultProbContext, ArcFamily>),
  TopKProofs(IntegrateContext<top_k_proofs::TopKProofsContext<ArcFamily>, ArcFamily>),
  TopBottomKClauses(
    IntegrateContext<top_bottom_k_clauses::TopBottomKClausesContext<ArcFamily>, ArcFamily>,
  ),
  DiffMinMaxProb(
    IntegrateContext<diff_min_max_prob::DiffMinMaxProbContext<Py<PyAny>, ArcFamily>, ArcFamily>,
  ),
  DiffAddMultProb(
    IntegrateContext<diff_add_mult_prob::DiffAddMultProbContext<Py<PyAny>, ArcFamily>, ArcFamily>,
  ),
  DiffSampleKProofs(
    IntegrateContext<
      diff_sample_k_proofs::DiffSampleKProofsContext<Py<PyAny>, ArcFamily>,
      ArcFamily,
    >,
  ),
  DiffTopKProofs(
    IntegrateContext<diff_top_k_proofs::DiffTopKProofsContext<Py<PyAny>, ArcFamily>, ArcFamily>,
    WMCType,
  ),
  DiffTopBottomKClauses(
    IntegrateContext<
      diff_top_bottom_k_clauses::DiffTopBottomKClausesContext<Py<PyAny>, ArcFamily>,
      ArcFamily,
    >,
  ),
  Custom(IntegrateContext<custom_tag::CustomTagContext, ArcFamily>),
}

fn process_unit_facts(elems: &PyList, tuple_type: TupleType) -> PyResult<Vec<(Option<()>, Tuple)>> {
  let elems: Vec<&PyAny> = elems.extract()?;
  elems
    .into_iter()
    .map(|elem| from_python_tuple(elem, &tuple_type).map(|e| (None, e)))
    .collect::<PyResult<Vec<_>>>()
}

fn process_prob_facts(
  elems: &PyList,
  tuple_type: TupleType,
) -> PyResult<Vec<(Option<f64>, Tuple)>> {
  let elems: Vec<&PyAny> = elems.extract()?;
  elems
    .into_iter()
    .map(|elem| {
      let info_tup: &PyTuple = elem.cast_as()?;
      let info: Option<f64> = info_tup.get_item(0)?.extract()?;
      let tup = from_python_tuple(info_tup.get_item(1)?, &tuple_type)?;
      Ok((info, tup))
    })
    .collect::<PyResult<Vec<_>>>()
}

fn process_diff_prob_facts(
  elems: &PyList,
  tuple_type: TupleType,
) -> PyResult<Vec<(Option<InputDiffProb<Py<PyAny>>>, Tuple)>> {
  let elems: Vec<&PyAny> = elems.extract()?;
  elems
    .into_iter()
    .map(|elem| {
      let info_tup: &PyTuple = elem.cast_as()?;
      let maybe_prob: Option<f64> = info_tup.get_item(0)?.extract()?;
      let tag = if let Some(prob) = maybe_prob {
        let tag: Py<PyAny> = info_tup.get_item(0)?.into();
        Some((prob, tag).into())
      } else {
        None
      };
      let tup = from_python_tuple(info_tup.get_item(1)?, &tuple_type)?;
      Ok((tag, tup))
    })
    .collect::<PyResult<Vec<_>>>()
}

fn process_custom_facts(
  elems: &PyList,
  tuple_type: TupleType,
) -> PyResult<Vec<(Option<Py<PyAny>>, Tuple)>> {
  let elems: Vec<&PyAny> = elems.extract()?;
  elems
    .into_iter()
    .map(|elem| {
      let info_tup: &PyTuple = elem.cast_as()?;
      let maybe_tag: Option<&PyAny> = info_tup.get_item(0)?.into();
      let tag = if let Some(tag) = maybe_tag {
        let tag: Py<PyAny> = tag.into();
        Some(tag)
      } else {
        None
      };
      let tup = from_python_tuple(info_tup.get_item(1)?, &tuple_type)?;
      Ok((tag, tup))
    })
    .collect::<PyResult<Vec<_>>>()
}

fn process_batched_inputs<P, G, T>(
  inputs: HashMap<String, Vec<(&PyList, Option<Vec<Vec<usize>>>)>>,
  process_facts: P,
  get_relation_type: G,
) -> PyResult<
  Vec<(
    String,
    Vec<(Vec<(Option<T>, Tuple)>, Option<Vec<Vec<usize>>>)>,
  )>,
>
where
  P: Fn(&PyList, TupleType) -> PyResult<Vec<(Option<T>, Tuple)>>,
  G: Fn(&str) -> Option<TupleType>,
{
  inputs
    .into_iter()
    .map(|(r, b)| {
      let tuple_type = get_relation_type(&r).ok_or(BindingError::UnknownRelation(r.clone()))?;
      let batch = b
        .into_iter()
        .map(|(elems, disj)| {
          let tuples = process_facts(elems, tuple_type.clone())?;
          Ok((tuples, disj))
        })
        .collect::<PyResult<Vec<_>>>()?;
      Ok((r, batch))
    })
    .collect::<PyResult<Vec<_>>>()
}

fn run_batch_parallel<C, F>(
  integrate_context: &IntegrateContext<C, ArcFamily>,
  batch_size: usize,
  inputs: Vec<(
    String,
    Vec<(Vec<(Option<C::InputTag>, Tuple)>, Option<Vec<Vec<usize>>>)>,
  )>,
  output_relations: Vec<&str>,
  iter_limit: Option<usize>,
  f: F,
) -> Result<Vec<Collection>, BindingError>
where
  C: ProvenanceContext + Clone + std::marker::Sync,
  <C as ProvenanceContext>::InputTag: std::marker::Sync,
  <C as ProvenanceContext>::OutputTag: std::marker::Sync + std::marker::Send,
  <C as ProvenanceContext>::Tag: std::marker::Sync + std::marker::Send,
  F: Fn(Arc<DynamicOutputCollection<C::Tag>>, &C) -> CollectionEnum + std::marker::Sync,
{
  let internal = integrate_context.internal_context();
  (0..batch_size)
    .into_par_iter()
    .map(|i| {
      let output_relation = output_relations[i];
      let mut temp_ctx = internal.clone();
      for (relation, batch) in &inputs {
        temp_ctx.add_facts_with_disjunction(
          relation,
          batch[i].0.clone(),
          batch[i].1.clone(),
          false,
        )?;
      }
      temp_ctx.run_with_iter_limit(iter_limit)?;
      let computed =
        temp_ctx
          .computed_rc_relation(output_relation)
          .ok_or(BindingError::RelationNotComputed(
            output_relation.to_string(),
          ))?;
      Ok(f(computed, temp_ctx.provenance_context()).into())
    })
    .collect()
}

fn is_all_equal<T: Iterator<Item = usize> + Clone>(i: T) -> bool {
  i.clone().min() == i.max()
}
