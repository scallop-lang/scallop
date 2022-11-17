use std::collections::*;

use pyo3::prelude::*;
use pyo3::types::PyList;

use rayon::prelude::*;

use scallop_core::common::tuple::*;
use scallop_core::common::tuple_type::*;
use scallop_core::compiler;
use scallop_core::integrate::*;
use scallop_core::runtime::monitor;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::*;

use crate::custom_tag;

use super::collection::*;
use super::error::*;
use super::io::*;
use super::provenance::*;
use super::tuple::*;

type AF = ArcFamily;

#[derive(Clone)]
pub enum ContextEnum {
  Unit(IntegrateContext<unit::UnitProvenance, AF>),
  Proofs(IntegrateContext<proofs::ProofsProvenance, AF>),
  MinMaxProb(IntegrateContext<min_max_prob::MinMaxProbProvenance, AF>),
  AddMultProb(IntegrateContext<add_mult_prob::AddMultProbContext, AF>),
  TopKProofs(IntegrateContext<top_k_proofs::TopKProofsContext<AF>, AF>),
  TopBottomKClauses(IntegrateContext<top_bottom_k_clauses::TopBottomKClausesContext<AF>, AF>),
  DiffMinMaxProb(IntegrateContext<diff_min_max_prob::DiffMinMaxProbContext<Py<PyAny>, AF>, AF>),
  DiffAddMultProb(IntegrateContext<diff_add_mult_prob::DiffAddMultProbContext<Py<PyAny>, AF>, AF>),
  DiffNandMultProb(IntegrateContext<diff_nand_mult_prob::DiffNandMultProbContext<Py<PyAny>, AF>, AF>),
  DiffMaxMultProb(IntegrateContext<diff_max_mult_prob::DiffMaxMultProbContext<Py<PyAny>, AF>, AF>),
  DiffNandMinProb(IntegrateContext<diff_nand_min_prob::DiffNandMinProbContext<Py<PyAny>, AF>, AF>),
  DiffSampleKProofs(IntegrateContext<diff_sample_k_proofs::DiffSampleKProofsContext<Py<PyAny>, AF>, AF>),
  DiffTopKProofs(IntegrateContext<diff_top_k_proofs::DiffTopKProofsContext<Py<PyAny>, AF>, AF>),
  DiffTopKProofsIndiv(IntegrateContext<diff_top_k_proofs_indiv::DiffTopKProofsIndivContext<Py<PyAny>, AF>, AF>),
  DiffTopBottomKClauses(IntegrateContext<diff_top_bottom_k_clauses::DiffTopBottomKClausesContext<Py<PyAny>, AF>, AF>),
  Custom(IntegrateContext<custom_tag::CustomTagContext, AF>),
}

macro_rules! match_context {
  ($ctx:expr, $v:ident, $e:expr) => {
    match $ctx {
      ContextEnum::Unit($v) => $e,
      ContextEnum::Proofs($v) => $e,
      ContextEnum::MinMaxProb($v) => $e,
      ContextEnum::AddMultProb($v) => $e,
      ContextEnum::TopKProofs($v) => $e,
      ContextEnum::TopBottomKClauses($v) => $e,
      ContextEnum::DiffMinMaxProb($v) => $e,
      ContextEnum::DiffAddMultProb($v) => $e,
      ContextEnum::DiffNandMultProb($v) => $e,
      ContextEnum::DiffMaxMultProb($v) => $e,
      ContextEnum::DiffNandMinProb($v) => $e,
      ContextEnum::DiffSampleKProofs($v) => $e,
      ContextEnum::DiffTopKProofs($v) => $e,
      ContextEnum::DiffTopKProofsIndiv($v) => $e,
      ContextEnum::DiffTopBottomKClauses($v) => $e,
      ContextEnum::Custom($v) => $e,
    }
  };
}

#[pyclass(unsendable, name = "InternalScallopContext")]
pub struct Context {
  pub ctx: ContextEnum,
}

#[pymethods]
impl Context {
  /// Create a new scallop context
  ///
  /// # Arguments
  ///
  /// * `provenance` - a string of the name of the provenance to use for this context
  /// * `k` - an unsigned integer serving as the hyper-parameter for provenance such as `"topkproofs"`
  /// * `custom_provenance` - an optional python object serving as the provenance context
  #[new]
  #[args(provenance = "\"unit\"", k = "3", custom_provenance = "None")]
  fn new(provenance: &str, k: usize, custom_provenance: Option<Py<PyAny>>) -> Result<Self, BindingError> {
    // Check provenance type
    match provenance {
      "unit" => Ok(Self {
        ctx: ContextEnum::Unit(IntegrateContext::new(unit::UnitProvenance::default())),
      }),
      "proofs" => Ok(Self {
        ctx: ContextEnum::Proofs(IntegrateContext::new(proofs::ProofsProvenance::default())),
      }),
      "minmaxprob" => Ok(Self {
        ctx: ContextEnum::MinMaxProb(IntegrateContext::new(min_max_prob::MinMaxProbProvenance::default())),
      }),
      "addmultprob" => Ok(Self {
        ctx: ContextEnum::AddMultProb(IntegrateContext::new(add_mult_prob::AddMultProbContext::default())),
      }),
      "topkproofs" => Ok(Self {
        ctx: ContextEnum::TopKProofs(IntegrateContext::new(top_k_proofs::TopKProofsContext::new(k))),
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
      "diffnandmultprob" => Ok(Self {
        ctx: ContextEnum::DiffNandMultProb(IntegrateContext::new(
          diff_nand_mult_prob::DiffNandMultProbContext::default(),
        )),
      }),
      "diffmaxmultprob" => Ok(Self {
        ctx: ContextEnum::DiffMaxMultProb(IntegrateContext::new(
          diff_max_mult_prob::DiffMaxMultProbContext::default(),
        )),
      }),
      "diffnandminprob" => Ok(Self {
        ctx: ContextEnum::DiffNandMinProb(IntegrateContext::new(
          diff_nand_min_prob::DiffNandMinProbContext::default(),
        )),
      }),
      "diffsamplekproofs" => Ok(Self {
        ctx: ContextEnum::DiffSampleKProofs(IntegrateContext::new(
          diff_sample_k_proofs::DiffSampleKProofsContext::new(k),
        )),
      }),
      "difftopkproofs" => Ok(Self {
        ctx: ContextEnum::DiffTopKProofs(IntegrateContext::new(diff_top_k_proofs::DiffTopKProofsContext::new(k))),
      }),
      "difftopkproofsindiv" => Ok(Self {
        ctx: ContextEnum::DiffTopKProofsIndiv(IntegrateContext::new(
          diff_top_k_proofs_indiv::DiffTopKProofsIndivContext::new(k),
        )),
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

  /// Creates a clone of the scallopy context
  fn clone(&self) -> Self {
    Self { ctx: self.ctx.clone() }
  }

  /// Compile the surface program stored in the scallopy context into the ram program.
  ///
  /// This function is usually used before creating a forward function.
  /// If called multiple times, the ram program compiled later will overwrite the previous ones.
  fn compile(&mut self) -> Result<(), BindingError> {
    match_context!(&mut self.ctx, c, c.compile().map_err(BindingError::from))
  }

  /// JIT compile the surface program into a python library, by invoking `sclc::pylib::create_pylib`.
  ///
  /// When succeeded, there will be a `target_file` created, which can be in turn imported by python runtime.
  fn jit_compile(&mut self, target_relations: Vec<&str>, target_file: String) -> Result<(), BindingError> {
    // First compile the program
    match_context!(
      &mut self.ctx,
      c,
      c.compile_with_output_relations(Some(target_relations))
        .map_err(BindingError::from)
    )?;

    // Get the ram from compilation context
    let ram = match_context!(&self.ctx, c, c.ram());

    // Invoke sclc to compile the ram into rust code
    let opt = sclc::options::Options {
      input: std::path::PathBuf::from(target_file),
      do_not_copy_artifact: true,
      ..Default::default()
    };
    let compile_opt = compiler::CompileOptions::default();
    sclc::pylib::create_pylib(&opt, compile_opt, &ram).expect("Compile Error");

    // Return result
    Ok(())
  }

  /// Import a module written in scallop file
  fn import_file(&mut self, file_name: &str) -> Result<(), BindingError> {
    match_context!(&mut self.ctx, c, c.import_file(file_name).map_err(BindingError::from))
  }

  /// Add a program in the form of string
  fn add_program(&mut self, program: &str) -> Result<(), BindingError> {
    match_context!(&mut self.ctx, c, c.add_program(program).map_err(BindingError::from))
  }

  /// Print the scallop program
  fn dump_front_ir(&self) {
    match_context!(&self.ctx, c, c.dump_front_ir())
  }

  /// Return the scallop program in its string form
  fn get_front_ir(&self) -> String {
    match_context!(&self.ctx, c, c.get_front_ir())
  }

  /// Obtain the list of input tags, if the underlying provenance supports it.
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
      ContextEnum::DiffNandMultProb(c) => Some(c.provenance_context().input_tags()),
      ContextEnum::DiffMaxMultProb(c) => Some(c.provenance_context().input_tags()),
      ContextEnum::DiffNandMinProb(c) => Some(c.provenance_context().input_tags()),
      ContextEnum::DiffSampleKProofs(c) => Some(c.provenance_context().input_tags()),
      ContextEnum::DiffTopKProofs(c) => Some(c.provenance_context().input_tags()),
      ContextEnum::DiffTopKProofsIndiv(c) => Some(c.provenance_context().input_tags()),
      ContextEnum::DiffTopBottomKClauses(c) => Some(c.provenance_context().input_tags()),
      ContextEnum::Custom(_) => None,
    }
  }

  /// Set the hyper-parameter `k`
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
      ContextEnum::DiffNandMultProb(_) => (),
      ContextEnum::DiffMaxMultProb(_) => (),
      ContextEnum::DiffNandMinProb(_) => (),
      ContextEnum::DiffSampleKProofs(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::DiffTopKProofs(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::DiffTopKProofsIndiv(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::DiffTopBottomKClauses(c) => {
        c.provenance_context_mut().set_k(k);
      }
      ContextEnum::Custom(_) => (),
    }
  }

  /// Declare a relation given a string of type declaration.
  ///
  /// # Example
  ///
  /// ```
  /// # ctx = Context::new("unit", 3, None).unwrap();
  /// ctx.add_relation("atom(usize, usize)", None, None).unwrap();
  /// ```
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
    let rtd = match_context!(&mut self.ctx, c, c.add_relation_with_attributes(relation, attrs)?);

    // Return; unwrap is ok here since we made sure that the relation is successfully added and there
    // will always be only one relation being added
    Ok(rtd.predicates().next().unwrap().to_string())
  }

  /// Add a list of facts to a relation
  fn add_facts(
    &mut self,
    relation: &str,
    elems: &PyList,
    disjunctions: Option<Vec<Vec<usize>>>,
  ) -> Result<(), BindingError> {
    match_context!(&mut self.ctx, c, add_py_facts(c, relation, elems, disjunctions))
  }

  /// Add a rule
  #[args(tag = "None", demand = "None")]
  fn add_rule(&mut self, rule: &str, tag: Option<&PyAny>, demand: Option<String>) -> Result<(), BindingError> {
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

    match_context!(&mut self.ctx, c, add_py_rule(c, rule, tag, attrs))
  }

  /// Execute the program
  ///
  /// If the context's ram program is already compiled, the program will be directly executed.
  /// Otherwise, the ram program will be compiled and than be executed.
  fn run(&mut self, iter_limit: Option<usize>) -> Result<(), BindingError> {
    match_context!(&mut self.ctx, c, c.run_with_iter_limit(iter_limit)?);
    Ok(())
  }

  /// Execute the program while monitoring the provenance and tags
  fn run_with_debug_tag(&mut self, iter_limit: Option<usize>) -> Result<(), BindingError> {
    let m = monitor::DebugTagsMonitor;
    match_context!(&mut self.ctx, c, c.run_with_iter_limit_and_monitor(iter_limit, &m)?);
    Ok(())
  }

  /// Check if the tuple matches the type of the queried relation.
  ///
  /// Error will be returned if the relation does not exist.
  fn check_tuple(&self, relation: &str, tup: &PyAny) -> Result<bool, BindingError> {
    match_context!(&self.ctx, c, check_py_tuple(c, relation, tup))
  }

  /// Check if a set of tuples match the type of the queried relation.
  ///
  /// Error will be returned if the relation does not exist.
  fn check_tuples(&self, relation: &str, tups: Vec<&PyAny>) -> Result<bool, BindingError> {
    match_context!(&self.ctx, c, check_py_tuples(c, relation, tups))
  }

  /// Run a batch of tasks
  ///
  /// # Arguments
  ///
  /// * `iter_limit` - the optional iteration count limit. If `None`, there is no limit to iteration count.
  /// * `output_relations` - a doubly nested vector of strings.
  ///   The first level is an array of batched tasks, and the second level is the list of output relations for that particular task
  /// * `inputs` - the input facts, a mapping from the relation name to a batch of input facts.
  ///   For each task, there is a `PyList` of input facts and another optional set of annotated disjunctions
  /// * `parallel` - whether the batch is to be executed in parallel
  fn run_batch(
    &self,
    iter_limit: Option<usize>,
    output_relations: Vec<Vec<&str>>,
    inputs: HashMap<String, Vec<(&PyList, Option<Vec<Vec<usize>>>)>>,
    parallel: bool,
  ) -> Result<Vec<Vec<Collection>>, BindingError> {
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

    // Depending on whether parallelism is enabled, choose the subsequent function to call
    if parallel {
      self.run_batch_parallel(iter_limit, output_relations, inputs)
    } else {
      self.run_batch_serial(iter_limit, output_relations, inputs)
    }
  }

  /// Get the output collection of the relation
  ///
  /// Has to be called after `ctx.run()`, otherwise the output collection will not be computed.
  /// Error is returned if the relation is not computed or does not exist.
  fn relation(&mut self, r: &str) -> Result<Collection, BindingError> {
    let maybe_coll_enum = match_context!(&mut self.ctx, c, get_output_collection(c, r));
    maybe_coll_enum.map(|c| c.into())
  }

  /// Get the output collection of the relation, while monitoring the provenance and tags
  ///
  /// Has to be called after `ctx.run()`, otherwise the output collection will not be computed.
  /// Error is returned if the relation is not computed or does not exist.
  fn relation_with_debug_tag(&mut self, r: &str) -> Result<Collection, BindingError> {
    let m = monitor::DebugTagsMonitor;
    let maybe_coll_enum = match_context!(&mut self.ctx, c, get_output_collection_monitor(c, &m, r));
    maybe_coll_enum.map(|c| c.into())
  }

  /// Check if the context contains a relation
  fn has_relation(&self, r: &str) -> bool {
    match_context!(&self.ctx, c, c.has_relation(r))
  }

  /// Check if the relation is computed in this context
  fn relation_is_computed(&self, r: &str) -> bool {
    match_context!(&self.ctx, c, c.is_computed(r))
  }

  /// Get the number of relations in this context.
  /// If `include_hidden` is set `true`, the result will also count the hidden relations
  #[args(include_hidden = false)]
  fn num_relations(&self, include_hidden: bool) -> usize {
    if include_hidden {
      match_context!(&self.ctx, c, c.num_all_relations())
    } else {
      match_context!(&self.ctx, c, c.num_relations())
    }
  }

  /// Get a list of relations in this context.
  /// If `include_hidden` is set `true`, the result will also include the hidden relations
  #[args(include_hidden = false)]
  fn relations(&self, include_hidden: bool) -> Vec<String> {
    if include_hidden {
      match_context!(&self.ctx, c, c.all_relations())
    } else {
      match_context!(&self.ctx, c, c.relations())
    }
  }
}

impl Context {
  fn run_batch_parallel(
    &self,
    iter_limit: Option<usize>,
    output_relations: Vec<Vec<&str>>,
    inputs: HashMap<String, Vec<(&PyList, Option<Vec<Vec<usize>>>)>>,
  ) -> Result<Vec<Vec<Collection>>, BindingError> {
    // Helper function for running batch parallel
    fn run<C>(
      c: &IntegrateContext<C, AF>,
      iter_limit: Option<usize>,
      output_relations: Vec<Vec<&str>>,
      inputs: HashMap<String, Vec<(&PyList, Option<Vec<Vec<usize>>>)>>,
    ) -> Result<Vec<Vec<Collection>>, BindingError>
    where
      C: PythonProvenance + Clone + std::marker::Sync,
      <C as Provenance>::InputTag: std::marker::Sync,
      <C as Provenance>::OutputTag: std::marker::Sync + std::marker::Send,
      <C as Provenance>::Tag: std::marker::Sync + std::marker::Send,
    {
      let batch_size = inputs.iter().next().unwrap().1.len();
      let inputs = process_batched_inputs::<C, _>(inputs, |r| c.relation_type(r))?;
      run_batch_parallel(c, batch_size, inputs, output_relations, iter_limit)
    }

    // Run the helper for each context
    match_context!(&self.ctx, c, run(c, iter_limit, output_relations, inputs))
  }

  fn run_batch_serial(
    &self,
    iter_limit: Option<usize>,
    output_relation: Vec<Vec<&str>>,
    inputs: HashMap<String, Vec<(&PyList, Option<Vec<Vec<usize>>>)>>,
  ) -> Result<Vec<Vec<Collection>>, BindingError> {
    let batch_size = inputs.iter().next().unwrap().1.len();
    (0..batch_size)
      .map(|i| {
        let mut temp_ctx = self.clone();
        for (relation, relation_inputs) in &inputs {
          temp_ctx.add_facts(relation, relation_inputs[i].0, relation_inputs[i].1.clone())?;
        }
        temp_ctx.run(iter_limit)?;
        output_relation[i].iter().map(|r| temp_ctx.relation(r)).collect()
      })
      .collect()
  }
}

fn add_py_rule<C>(
  c: &mut IntegrateContext<C, AF>,
  rule: &str,
  tag: Option<&PyAny>,
  attrs: Vec<Attribute>,
) -> Result<(), BindingError>
where
  C: PythonProvenance,
{
  let tag: Option<C::InputTag> = C::process_optional_py_tag(tag)?;
  c.add_rule_with_options(rule, tag, attrs)?;
  Ok(())
}

fn add_py_facts<C>(
  c: &mut IntegrateContext<C, AF>,
  relation: &str,
  elems: &PyList,
  disjunctions: Option<Vec<Vec<usize>>>,
) -> Result<(), BindingError>
where
  C: PythonProvenance,
{
  if let Some(tuple_type) = c.relation_type(relation) {
    let tuples = C::process_typed_py_facts(elems, &tuple_type)?;
    c.add_facts_with_disjunction(relation, tuples, disjunctions, false)?;
    Ok(())
  } else {
    Err(BindingError::UnknownRelation(relation.to_string()).into())
  }
}

fn check_py_tuple<C>(c: &IntegrateContext<C, AF>, relation: &str, py_tup: &PyAny) -> Result<bool, BindingError>
where
  C: PythonProvenance,
{
  if let Some(tuple_type) = c.relation_type(relation) {
    Ok(from_python_tuple(py_tup, &tuple_type).is_ok())
  } else {
    Err(BindingError::UnknownRelation(relation.to_string()).into())
  }
}

fn check_py_tuples<C>(c: &IntegrateContext<C, AF>, relation: &str, py_tups: Vec<&PyAny>) -> Result<bool, BindingError>
where
  C: PythonProvenance,
{
  if let Some(tuple_type) = c.relation_type(relation) {
    for py_tup in py_tups {
      if from_python_tuple(py_tup, &tuple_type).is_err() {
        return Ok(false);
      }
    }
    Ok(true)
  } else {
    Err(BindingError::UnknownRelation(relation.to_string()).into())
  }
}

fn process_batched_inputs<C, G>(
  inputs: HashMap<String, Vec<(&PyList, Option<Vec<Vec<usize>>>)>>,
  get_relation_type: G,
) -> PyResult<
  Vec<(
    String,
    Vec<(Vec<(Option<C::InputTag>, Tuple)>, Option<Vec<Vec<usize>>>)>,
  )>,
>
where
  C: PythonProvenance,
  G: Fn(&str) -> Option<TupleType>,
{
  inputs
    .into_iter()
    .map(|(r, b)| {
      let tuple_type = get_relation_type(&r).ok_or(BindingError::UnknownRelation(r.clone()))?;
      let batch = b
        .into_iter()
        .map(|(elems, disj)| {
          let tuples = C::process_typed_py_facts(elems, &tuple_type)?;
          Ok((tuples, disj))
        })
        .collect::<PyResult<Vec<_>>>()?;
      Ok((r, batch))
    })
    .collect::<PyResult<Vec<_>>>()
}

fn run_batch_parallel<C>(
  integrate_context: &IntegrateContext<C, AF>,
  batch_size: usize,
  inputs: Vec<(
    String,
    Vec<(Vec<(Option<C::InputTag>, Tuple)>, Option<Vec<Vec<usize>>>)>,
  )>,
  batch_output_relations: Vec<Vec<&str>>,
  iter_limit: Option<usize>,
) -> Result<Vec<Vec<Collection>>, BindingError>
where
  C: PythonProvenance + Clone + std::marker::Sync,
  <C as Provenance>::InputTag: std::marker::Sync,
  <C as Provenance>::OutputTag: std::marker::Sync + std::marker::Send,
  <C as Provenance>::Tag: std::marker::Sync + std::marker::Send,
{
  let internal = integrate_context.internal_context();
  (0..batch_size)
    .into_par_iter()
    .map(|i| {
      let output_relations = &batch_output_relations[i];
      let mut temp_ctx = internal.clone();
      for (relation, batch) in &inputs {
        temp_ctx.add_facts_with_disjunction(relation, batch[i].0.clone(), batch[i].1.clone(), false)?;
      }
      temp_ctx.run_with_iter_limit(iter_limit)?;
      output_relations
        .iter()
        .map(|r| {
          let computed = temp_ctx
            .computed_rc_relation(r)
            .ok_or(BindingError::RelationNotComputed(r.to_string()))?;
          Ok(C::to_collection_enum(computed, temp_ctx.provenance_context()).into())
        })
        .collect::<Result<Vec<Collection>, _>>()
    })
    .collect()
}

fn is_all_equal<T: Iterator<Item = usize> + Clone>(i: T) -> bool {
  i.clone().min() == i.max()
}

fn get_output_collection<C>(c: &mut IntegrateContext<C, ArcFamily>, r: &str) -> Result<CollectionEnum, BindingError>
where
  C: PythonProvenance,
{
  if c.has_relation(r) {
    if let Some(collection) = c.computed_rc_relation(r) {
      Ok(C::to_collection_enum(
        ArcFamily::clone_ptr(&collection),
        c.provenance_context(),
      ))
    } else {
      Err(BindingError::RelationNotComputed(r.to_string()))
    }
  } else {
    Err(BindingError::UnknownRelation(r.to_string()))
  }
}

fn get_output_collection_monitor<C, M>(
  c: &mut IntegrateContext<C, ArcFamily>,
  m: &M,
  r: &str,
) -> Result<CollectionEnum, BindingError>
where
  C: PythonProvenance,
  M: monitor::Monitor<C>,
{
  if c.has_relation(r) {
    if let Some(collection) = c.computed_rc_relation_with_monitor(r, m) {
      Ok(C::to_collection_enum(
        ArcFamily::clone_ptr(&collection),
        c.provenance_context(),
      ))
    } else {
      Err(BindingError::RelationNotComputed(r.to_string()))
    }
  } else {
    Err(BindingError::UnknownRelation(r.to_string()))
  }
}
