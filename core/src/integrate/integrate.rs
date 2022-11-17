use std::collections::HashMap;

use crate::common::input_tag::FromInputTag;
use crate::common::tuple::Tuple;
use crate::common::tuple_type::TupleType;
use crate::compiler;
use crate::runtime::dynamic;
use crate::runtime::monitor::Monitor;
use crate::runtime::provenance;
use crate::runtime::provenance::Provenance;
use crate::utils::{PointerFamily, RcFamily};

use super::Attribute;

pub fn interpret_string(string: String) -> Result<dynamic::Output<provenance::unit::UnitProvenance>, IntegrateError> {
  let ram = compiler::compile_string_to_ram(string).map_err(IntegrateError::Compile)?;
  let mut ctx = provenance::unit::UnitProvenance::default();
  dynamic::interpret(ram, &mut ctx).map_err(IntegrateError::Runtime)
}

pub fn interpret_string_with_ctx<Prov: Provenance>(
  string: String,
  ctx: &mut Prov,
) -> Result<dynamic::Output<Prov>, IntegrateError> {
  let ram = compiler::compile_string_to_ram(string).map_err(IntegrateError::Compile)?;
  dynamic::interpret(ram, ctx).map_err(IntegrateError::Runtime)
}

#[derive(Clone)]
pub struct IntegrateContext<Prov: Provenance, P: PointerFamily = RcFamily> {
  options: compiler::CompileOptions,
  front_ctx: compiler::front::FrontContext,
  internal: CompiledIntegrateContext<Prov, P>,
  front_has_changed: bool,
}

impl<Prov: Provenance, P: PointerFamily> IntegrateContext<Prov, P> {
  pub fn new(prov_ctx: Prov) -> Self {
    Self {
      internal: CompiledIntegrateContext {
        prov_ctx,
        ram_program: compiler::ram::Program::new(),
        exec_ctx: dynamic::DynamicExecutionContext::new(),
        computed_output_relations: HashMap::new(),
      },
      options: compiler::CompileOptions::default(),
      front_ctx: compiler::front::FrontContext::new(),
      front_has_changed: false,
    }
  }

  pub fn provenance_context(&self) -> &Prov {
    &self.internal.prov_ctx
  }

  pub fn provenance_context_mut(&mut self) -> &mut Prov {
    &mut self.internal.prov_ctx
  }

  pub fn internal_context(&self) -> &CompiledIntegrateContext<Prov, P> {
    &self.internal
  }

  /// Import file
  pub fn import_file(&mut self, file_name: &str) -> Result<(), IntegrateError> {
    use std::path::PathBuf;
    let source = compiler::front::FileSource::new(&PathBuf::from(file_name.to_string())).map_err(|e| {
      let front_err = compiler::front::FrontCompileError::singleton(e);
      let compile_err = compiler::CompileError::Front(front_err);
      IntegrateError::Compile(vec![compile_err])
    })?;
    self.front_ctx.compile_source(source).map_err(IntegrateError::front)?;
    self.front_has_changed = true;
    Ok(())
  }

  /// Add a program string
  pub fn add_program(&mut self, program: &str) -> Result<(), IntegrateError> {
    let source = compiler::front::StringSource::new(program.to_string());
    self.front_ctx.compile_source(source).map_err(IntegrateError::front)?;
    self.front_has_changed = true;
    Ok(())
  }

  /// Dump front ir
  pub fn dump_front_ir(&self) {
    self.front_ctx.dump_ir();
  }

  /// Get front ir
  pub fn get_front_ir(&self) -> String {
    self.front_ctx.get_ir()
  }

  /// Compile a relation declaration
  pub fn add_relation(&mut self, string: &str) -> Result<&compiler::front::RelationTypeDecl, IntegrateError> {
    self.front_has_changed = true;
    let source = compiler::front::StringSource::new(string.to_string());
    self
      .front_ctx
      .compile_relation(source)
      .map_err(IntegrateError::front)
      .map(move |sid| self.front_ctx.relation_type_decl_of_source_id(sid).unwrap())
  }

  /// Compile a relation declaration
  pub fn add_relation_with_attributes(
    &mut self,
    string: &str,
    attrs: Vec<Attribute>,
  ) -> Result<&compiler::front::RelationTypeDecl, IntegrateError> {
    self.front_has_changed = true;
    let source = compiler::front::StringSource::new(string.to_string());
    self
      .front_ctx
      .compile_relation_with_annotator(source, |item| {
        item.attributes_mut().extend(attrs.iter().map(Attribute::to_front))
      })
      .map_err(IntegrateError::front)
      .map(move |sid| self.front_ctx.relation_type_decl_of_source_id(sid).unwrap())
  }

  /// Compile a rule
  pub fn add_rule(&mut self, string: &str) -> Result<compiler::front::SourceId, IntegrateError> {
    self.front_has_changed = true;
    let source = compiler::front::StringSource::new(string.to_string());
    self.front_ctx.compile_rule(source).map_err(IntegrateError::front)
  }

  /// Compile a rule
  pub fn add_rule_with_options(
    &mut self,
    string: &str,
    tag: Option<Prov::InputTag>,
    mut attrs: Vec<Attribute>,
  ) -> Result<compiler::front::SourceId, IntegrateError> {
    // First generate all attributes
    if let Some(_) = tag {
      attrs.push(Attribute::named("probabilistic"));
    }

    // Compile and get the source id
    let source_id = self.add_rule_with_attributes(string, attrs)?;

    // Process the tag
    if let Some(rd) = self.front_ctx.rule_decl_of_source_id(source_id) {
      let pred = rd.rule_tag_predicate();
      self.internal.exec_ctx.add_facts(&pred, vec![(tag, Tuple::empty())]);
    }

    // Set changed to true
    self.front_has_changed = true;

    // Return source id
    Ok(source_id)
  }

  /// Add a rule with attributes
  pub fn add_rule_with_attributes(
    &mut self,
    string: &str,
    attrs: Vec<Attribute>,
  ) -> Result<compiler::front::SourceId, IntegrateError> {
    self.front_has_changed = true;
    let source = compiler::front::StringSource::new(string.to_string());
    self
      .front_ctx
      .compile_rule_with_annotator(source, |item: &mut compiler::front::Item| {
        item.attributes_mut().extend(attrs.iter().map(Attribute::to_front))
      })
      .map_err(IntegrateError::front)
  }

  /// Add a list of facts to the given predicate
  pub fn add_facts(
    &mut self,
    predicate: &str,
    facts: Vec<(Option<Prov::InputTag>, Tuple)>,
    type_check: bool,
  ) -> Result<(), IntegrateError> {
    self.add_facts_with_disjunction(predicate, facts, None, type_check)
  }

  /// Add a list of facts to the given predicate, with the option to pass in disjunctions
  pub fn add_facts_with_disjunction(
    &mut self,
    predicate: &str,
    facts: Vec<(Option<Prov::InputTag>, Tuple)>,
    disjunctions: Option<Vec<Vec<usize>>>,
    type_check: bool,
  ) -> Result<(), IntegrateError> {
    // Check type
    if type_check {
      let pred_tuple_type = self.front_ctx.relation_tuple_type(predicate);
      if let Some(tuple_type) = pred_tuple_type {
        for (_, fact) in &facts {
          if fact.tuple_type() != tuple_type {
            return Err(IntegrateError::Runtime(dynamic::RuntimeError::TypeError(
              format!("{}", fact),
              tuple_type.clone(),
            )));
          }
        }
      } else {
        return Err(IntegrateError::Runtime(dynamic::RuntimeError::UnknownRelation(
          predicate.to_string(),
        )));
      }
    }

    // Actually insert
    self
      .internal
      .exec_ctx
      .add_facts_with_disjunction(predicate, facts, disjunctions);
    Ok(())
  }

  /// Compile the front context into back
  pub fn compile(&mut self) -> Result<(), IntegrateError> {
    self.compile_with_output_relations(None)
  }

  /// Compile the front context into back
  pub fn compile_with_output_relations(&mut self, outputs: Option<Vec<&str>>) -> Result<(), IntegrateError> {
    if self.front_has_changed {
      // First convert front to back
      let mut back_ir = self.front_ctx.to_back_program();

      // Make sure that back ir only outputting required relations
      if let Some(outputs) = outputs {
        back_ir.set_output_relations(outputs)
      }

      // Apply back optimizations
      if let Err(e) = back_ir.apply_optimizations(&self.options) {
        return Err(IntegrateError::Compile(vec![compiler::CompileError::Back(e)]));
      }

      // Then convert back to ram
      let mut ram = match back_ir.to_ram_program(&self.options) {
        Ok(ram) => ram,
        Err(e) => {
          return Err(IntegrateError::Compile(vec![compiler::CompileError::Back(e)]));
        }
      };

      // Optimize the ram
      compiler::ram::optimizations::optimize_ram(&mut ram);

      // Store the ram
      self.internal.ram_program = ram;

      // Set front_has_changed to false
      self.front_has_changed = false;
    }

    // Return success
    Ok(())
  }

  pub fn ram(&self) -> &compiler::ram::Program {
    &self.internal.ram_program
  }

  /// Execute the program in its current state, with a limit set on iteration count
  pub fn run_with_iter_limit_and_monitor<M>(&mut self, iter_limit: Option<usize>, m: &M) -> Result<(), IntegrateError>
  where
    Prov::InputTag: FromInputTag,
    M: Monitor<Prov>,
  {
    // First compile the code
    self.compile()?;

    // Finally execute the ram
    self.internal.run_with_iter_limit_and_monitor(iter_limit, m)
  }

  /// Execute the program in its current state
  ///
  /// Note: the results should be inspected using `relation` function
  pub fn run_with_monitor<M>(&mut self, m: &M) -> Result<(), IntegrateError>
  where
    Prov::InputTag: FromInputTag,
    M: Monitor<Prov>,
  {
    self.run_with_iter_limit_and_monitor(None, m)
  }

  /// Execute the program in its current state, with a limit set on iteration count
  pub fn run_with_iter_limit(&mut self, iter_limit: Option<usize>) -> Result<(), IntegrateError>
  where
    Prov::InputTag: FromInputTag,
  {
    // First compile the code
    self.compile()?;

    // Finally execute the ram
    self.internal.run_with_iter_limit(iter_limit)
  }

  /// Execute the program in its current state
  ///
  /// Note: the results should be inspected using `relation` function
  pub fn run(&mut self) -> Result<(), IntegrateError>
  where
    Prov::InputTag: FromInputTag,
  {
    self.run_with_iter_limit(None)
  }

  /// Get the relation type
  pub fn relation_type(&self, relation: &str) -> Option<TupleType> {
    self.front_ctx.relation_tuple_type(relation)
  }

  /// Has relation
  pub fn has_relation(&self, relation: &str) -> bool {
    self.front_ctx.has_relation(relation)
  }

  /// Get the number user defined relations
  pub fn num_relations(&self) -> usize {
    self.front_ctx.num_relations()
  }

  /// Get the number of all relations
  pub fn num_all_relations(&self) -> usize {
    self.internal.num_all_relations()
  }

  /// Get the user defined relations
  pub fn relations(&self) -> Vec<String> {
    self.front_ctx.relations()
  }

  /// Get all relations (including hidden ones)
  pub fn all_relations(&self) -> Vec<String> {
    self.internal.all_relations()
  }

  /// Check if a relation is computed
  pub fn is_computed(&self, relation: &str) -> bool {
    self.internal.is_computed(relation)
  }

  /// Get the relation output collection of a given relation
  pub fn computed_relation(&mut self, relation: &str) -> Option<&dynamic::DynamicOutputCollection<Prov>> {
    self.internal.computed_relation(relation)
  }

  /// Get the relation output collection of a given relation
  pub fn computed_rc_relation(
    &mut self,
    relation: &str,
  ) -> Option<P::Pointer<dynamic::DynamicOutputCollection<Prov>>> {
    self.internal.computed_rc_relation(relation)
  }

  /// Get the relation output collection of a given relation
  pub fn computed_rc_relation_with_monitor<M>(
    &mut self,
    relation: &str,
    m: &M,
  ) -> Option<P::Pointer<dynamic::DynamicOutputCollection<Prov>>>
  where
    M: Monitor<Prov>,
  {
    self.internal.computed_rc_relation_with_monitor(relation, m)
  }

  /// Get the Reference Counted (RC) version of the collection
  pub fn computed_internal_rc_relation(
    &self,
    relation: &str,
  ) -> Option<P::Pointer<dynamic::DynamicCollection<Prov>>> {
    self.internal.computed_internal_rc_relation(relation)
  }
}

pub struct CompiledIntegrateContext<Prov: Provenance, P: PointerFamily> {
  prov_ctx: Prov,
  ram_program: compiler::ram::Program,
  exec_ctx: dynamic::DynamicExecutionContext<Prov, P>,
  computed_output_relations: HashMap<String, P::Pointer<dynamic::DynamicOutputCollection<Prov>>>,
}

impl<Prov: Provenance, P: PointerFamily> Clone for CompiledIntegrateContext<Prov, P> {
  fn clone(&self) -> Self {
    Self {
      prov_ctx: self.prov_ctx.clone(),
      ram_program: self.ram_program.clone(),
      exec_ctx: self.exec_ctx.clone(),
      computed_output_relations: self
        .computed_output_relations
        .iter()
        .map(|(r, rc)| (r.clone(), P::new(P::get(rc).clone())))
        .collect::<HashMap<String, _>>(),
    }
  }
}

impl<Prov: Provenance, P: PointerFamily> CompiledIntegrateContext<Prov, P> {
  pub fn provenance_context(&self) -> &Prov {
    &self.prov_ctx
  }

  /// Add a list of facts to the given predicate
  pub fn add_facts(
    &mut self,
    predicate: &str,
    facts: Vec<(Option<Prov::InputTag>, Tuple)>,
    type_check: bool,
  ) -> Result<(), IntegrateError> {
    self.add_facts_with_disjunction(predicate, facts, None, type_check)
  }

  /// Add a list of facts to the given predicate, with the option to pass in disjunctions
  pub fn add_facts_with_disjunction(
    &mut self,
    predicate: &str,
    facts: Vec<(Option<Prov::InputTag>, Tuple)>,
    disjunctions: Option<Vec<Vec<usize>>>,
    type_check: bool,
  ) -> Result<(), IntegrateError> {
    // Check type
    if type_check {
      let pred_tuple_type = self.ram_program.relation_tuple_type(predicate);
      if let Some(tuple_type) = pred_tuple_type {
        for (_, fact) in &facts {
          if fact.tuple_type() != tuple_type {
            return Err(IntegrateError::Runtime(dynamic::RuntimeError::TypeError(
              format!("{}", fact),
              tuple_type.clone(),
            )));
          }
        }
      } else {
        return Err(IntegrateError::Runtime(dynamic::RuntimeError::UnknownRelation(
          predicate.to_string(),
        )));
      }
    }

    // Actually insert
    self.exec_ctx.add_facts_with_disjunction(predicate, facts, disjunctions);
    Ok(())
  }

  /// Execute the program in its current state, with a limit set on iteration count
  pub fn run_with_iter_limit_and_monitor<M>(&mut self, iter_limit: Option<usize>, m: &M) -> Result<(), IntegrateError>
  where
    Prov::InputTag: FromInputTag,
    M: Monitor<Prov>,
  {
    // Finally execute the ram
    self
      .exec_ctx
      .execute_with_iter_limit_and_monitor(self.ram_program.clone(), &mut self.prov_ctx, iter_limit, m)
      .map_err(IntegrateError::Runtime)?;

    // Success
    Ok(())
  }

  /// Execute the program in its current state, with a limit set on iteration count
  pub fn run_with_iter_limit(&mut self, iter_limit: Option<usize>) -> Result<(), IntegrateError>
  where
    Prov::InputTag: FromInputTag,
  {
    // Finally execute the ram
    self
      .exec_ctx
      .execute_with_iter_limit(self.ram_program.clone(), &mut self.prov_ctx, iter_limit)
      .map_err(IntegrateError::Runtime)?;

    // Success
    Ok(())
  }

  /// Execute the program in its current state
  ///
  /// Note: the results should be inspected using `relation` function
  pub fn run_with_monitor<M>(&mut self, m: &M) -> Result<(), IntegrateError>
  where
    Prov::InputTag: FromInputTag,
    M: Monitor<Prov>,
  {
    self.run_with_iter_limit_and_monitor(None, m)
  }

  /// Execute the program in its current state
  ///
  /// Note: the results should be inspected using `relation` function
  pub fn run(&mut self) -> Result<(), IntegrateError>
  where
    Prov::InputTag: FromInputTag,
  {
    self.run_with_iter_limit(None)
  }

  /// Get the number of all relations
  pub fn num_all_relations(&self) -> usize {
    self.exec_ctx.num_relations()
  }

  /// Get all relations (including hidden ones)
  pub fn all_relations(&self) -> Vec<String> {
    self.exec_ctx.relations()
  }

  /// Check if a relation is computed
  pub fn is_computed(&self, relation: &str) -> bool {
    self.exec_ctx.is_computed(relation)
  }

  fn recover_relation_and_cache(&mut self, relation: &str) {
    // If not already computed, recover output collection from execution context, and cache it
    if !self.computed_output_relations.contains_key(relation) {
      if let Some(rc) = self.exec_ctx.relation(relation, self.provenance_context()) {
        self.computed_output_relations.insert(relation.to_string(), P::new(rc));
      }
    }
  }

  fn recover_relation_and_cache_with_monitor<M>(&mut self, relation: &str, m: &M)
  where
    M: Monitor<Prov>,
  {
    // If not already computed, recover output collection from execution context, and cache it
    if !self.computed_output_relations.contains_key(relation) {
      if let Some(rc) = self
        .exec_ctx
        .relation_with_monitor(relation, self.provenance_context(), m)
      {
        self.computed_output_relations.insert(relation.to_string(), P::new(rc));
      }
    }
  }

  pub fn computed_relation(&mut self, relation: &str) -> Option<&dynamic::DynamicOutputCollection<Prov>> {
    self.recover_relation_and_cache(relation);
    self.computed_output_relations.get(relation).map(|p| P::get(p))
  }

  /// Get the RC'ed output collection of a given relation
  pub fn computed_rc_relation(
    &mut self,
    relation: &str,
  ) -> Option<P::Pointer<dynamic::DynamicOutputCollection<Prov>>> {
    self.recover_relation_and_cache(relation);
    self.computed_output_relations.get(relation).map(|p| P::clone_ptr(p))
  }

  /// Get the RC'ed output collection of a given relation
  pub fn computed_rc_relation_with_monitor<M>(
    &mut self,
    relation: &str,
    m: &M,
  ) -> Option<P::Pointer<dynamic::DynamicOutputCollection<Prov>>>
  where
    M: Monitor<Prov>,
  {
    self.recover_relation_and_cache_with_monitor(relation, m);
    self.computed_output_relations.get(relation).map(|p| P::clone_ptr(p))
  }

  /// Get the Reference Counted (RC) version of the collection
  pub fn computed_internal_rc_relation(
    &self,
    relation: &str,
  ) -> Option<P::Pointer<dynamic::DynamicCollection<Prov>>> {
    self.exec_ctx.internal_rc_relation(relation)
  }
}

#[derive(Clone, Debug)]
pub enum IntegrateError {
  Compile(Vec<compiler::CompileError>),
  Runtime(dynamic::RuntimeError),
}

impl IntegrateError {
  pub fn front(e: compiler::front::FrontCompileError) -> Self {
    Self::Compile(vec![compiler::CompileError::Front(e)])
  }
}

impl std::fmt::Display for IntegrateError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Compile(errs) => {
        for (i, e) in errs.iter().enumerate() {
          if i > 0 {
            f.write_str("\n")?;
          }
          std::fmt::Display::fmt(e, f)?;
        }
        Ok(())
      }
      Self::Runtime(e) => std::fmt::Display::fmt(e, f),
    }
  }
}
