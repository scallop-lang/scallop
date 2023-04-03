use crate::common::foreign_function::*;
use crate::common::foreign_predicate::*;
use crate::common::tuple::*;
use crate::common::tuple_type::*;

use crate::compiler;
use crate::runtime::database::extensional::*;
use crate::runtime::database::*;
use crate::runtime::dynamic;
use crate::runtime::env::*;
use crate::runtime::error::*;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;
use crate::utils::*;

use super::*;

#[derive(Clone)]
pub struct IntegrateContext<Prov: Provenance, P: PointerFamily = RcFamily> {
  /// The compile options
  options: compiler::CompileOptions,

  /// The compilation context, in particular the Front-IR compilation context
  ///
  /// This is for incremental compilation
  front_ctx: compiler::front::FrontContext,

  /// Flag denoting whether the Front-IR has changed or not; initialized to not changed.
  /// Once the front is compiled and stayed unchanged, no further analysis will be performed
  /// on the front compilation context
  front_has_changed: bool,

  /// The internal integrate context to be separated from the compilation
  internal: InternalIntegrateContext<Prov, P>,
}

impl<Prov: Provenance, P: PointerFamily> IntegrateContext<Prov, P> {
  pub fn new(prov_ctx: Prov) -> Self {
    Self {
      options: compiler::CompileOptions::default(),
      front_ctx: compiler::front::FrontContext::new(),
      internal: InternalIntegrateContext {
        prov_ctx,
        runtime_env: RuntimeEnvironment::default(),
        ram_program: compiler::ram::Program::new(),
        exec_ctx: dynamic::DynamicExecutionContext::new_with_options(dynamic::ExecutionOptions {
          type_check: false,
          ..Default::default()
        }),
      },
      front_has_changed: false,
    }
  }

  pub fn new_incremental(prov_ctx: Prov) -> Self {
    Self {
      options: compiler::CompileOptions::default(),
      front_ctx: compiler::front::FrontContext::new(),
      internal: InternalIntegrateContext {
        prov_ctx,
        runtime_env: RuntimeEnvironment::default(),
        ram_program: compiler::ram::Program::new(),
        exec_ctx: dynamic::DynamicExecutionContext::new_with_options(dynamic::ExecutionOptions {
          type_check: false,
          incremental_maintain: true,
          ..Default::default()
        }),
      },
      front_has_changed: false,
    }
  }

  pub fn new_with_options(prov_ctx: Prov, options: IntegrateOptions) -> Self {
    Self {
      options: options.compiler_options,
      front_ctx: compiler::front::FrontContext::new(),
      internal: InternalIntegrateContext {
        prov_ctx,
        runtime_env: RuntimeEnvironment::default(),
        ram_program: compiler::ram::Program::new(),
        exec_ctx: dynamic::DynamicExecutionContext::new_with_options(dynamic::ExecutionOptions {
          type_check: false,
          ..options.execution_options
        }),
      },
      front_has_changed: false,
    }
  }

  pub fn provenance_context(&self) -> &Prov {
    &self.internal.prov_ctx
  }

  pub fn provenance_context_mut(&mut self) -> &mut Prov {
    &mut self.internal.prov_ctx
  }

  pub fn internal_context(&self) -> &InternalIntegrateContext<Prov, P> {
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
      self
        .internal
        .exec_ctx
        .add_facts(&pred, vec![(tag, Tuple::empty())])
        .map_err(|e| IntegrateError::Runtime(RuntimeError::Database(e)))?;
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
    // Check type
    if type_check {
      let pred_tuple_type = self.front_ctx.relation_tuple_type(predicate);
      if let Some(tuple_type) = pred_tuple_type {
        for (_, tuple) in &facts {
          if !tuple_type.matches(&tuple) {
            return Err(IntegrateError::Runtime(RuntimeError::Database(
              DatabaseError::TypeError {
                relation: predicate.to_string(),
                relation_type: tuple_type.clone(),
                tuple: tuple.clone(),
              },
            )));
          }
        }
      } else {
        return Err(IntegrateError::Runtime(RuntimeError::Database(
          DatabaseError::UnknownRelation {
            relation: predicate.to_string(),
          },
        )));
      }
    }

    // Actually insert
    self
      .internal
      .exec_ctx
      .add_facts(predicate, facts)
      .map_err(|e| IntegrateError::Runtime(RuntimeError::Database(e)))?;
    Ok(())
  }

  /// Register a foreign function to the context
  pub fn register_foreign_function<F>(&mut self, ff: F) -> Result<(), IntegrateError>
  where
    F: ForeignFunction + Send + Sync + 'static,
  {
    // Add the function to front compilation context
    self
      .front_ctx
      .register_foreign_function(ff)
      .map_err(|e| IntegrateError::Runtime(RuntimeError::ForeignFunction(e)))?;

    // If goes through, then the front context has changed
    self.front_has_changed = true;

    // Return Ok
    Ok(())
  }

  /// Register a foreign predicate to the context
  pub fn register_foreign_predicate<F>(&mut self, fp: F) -> Result<(), IntegrateError>
  where
    F: ForeignPredicate + Send + Sync + Clone + 'static,
  {
    // Add the predicate to front compilation context
    self
      .front_ctx
      .register_foreign_predicate(fp)
      .map_err(|e| IntegrateError::Runtime(RuntimeError::ForeignPredicate(e)))?;

    // If goes through, then the front context has changed
    self.front_has_changed = true;

    // Return Ok
    Ok(())
  }

  /// Set the context to be non-incremental anymore
  pub fn set_non_incremental(&mut self) {
    self.internal.exec_ctx.set_non_incremental();
  }

  /// Set whether to perform early discard
  pub fn set_early_discard(&mut self, early_discard: bool) {
    self.internal.runtime_env.set_early_discard(early_discard)
  }

  /// Set the iteration limit
  pub fn set_iter_limit(&mut self, k: usize) {
    self.internal.runtime_env.set_iter_limit(k)
  }

  /// Remove the iteration limit
  pub fn remove_iter_limit(&mut self) {
    self.internal.runtime_env.remove_iter_limit()
  }

  /// Get a mutable refernce to the Extensional Database (EDB)
  pub fn edb(&mut self) -> &mut ExtensionalDatabase<Prov> {
    &mut self.internal.exec_ctx.edb
  }

  /// Compile the front context into back
  pub fn compile(&mut self) -> Result<(), IntegrateError> {
    self.compile_with_output_relations(None)?;
    Ok(())
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
  pub fn run_with_monitor<M>(&mut self, m: &M) -> Result<(), IntegrateError>
  where
    M: Monitor<Prov>,
  {
    // First compile the code
    self.compile()?;

    // Finally execute the ram
    self.internal.run_with_monitor(m)
  }

  /// Execute the program in its current state, with a limit set on iteration count
  pub fn run(&mut self) -> Result<(), IntegrateError> {
    // First compile the code
    self.compile()?;

    // Finally execute the ram
    self.internal.run()
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
  pub fn computed_relation_ref(&mut self, relation: &str) -> Option<&dynamic::DynamicOutputCollection<Prov>> {
    self.internal.computed_relation_ref(relation)
  }

  /// Get the relation output collection of a given relation
  pub fn computed_relation(&mut self, relation: &str) -> Option<P::Rc<dynamic::DynamicOutputCollection<Prov>>> {
    self.internal.computed_relation(relation)
  }

  /// Get the relation output collection of a given relation
  pub fn computed_relation_with_monitor<M>(
    &mut self,
    relation: &str,
    m: &M,
  ) -> Option<P::Rc<dynamic::DynamicOutputCollection<Prov>>>
  where
    M: Monitor<Prov>,
  {
    self.internal.computed_relation_with_monitor(relation, m)
  }
}

pub struct InternalIntegrateContext<Prov: Provenance, P: PointerFamily> {
  /// The provenance context
  pub prov_ctx: Prov,

  /// The runtime environment
  pub runtime_env: RuntimeEnvironment,

  /// The ram program to be evaluated
  pub ram_program: compiler::ram::Program,

  /// The dynamic execution context; within which there are EDB, IDB, and a program
  ///
  /// Note that the `ram_program` stored in this struct and the `program` stored in
  /// the execution context could be different, in which case incremental evaluation
  /// will be performed -- not all relations will be recomputed.
  pub exec_ctx: dynamic::DynamicExecutionContext<Prov, P>,
}

impl<Prov: Provenance, P: PointerFamily> Clone for InternalIntegrateContext<Prov, P> {
  fn clone(&self) -> Self {
    Self {
      prov_ctx: self.prov_ctx.clone(),
      runtime_env: self.runtime_env.clone(),
      ram_program: self.ram_program.clone(),
      exec_ctx: self.exec_ctx.clone(),
    }
  }
}

impl<Prov: Provenance, P: PointerFamily> InternalIntegrateContext<Prov, P> {
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
    // Check type
    if type_check {
      let pred_tuple_type = self.ram_program.relation_tuple_type(predicate);
      if let Some(tuple_type) = pred_tuple_type {
        for (_, tuple) in &facts {
          if !tuple_type.matches(&tuple) {
            return Err(IntegrateError::Runtime(RuntimeError::Database(
              DatabaseError::TypeError {
                relation: predicate.to_string(),
                relation_type: tuple_type.clone(),
                tuple: tuple.clone(),
              },
            )));
          }
        }
      } else {
        return Err(IntegrateError::Runtime(RuntimeError::Database(
          DatabaseError::UnknownRelation {
            relation: predicate.to_string(),
          },
        )));
      }
    }

    // Actually insert
    self
      .exec_ctx
      .add_facts(predicate, facts)
      .map_err(|e| IntegrateError::Runtime(RuntimeError::Database(e)))?;
    Ok(())
  }

  /// Execute the program in its current state, with a limit set on iteration count
  pub fn run_with_monitor<M>(&mut self, m: &M) -> Result<(), IntegrateError>
  where
    M: Monitor<Prov>,
  {
    // Populate the runtime foreign function/predicate registry
    self.runtime_env.function_registry = self.ram_program.function_registry.clone();
    self.runtime_env.predicate_registry = self.ram_program.predicate_registry.clone();

    // Finally execute the ram
    self
      .exec_ctx
      .incremental_execute_with_monitor(self.ram_program.clone(), &mut self.runtime_env, &mut self.prov_ctx, m)
      .map_err(IntegrateError::Runtime)?;

    // Success
    Ok(())
  }

  /// Execute the program in its current state, with a limit set on iteration count
  pub fn run(&mut self) -> Result<(), IntegrateError> {
    // Populate the runtime foreign function/predicate registry
    self.runtime_env.function_registry = self.ram_program.function_registry.clone();
    self.runtime_env.predicate_registry = self.ram_program.predicate_registry.clone();

    // Finally execute the ram
    self
      .exec_ctx
      .incremental_execute(self.ram_program.clone(), &mut self.runtime_env, &mut self.prov_ctx)
      .map_err(IntegrateError::Runtime)?;

    // Success
    Ok(())
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

  pub fn computed_relation_ref(&mut self, relation: &str) -> Option<&dynamic::DynamicOutputCollection<Prov>> {
    self.exec_ctx.recover(relation, &self.prov_ctx);
    self.exec_ctx.relation_ref(relation)
  }

  /// Get the RC'ed output collection of a given relation
  pub fn computed_relation(&mut self, relation: &str) -> Option<P::Rc<dynamic::DynamicOutputCollection<Prov>>> {
    self.exec_ctx.recover(relation, &self.prov_ctx);
    self.exec_ctx.relation(relation)
  }

  /// Get the RC'ed output collection of a given relation
  pub fn computed_relation_with_monitor<M: Monitor<Prov>>(
    &mut self,
    relation: &str,
    m: &M,
  ) -> Option<P::Rc<dynamic::DynamicOutputCollection<Prov>>> {
    self.exec_ctx.recover_with_monitor(relation, &self.prov_ctx, m);
    self.exec_ctx.relation(relation)
  }
}
