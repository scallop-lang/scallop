use std::path::PathBuf;

use crate::common;
use crate::compiler;
use crate::runtime::database;
use crate::runtime::dynamic;
use crate::runtime::env;
use crate::runtime::monitor;
use crate::runtime::provenance;
use crate::utils::*;

use super::*;

#[derive(Clone)]
pub struct InterpretContext<Prov: provenance::Provenance, Ptr: PointerFamily = RcFamily> {
  pub provenance: Prov,
  pub runtime_env: env::RuntimeEnvironment,
  pub execution_context: dynamic::DynamicExecutionContext<Prov, Ptr>,
}

impl<Prov: provenance::Provenance, Ptr: PointerFamily> InterpretContext<Prov, Ptr> {
  pub fn new(program: String, provenance: Prov) -> Result<Self, IntegrateError> {
    Self::new_with_options(program, provenance, IntegrateOptions::default())
  }

  pub fn new_with_options(
    program_string: String,
    provenance: Prov,
    options: IntegrateOptions,
  ) -> Result<Self, IntegrateError> {
    let program = compiler::compile_string_to_ram_with_options(program_string, &options.compiler_options)
      .map_err(IntegrateError::Compile)?;
    let runtime_env = options.runtime_environment_options.build();
    let execution_context =
      dynamic::DynamicExecutionContext::new_with_program_and_options(program, options.execution_options);
    Ok(Self {
      provenance,
      runtime_env,
      execution_context,
    })
  }

  pub fn new_from_file(file_name: &PathBuf, provenance: Prov) -> Result<Self, IntegrateError> {
    Self::new_from_file_with_options(file_name, provenance, IntegrateOptions::default())
  }

  pub fn new_from_file_with_options(
    file_name: &PathBuf,
    provenance: Prov,
    options: IntegrateOptions,
  ) -> Result<Self, IntegrateError> {
    let program = compiler::compile_file_to_ram_with_options(file_name, &options.compiler_options)
      .map_err(IntegrateError::Compile)?;
    let runtime_env = options.runtime_environment_options.build();
    let execution_context =
      dynamic::DynamicExecutionContext::new_with_program_and_options(program, options.execution_options);
    Ok(Self {
      provenance,
      runtime_env,
      execution_context,
    })
  }

  pub fn edb(&mut self) -> &mut database::extensional::ExtensionalDatabase<Prov> {
    &mut self.execution_context.edb
  }

  pub fn run(&mut self) -> Result<(), IntegrateError> {
    // Execute the program
    self
      .execution_context
      .execute(&mut self.runtime_env, &mut self.provenance)
      .map_err(IntegrateError::Runtime)?;

    // Recover output collections
    for (predicate, relation) in &mut self.execution_context.idb {
      use common::output_option::*;
      match self.execution_context.program.output_option(predicate).unwrap() {
        // Unwrap because predicate is absolutely part of the program
        OutputOption::Hidden => {}
        OutputOption::Default => {
          relation.recover(&self.provenance, true);
        }
        OutputOption::File(_) => {
          unimplemented!("Cannot output into file for now")
        }
      }
    }

    // Retain only related collections
    let to_keep_relations = self
      .execution_context
      .program
      .relations()
      .filter(|r| r.output.is_default())
      .map(|r| r.predicate.clone())
      .collect();
    self.execution_context.idb.retain_relations(&to_keep_relations);

    // Return Ok!
    Ok(())
  }

  pub fn run_with_monitor<M: monitor::Monitor<Prov>>(&mut self, m: &M) -> Result<(), IntegrateError> {
    // Execute the program
    self
      .execution_context
      .execute_with_monitor(&mut self.runtime_env, &mut self.provenance, m)
      .map_err(IntegrateError::Runtime)?;

    // Recover output collections
    for (predicate, relation) in &mut self.execution_context.idb {
      use common::output_option::*;
      match self.execution_context.program.output_option(predicate).unwrap() {
        // Unwrap because predicate is absolutely part of the program
        OutputOption::Hidden => {}
        OutputOption::Default => {
          relation.recover_with_monitor(&self.provenance, m, true);
        }
        OutputOption::File(_) => {
          unimplemented!("Cannot output into file for now")
        }
      }
    }

    // Retain only related collections
    let to_keep_relations = self
      .execution_context
      .program
      .relations()
      .filter(|r| r.output.is_default())
      .map(|r| r.predicate.clone())
      .collect();
    self.execution_context.idb.retain_relations(&to_keep_relations);

    // Return Ok!
    Ok(())
  }

  pub fn idb(self) -> database::intentional::IntentionalDatabase<Prov, Ptr> {
    self.execution_context.idb
  }
}
