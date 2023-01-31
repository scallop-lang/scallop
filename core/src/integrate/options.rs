use crate::compiler::CompileOptions;
use crate::runtime::dynamic::ExecutionOptions;
use crate::runtime::env::RuntimeEnvironmentOptions;

#[derive(Clone, Debug, Default)]
pub struct IntegrateOptions {
  pub compiler_options: CompileOptions,
  pub execution_options: ExecutionOptions,
  pub runtime_environment_options: RuntimeEnvironmentOptions,
}

impl IntegrateOptions {
  pub fn new() -> Self {
    Self::default()
  }
}
