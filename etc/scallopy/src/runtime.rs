use scallop_core::runtime::env::*;

#[derive(Clone)]
pub struct PythonRuntimeEnvironment {
  pub symbol_registry: SymbolRegistry2,
  pub tensor_registry: TensorRegistry2,
  pub dynamic_entity_store: DynamicEntityStorage2,
}

impl<'a> From<&'a RuntimeEnvironment> for PythonRuntimeEnvironment {
  fn from(env: &'a RuntimeEnvironment) -> Self {
    Self {
      symbol_registry: env.symbol_registry.clone(),
      tensor_registry: env.tensor_registry.clone(),
      dynamic_entity_store: env.dynamic_entity_store.clone(),
    }
  }
}
