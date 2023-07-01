use std::sync::*;

use crate::common::symbol_registry::*;

/// A symbol registry with shared internal mutability
#[derive(Clone, Debug)]
pub struct SymbolRegistry2 {
  pub registry: Arc<Mutex<SymbolRegistry>>,
}

impl SymbolRegistry2 {
  pub fn new() -> Self {
    Self {
      registry: Arc::new(Mutex::new(SymbolRegistry::new())),
    }
  }

  pub fn is_empty(&self) -> bool {
    self.registry.lock().unwrap().is_empty()
  }

  pub fn register(&self, symbol: String) -> usize {
    self.registry.lock().unwrap().register(symbol)
  }

  pub fn get_id(&self, symbol: &str) -> Option<usize> {
    self.registry.lock().unwrap().get_id(symbol)
  }

  pub fn get_symbol(&self, id: usize) -> Option<String> {
    self.registry.lock().unwrap().get_symbol(id).cloned()
  }
}
