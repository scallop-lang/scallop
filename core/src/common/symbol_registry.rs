use std::collections::*;

#[derive(Clone, Debug)]
pub struct SymbolRegistry {
  pub symbol_to_id_map: BTreeMap<String, usize>,
  pub id_to_symbol_map: Vec<String>,
}

impl SymbolRegistry {
  /// Create a new symbol registry
  pub fn new() -> Self {
    Self {
      symbol_to_id_map: BTreeMap::new(),
      id_to_symbol_map: Vec::new(),
    }
  }

  /// Check if the registry is empty
  pub fn is_empty(&self) -> bool {
    self.id_to_symbol_map.is_empty()
  }

  /// Register a symbol into the registry and return an ID to represent the symbol
  pub fn register(&mut self, symbol: String) -> usize {
    if let Some(id) = self.symbol_to_id_map.get(&symbol) {
      id.clone()
    } else {
      let id = self.id_to_symbol_map.len();
      self.symbol_to_id_map.insert(symbol.clone(), id);
      self.id_to_symbol_map.push(symbol);
      id
    }
  }

  /// Getting the symbol id using the string
  pub fn get_id(&self, symbol: &str) -> Option<usize> {
    self.symbol_to_id_map.get(symbol).cloned()
  }

  /// Getting a symbol using its ID
  pub fn get_symbol(&self, id: usize) -> Option<&String> {
    self.id_to_symbol_map.get(id)
  }
}
