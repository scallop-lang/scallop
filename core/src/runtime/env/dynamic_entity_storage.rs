use std::collections::*;
use std::sync::*;

use crate::common::adt_variant_registry::*;
use crate::common::tuple::*;
use crate::common::value::*;

/// The runtime data structure for processing dynamic entities
#[derive(Clone, Debug)]
pub struct DynamicEntityStorage {
  pub registry: ADTVariantRegistry,
  pub entities: HashMap<String, HashMap<Value, Vec<Tuple>>>,
}

impl DynamicEntityStorage {
  pub fn new() -> Self {
    Self {
      registry: ADTVariantRegistry::new(),
      entities: HashMap::new(),
    }
  }

  pub fn update_variant_registry(&mut self, registry: ADTVariantRegistry) {
    self.registry = registry;
  }

  pub fn registry(&self) -> &ADTVariantRegistry {
    &self.registry
  }

  pub fn compile_and_add_entity_string(&mut self, s: &str) -> Result<Value, ADTEntityError> {
    // First parse the string and produce the final entity along with the facts
    let result = self.registry.parse(s)?;

    // Add the facts to the new_entities registry
    for fact in result.facts {
      self.add_entity_fact(&fact.0, fact.1, Tuple::from_values(fact.2.into_iter()));
    }

    // Return the entity value itself
    Ok(result.entity)
  }

  pub fn add_entity_fact(&mut self, functor: &str, value: Value, tuple: Tuple) {
    self
      .entities
      .entry(functor.to_string())
      .or_default()
      .entry(value)
      .or_default()
      .push(tuple);
  }

  pub fn drain_entities<F: Fn(&str) -> bool>(&mut self, relation_filter: F) -> HashMap<String, Vec<Tuple>> {
    self
      .entities
      .extract_if(|k, _| relation_filter(k))
      .map(|(relation_name, id_tuple_map)| {
        let tuples = id_tuple_map
          .into_iter()
          .flat_map(|(id, tuples)| {
            tuples
              .into_iter()
              .map(move |tuple| Tuple::from_values(std::iter::once(id.clone()).chain(tuple.as_values())))
          })
          .collect::<Vec<_>>();
        (relation_name, tuples)
      })
      .collect()
  }
}

#[derive(Debug)]
pub struct DynamicEntityStorage2 {
  storage: Mutex<DynamicEntityStorage>,
}

impl Clone for DynamicEntityStorage2 {
  fn clone(&self) -> Self {
    Self {
      storage: Mutex::new(self.storage.lock().unwrap().clone()),
    }
  }
}

impl DynamicEntityStorage2 {
  pub fn new() -> Self {
    Self {
      storage: Mutex::new(DynamicEntityStorage::new()),
    }
  }

  pub fn update_variant_registry(&self, registry: ADTVariantRegistry) {
    self.storage.lock().unwrap().update_variant_registry(registry);
  }

  pub fn with_registry<T, F: FnOnce(&ADTVariantRegistry) -> T>(&self, f: F) -> T {
    f(self.storage.lock().unwrap().registry())
  }

  pub fn compile_and_add_entity_string(&self, s: &str) -> Result<Value, ADTEntityError> {
    self.storage.lock().unwrap().compile_and_add_entity_string(s)
  }

  pub fn add_entity_fact(&self, functor: &str, id: Value, tuple: Tuple) {
    self.storage.lock().unwrap().add_entity_fact(functor, id, tuple)
  }

  pub fn drain_entities<F: Fn(&str) -> bool>(&self, f: F) -> HashMap<String, Vec<Tuple>> {
    self.storage.lock().unwrap().drain_entities(f)
  }
}
