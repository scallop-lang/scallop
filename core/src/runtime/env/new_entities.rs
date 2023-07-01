use std::collections::*;
use std::sync::*;

use crate::common::tuple::*;
use crate::common::value::*;

#[derive(Clone, Debug)]
pub struct NewEntitiesStorage {
  pub entities: HashMap<String, HashMap<Value, Vec<Tuple>>>,
}

impl NewEntitiesStorage {
  pub fn new() -> Self {
    Self {
      entities: HashMap::new(),
    }
  }

  pub fn add(&mut self, functor: &str, value: Value, tuple: Tuple) {
    self
      .entities
      .entry(functor.to_string())
      .or_default()
      .entry(value)
      .or_default()
      .push(tuple);
  }

  pub fn drain_entities(&mut self) -> HashMap<String, Vec<Tuple>> {
    // Create an empty dictionary and swap it with the internal one
    let mut entities = HashMap::new();
    std::mem::swap(&mut self.entities, &mut entities);

    // Post-process the entities into vector of tuples
    entities
      .into_iter()
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
pub struct NewEntitiesStorage2 {
  storage: Mutex<NewEntitiesStorage>,
}

impl Clone for NewEntitiesStorage2 {
  fn clone(&self) -> Self {
    Self {
      storage: Mutex::new(self.storage.lock().unwrap().clone()),
    }
  }
}

impl NewEntitiesStorage2 {
  pub fn new() -> Self {
    Self {
      storage: Mutex::new(NewEntitiesStorage::new()),
    }
  }

  pub fn add(&self, functor: &str, id: Value, tuple: Tuple) {
    self.storage.lock().unwrap().add(functor, id, tuple)
  }

  pub fn drain_entities(&self) -> HashMap<String, Vec<Tuple>> {
    self.storage.lock().unwrap().drain_entities()
  }
}
