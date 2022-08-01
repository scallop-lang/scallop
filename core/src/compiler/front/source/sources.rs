use std::rc::Rc;

use super::*;

pub struct Sources {
  pub sources: Vec<Rc<dyn Source>>,
}

impl Sources {
  pub fn new() -> Self {
    Self { sources: vec![] }
  }

  pub fn add<S: Source>(&mut self, s: S) -> usize {
    let id = self.sources.len();
    self.sources.push(Rc::new(s));
    id
  }

  pub fn last(&self) -> &Rc<dyn Source> {
    self.sources.last().unwrap()
  }
}

impl Clone for Sources {
  fn clone(&self) -> Self {
    Self {
      sources: self.sources.iter().map(|s| Rc::clone(s)).collect(),
    }
  }
}

impl std::fmt::Debug for Sources {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_set().entries(self.sources.iter().map(|s| &*s)).finish()
  }
}

impl std::ops::Index<usize> for Sources {
  type Output = Rc<dyn Source>;

  fn index(&self, index: usize) -> &Self::Output {
    &self.sources[index]
  }
}
