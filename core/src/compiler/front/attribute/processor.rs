use dyn_clone::DynClone;

use super::super::ast;

use super::*;

pub trait AttributeProcessor: DynClone + 'static {
  fn name(&self) -> String;

  fn apply(&self, item: &ast::Item, attr: &ast::Attribute) -> Result<AttributeAction, AttributeError>;
}

impl std::fmt::Debug for dyn AttributeProcessor {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(&self.name())
  }
}

#[derive(Debug)]
pub struct DynamicAttributeProcessor {
  proc: Box<dyn AttributeProcessor>,
}

impl DynamicAttributeProcessor {
  pub fn new<P: AttributeProcessor>(p: P) -> Self {
    Self { proc: Box::new(p) }
  }
}

impl AttributeProcessor for DynamicAttributeProcessor {
  fn name(&self) -> String {
    self.proc.name()
  }

  fn apply(&self, item: &ast::Item, attr: &ast::Attribute) -> Result<AttributeAction, AttributeError> {
    self.proc.apply(item, attr)
  }
}

impl Clone for DynamicAttributeProcessor {
  fn clone(&self) -> Self {
    Self {
      proc: dyn_clone::clone_box(&*self.proc),
    }
  }
}
