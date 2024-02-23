use std::collections::*;

use crate::compiler::front::FrontContext;

use super::super::ast;
use super::*;

const RESERVED_ATTRIBUTES: [&'static str; 4] = ["hidden", "file", "demand", "tagged"];

#[derive(Clone, Debug)]
pub struct AttributeProcessorRegistry {
  pub registry: HashMap<String, DynamicAttributeProcessor>,
}

impl AttributeProcessorRegistry {
  pub fn new() -> Self {
    Self {
      registry: HashMap::new(),
    }
  }

  pub fn has_attribute_processor(&self, name: &str) -> bool {
    self.registry.contains_key(name)
  }

  pub fn get_attribute_processor(&self, name: &str) -> Option<&DynamicAttributeProcessor> {
    self.registry.get(name)
  }

  pub fn register<P>(&mut self, p: P) -> Result<(), AttributeError>
  where
    P: AttributeProcessor,
  {
    let name = p.name();
    if RESERVED_ATTRIBUTES.contains(&name.as_str()) {
      Err(AttributeError::ReservedAttribute { name })
    } else if self.registry.contains_key(&name) {
      Err(AttributeError::DuplicatedAttributeProcessor { name })
    } else {
      let dyn_p = DynamicAttributeProcessor::new(p);
      self.registry.insert(name.to_string(), dyn_p);
      Ok(())
    }
  }

  pub fn analyze(&self, items: &Vec<ast::Item>) -> Result<PostProcessingContext, AttributeError> {
    let mut post_proc_ctx = PostProcessingContext::new();
    for (item_index, item) in items.iter().enumerate() {
      for attr in item.attrs() {
        let attr_name = attr.name().name();
        if let Some(proc) = self.get_attribute_processor(attr_name) {
          let action = proc.apply(item, attr)?;
          post_proc_ctx.add_action(action, item_index);
        } else if !RESERVED_ATTRIBUTES.contains(&attr_name.as_str()) {
          return Err(AttributeError::UnknownAttribute {
            name: attr_name.clone(),
          });
        }
      }
    }
    Ok(post_proc_ctx)
  }

  pub fn analyze_and_process(&self, ctx: &mut FrontContext, items: &mut Vec<ast::Item>) -> Result<(), AttributeError> {
    let attr_pos_proc = self.analyze(items)?;
    attr_pos_proc.process(ctx, items)
  }
}
