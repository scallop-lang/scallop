use super::super::*;
use crate::compiler::back;

impl FrontContext {
  pub fn back_relation_attributes(&self, relation: &String) -> back::Attributes {
    let mut attrs = back::Attributes::new();

    // Check demand attributes
    if let Some(pattern) = self.analysis.borrow().demand_attr_analysis.demand_pattern(relation) {
      attrs.add_attribute(back::Attribute::Demand(back::DemandAttribute {
        pattern: pattern.clone(),
      }));
    }

    // Check input files
    if let Some(input_file) = self.analysis.borrow().input_files_analysis.input_file(relation) {
      attrs.add_attribute(back::Attribute::InputFile(back::InputFileAttribute {
        input_file: input_file.clone(),
      }));
    }

    attrs
  }
}
