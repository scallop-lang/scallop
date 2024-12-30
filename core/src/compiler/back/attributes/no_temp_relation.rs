use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct NoTemporaryRelationAttribute;

impl AttributeTrait for NoTemporaryRelationAttribute {
  fn name(&self) -> String {
    "no_temp_relation".to_string()
  }
}
