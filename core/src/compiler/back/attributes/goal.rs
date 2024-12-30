use super::*;

#[derive(Debug, Clone)]
pub struct GoalAttribute;

impl AttributeTrait for GoalAttribute {
  fn name(&self) -> String {
    "goal".to_string()
  }
}
