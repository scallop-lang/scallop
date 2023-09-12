use super::*;

/// Parsing entity from string to an entity.
/// Note that this function will possibly populate the entities in the database.
///
/// ``` scl
/// extern fn $parse_entity(s: String) -> Entity
/// ```
#[derive(Clone)]
pub struct ParseEntity;

impl ForeignFunction for ParseEntity {
  fn name(&self) -> String {
    "parse_entity".to_string()
  }

  fn num_static_arguments(&self) -> usize {
    1
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert_eq!(i, 0);
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::Entity)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match &args[0] {
      Value::String(s) => Some(Value::EntityString(s.to_string())),
      _ => None,
    }
  }
}
