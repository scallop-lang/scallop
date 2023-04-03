use super::*;

/// Hash
///
/// ``` scl
/// extern fn $hash(x: Any...) -> u64
/// ```
#[derive(Clone)]
pub struct Hash;

impl ForeignFunction for Hash {
  fn name(&self) -> String {
    "hash".to_string()
  }

  fn has_variable_arguments(&self) -> bool {
    true
  }

  fn variable_argument_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::TypeFamily(TypeFamily::Any)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::U64)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut s = DefaultHasher::new();
    args.hash(&mut s);
    Some(s.finish().into())
  }
}
