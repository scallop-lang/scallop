use super::*;

use crate::runtime::env::*;

/// Format foreign function
///
/// ``` scl
/// extern fn $format(f: String, args: Any...) -> String
/// ```
#[derive(Clone)]
pub struct Format;

impl ForeignFunction for Format {
  fn name(&self) -> String {
    "format".to_string()
  }

  fn num_generic_types(&self) -> usize {
    0
  }

  fn num_static_arguments(&self) -> usize {
    1
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert_eq!(i, 0);
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn has_variable_arguments(&self) -> bool {
    true
  }

  fn variable_argument_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::TypeFamily(TypeFamily::Any)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::BaseType(ValueType::String)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match &args[0] {
      Value::String(f) => {
        let split_str: Vec<&str> = f.split("{}").collect();
        let mut result = "".to_string();
        result += split_str[0];
        // boths lens should be # of braces + 1
        assert_eq!(args.len(), split_str.len());
        for i in 1..args.len() {
          let s = match &args[i] {
            Value::I8(i) => i.to_string(),
            Value::I16(i) => i.to_string(),
            Value::I32(i) => i.to_string(),
            Value::I64(i) => i.to_string(),
            Value::I128(i) => i.to_string(),
            Value::ISize(i) => i.to_string(),
            Value::U8(i) => i.to_string(),
            Value::U16(i) => i.to_string(),
            Value::U32(i) => i.to_string(),
            Value::U64(i) => i.to_string(),
            Value::U128(i) => i.to_string(),
            Value::USize(i) => i.to_string(),
            Value::F32(i) => i.to_string(),
            Value::F64(i) => i.to_string(),
            Value::Char(i) => i.to_string(),
            Value::Bool(i) => i.to_string(),
            Value::Str(i) => i.to_string(),
            Value::String(i) => i.to_string(),
            Value::Symbol(_) => panic!("[Internal Error] Symbol should not be processed"),
            Value::SymbolString(s) => s.to_string(),
            Value::DateTime(i) => i.to_string(),
            Value::Duration(i) => i.to_string(),
            Value::Entity(e) => format!("entity({e:#x})"),
            Value::Tensor(_) | Value::TensorValue(_) => "tensor".to_string(),
          };
          result += s.as_str();
          result += split_str[i];
        }
        Some(Value::from(result))
      }
      _ => panic!("Format argument not a string"),
    }
  }

  fn execute_with_env(&self, env: &RuntimeEnvironment, args: Vec<Value>) -> Option<Value> {
    match &args[0] {
      Value::String(f) => {
        let split_str: Vec<&str> = f.split("{}").collect();
        let mut result = "".to_string();
        result += split_str[0];
        // boths lens should be # of braces + 1
        assert_eq!(args.len(), split_str.len());
        for i in 1..args.len() {
          let s = match &args[i] {
            Value::I8(i) => i.to_string(),
            Value::I16(i) => i.to_string(),
            Value::I32(i) => i.to_string(),
            Value::I64(i) => i.to_string(),
            Value::I128(i) => i.to_string(),
            Value::ISize(i) => i.to_string(),
            Value::U8(i) => i.to_string(),
            Value::U16(i) => i.to_string(),
            Value::U32(i) => i.to_string(),
            Value::U64(i) => i.to_string(),
            Value::U128(i) => i.to_string(),
            Value::USize(i) => i.to_string(),
            Value::F32(i) => i.to_string(),
            Value::F64(i) => i.to_string(),
            Value::Char(i) => i.to_string(),
            Value::Bool(i) => i.to_string(),
            Value::Str(i) => i.to_string(),
            Value::String(i) => i.to_string(),
            Value::Symbol(i) => env
              .symbol_registry
              .get_symbol(*i)
              .expect("[Internal Error] Cannot find symbol"),
            Value::SymbolString(_) => panic!("[Internal Error] SymbolString should not be processed"),
            Value::DateTime(i) => i.to_string(),
            Value::Duration(i) => i.to_string(),
            Value::Entity(e) => format!("entity({e:#x})"),
            Value::Tensor(_) | Value::TensorValue(_) => "tensor".to_string(),
          };
          result += s.as_str();
          result += split_str[i];
        }
        Some(Value::from(result))
      }
      _ => panic!("Format argument not a string"),
    }
  }
}
