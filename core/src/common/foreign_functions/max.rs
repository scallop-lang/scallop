use super::*;

/// Max
///
/// ``` scl
/// extern fn $max<T: Number>(x: T...) -> T
/// ```
#[derive(Clone)]
pub struct Max;

impl Max {
  fn dyn_max<T: PartialOrd>(args: Vec<Value>) -> Option<T> where Value: TryInto<T> {
    let mut iter = args.into_iter();
    let mut curr_max: T = iter.next()?.try_into().ok()?;
    while let Some(next_elem) = iter.next() {
      let next_elem = next_elem.try_into().ok()?;
      if next_elem > curr_max {
        curr_max = next_elem;
      }
    }
    Some(curr_max)
  }
}

impl ForeignFunction for Max {
  fn name(&self) -> String {
    "max".to_string()
  }

  fn num_generic_types(&self) -> usize {
    1
  }

  fn generic_type_family(&self, i: usize) -> TypeFamily {
    assert_eq!(i, 0);
    TypeFamily::Number
  }

  fn has_variable_arguments(&self) -> bool {
    true
  }

  fn variable_argument_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::Generic(0)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::Generic(0)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    let rt = self.infer_return_type(&args);
    match rt {
      ValueType::I8 => Self::dyn_max(args).map(Value::I8),
      ValueType::I16 => Self::dyn_max(args).map(Value::I16),
      ValueType::I32 => Self::dyn_max(args).map(Value::I32),
      ValueType::I64 => Self::dyn_max(args).map(Value::I64),
      ValueType::I128 => Self::dyn_max(args).map(Value::I128),
      ValueType::ISize => Self::dyn_max(args).map(Value::ISize),
      ValueType::U8 => Self::dyn_max(args).map(Value::U8),
      ValueType::U16 => Self::dyn_max(args).map(Value::U16),
      ValueType::U32 => Self::dyn_max(args).map(Value::U32),
      ValueType::U64 => Self::dyn_max(args).map(Value::U64),
      ValueType::U128 => Self::dyn_max(args).map(Value::U128),
      ValueType::USize => Self::dyn_max(args).map(Value::USize),
      ValueType::F32 => Self::dyn_max(args).map(Value::F32),
      ValueType::F64 => Self::dyn_max(args).map(Value::F64),
      _ => None,
    }
  }
}
