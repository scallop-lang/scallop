use super::*;

/// Min
///
/// ``` scl
/// extern fn $min<T: Number>(x: T...) -> T
/// ```
#[derive(Clone)]
pub struct Min;

impl Min {
  fn dyn_min<T: PartialOrd>(args: Vec<Value>) -> Option<T>
  where
    Value: TryInto<T>,
  {
    let mut iter = args.into_iter();
    let mut curr_min: T = iter.next()?.try_into().ok()?;
    while let Some(next_elem) = iter.next() {
      let next_elem = next_elem.try_into().ok()?;
      if next_elem < curr_min {
        curr_min = next_elem;
      }
    }
    Some(curr_min)
  }
}

impl ForeignFunction for Min {
  fn name(&self) -> String {
    "min".to_string()
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
      ValueType::I8 => Self::dyn_min(args).map(Value::I8),
      ValueType::I16 => Self::dyn_min(args).map(Value::I16),
      ValueType::I32 => Self::dyn_min(args).map(Value::I32),
      ValueType::I64 => Self::dyn_min(args).map(Value::I64),
      ValueType::I128 => Self::dyn_min(args).map(Value::I128),
      ValueType::ISize => Self::dyn_min(args).map(Value::ISize),
      ValueType::U8 => Self::dyn_min(args).map(Value::U8),
      ValueType::U16 => Self::dyn_min(args).map(Value::U16),
      ValueType::U32 => Self::dyn_min(args).map(Value::U32),
      ValueType::U64 => Self::dyn_min(args).map(Value::U64),
      ValueType::U128 => Self::dyn_min(args).map(Value::U128),
      ValueType::USize => Self::dyn_min(args).map(Value::USize),
      ValueType::F32 => Self::dyn_min(args).map(Value::F32),
      ValueType::F64 => Self::dyn_min(args).map(Value::F64),
      _ => None,
    }
  }
}
