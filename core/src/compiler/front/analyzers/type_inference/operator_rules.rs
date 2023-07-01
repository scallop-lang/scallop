use lazy_static::lazy_static;

use crate::common::value_type::*;

lazy_static! {
  pub static ref ADD_TYPING_RULES: Vec<(ValueType, ValueType, ValueType)> = {
    use ValueType::*;
    vec![
      (I8, I8, I8),
      (I16, I16, I16),
      (I32, I32, I32),
      (I64, I64, I64),
      (I128, I128, I128),
      (ISize, ISize, ISize),
      (U8, U8, U8),
      (U16, U16, U16),
      (U32, U32, U32),
      (U64, U64, U64),
      (U128, U128, U128),
      (USize, USize, USize),
      (F32, F32, F32),
      (F64, F64, F64),
      (String, String, String),
      (DateTime, Duration, DateTime),
      (Duration, DateTime, DateTime),
      (Duration, Duration, Duration),
      (Tensor, Tensor, Tensor),
      (Tensor, F64, Tensor),
      (F64, Tensor, Tensor),
    ]
  };
  pub static ref SUB_TYPING_RULES: Vec<(ValueType, ValueType, ValueType)> = {
    use ValueType::*;
    vec![
      (I8, I8, I8),
      (I16, I16, I16),
      (I32, I32, I32),
      (I64, I64, I64),
      (I128, I128, I128),
      (ISize, ISize, ISize),
      (U8, U8, U8),
      (U16, U16, U16),
      (U32, U32, U32),
      (U64, U64, U64),
      (U128, U128, U128),
      (USize, USize, USize),
      (F32, F32, F32),
      (F64, F64, F64),
      (DateTime, Duration, DateTime),
      (DateTime, DateTime, Duration),
      (Duration, Duration, Duration),
      (Tensor, Tensor, Tensor),
      (Tensor, F64, Tensor),
      (F64, Tensor, Tensor),
    ]
  };
  pub static ref MULT_TYPING_RULES: Vec<(ValueType, ValueType, ValueType)> = {
    use ValueType::*;
    vec![
      (I8, I8, I8),
      (I16, I16, I16),
      (I32, I32, I32),
      (I64, I64, I64),
      (I128, I128, I128),
      (ISize, ISize, ISize),
      (U8, U8, U8),
      (U16, U16, U16),
      (U32, U32, U32),
      (U64, U64, U64),
      (U128, U128, U128),
      (USize, USize, USize),
      (F32, F32, F32),
      (F64, F64, F64),
      (Duration, I32, Duration),
      (I32, Duration, Duration),
      (Tensor, Tensor, Tensor),
      (Tensor, F64, Tensor),
      (F64, Tensor, Tensor),
    ]
  };
  pub static ref DIV_TYPING_RULES: Vec<(ValueType, ValueType, ValueType)> = {
    use ValueType::*;
    vec![
      (I8, I8, I8),
      (I16, I16, I16),
      (I32, I32, I32),
      (I64, I64, I64),
      (I128, I128, I128),
      (ISize, ISize, ISize),
      (U8, U8, U8),
      (U16, U16, U16),
      (U32, U32, U32),
      (U64, U64, U64),
      (U128, U128, U128),
      (USize, USize, USize),
      (F32, F32, F32),
      (F64, F64, F64),
      (Duration, I32, Duration),
    ]
  };
  pub static ref MOD_TYPING_RULES: Vec<(ValueType, ValueType, ValueType)> = {
    use ValueType::*;
    vec![
      (I8, I8, I8),
      (I16, I16, I16),
      (I32, I32, I32),
      (I64, I64, I64),
      (I128, I128, I128),
      (ISize, ISize, ISize),
      (U8, U8, U8),
      (U16, U16, U16),
      (U32, U32, U32),
      (U64, U64, U64),
      (U128, U128, U128),
      (USize, USize, USize),
      (F32, F32, F32),
      (F64, F64, F64),
    ]
  };
  pub static ref COMPARE_TYPING_RULES: Vec<(ValueType, ValueType)> = {
    use ValueType::*;
    vec![
      (I8, I8),
      (I16, I16),
      (I32, I32),
      (I64, I64),
      (I128, I128),
      (ISize, ISize),
      (U8, U8),
      (U16, U16),
      (U32, U32),
      (U64, U64),
      (U128, U128),
      (USize, USize),
      (F32, F32),
      (F64, F64),
      (Duration, Duration),
      (DateTime, DateTime),
    ]
  };
}
