use std::str::FromStr;

use super::value::*;

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub enum Function {
  Abs,
  Hash,
  StringConcat,
  StringLength,
  Substring,
  StringCharAt,
}

impl std::fmt::Display for Function {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Abs => f.write_str("abs"),
      Self::Hash => f.write_str("hash"),
      Self::StringConcat => f.write_str("string_concat"),
      Self::StringLength => f.write_str("string_length"),
      Self::Substring => f.write_str("substring"),
      Self::StringCharAt => f.write_str("string_char_at"),
    }
  }
}

impl FromStr for Function {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "abs" => Ok(Self::Abs),
      "hash" => Ok(Self::Hash),
      "string_concat" => Ok(Self::StringConcat),
      "string_length" => Ok(Self::StringLength),
      "substring" => Ok(Self::Substring),
      "string_char_at" => Ok(Self::StringCharAt),
      _ => Err(s.to_string()),
    }
  }
}

impl Function {
  pub fn is_acceptable_num_args(&self, i: usize) -> bool {
    match self {
      Self::Abs => i == 1,
      Self::Hash => i > 0,
      Self::StringConcat => i > 0,
      Self::StringLength => i == 1,
      Self::Substring => i == 2 || i == 3,
      Self::StringCharAt => i == 2,
    }
  }

  pub fn call(&self, args: Vec<Value>) -> Option<Value> {
    match self {
      Self::Abs => {
        let result = match &args[0] {
          Value::I8(i) => Value::I8(i.abs()),
          Value::I16(i) => Value::I16(i.abs()),
          Value::I32(i) => Value::I32(i.abs()),
          Value::I64(i) => Value::I64(i.abs()),
          Value::I128(i) => Value::I128(i.abs()),
          Value::ISize(i) => Value::ISize(i.abs()),
          Value::U8(i) => Value::U8(*i),
          Value::U16(i) => Value::U16(*i),
          Value::U32(i) => Value::U32(*i),
          Value::U64(i) => Value::U64(*i),
          Value::U128(i) => Value::U128(*i),
          Value::USize(i) => Value::USize(*i),
          Value::F32(f) => Value::F32(f.abs()),
          Value::F64(f) => Value::F64(f.abs()),
          v => panic!("Cannot call `abs` on `{}`", v),
        };
        Some(result)
      }
      Self::Hash => {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut s = DefaultHasher::new();
        args.hash(&mut s);
        Some(s.finish().into())
      }
      Self::StringConcat => {
        let result = args
          .into_iter()
          .map(|a| a.as_str().to_string())
          .collect::<Vec<_>>()
          .join("")
          .into();
        Some(result)
      }
      Self::StringLength => match &args[0] {
        Value::Str(s) => Some(s.len().into()),
        Value::String(s) => Some(s.len().into()),
        _ => panic!("Cannot perform string length on value `{}`", args[0]),
      },
      Self::Substring => {
        if args.len() == 2 {
          let start = args[1].as_usize();
          match &args[0] {
            Value::Str(s) => Some(s[start..].to_string().into()),
            Value::String(s) => Some(s[start..].to_string().into()),
            _ => panic!("Cannot perform substring on value `{}`", args[0]),
          }
        } else if args.len() == 3 {
          let start = args[1].as_usize();
          let end = args[2].as_usize();
          match &args[0] {
            Value::Str(s) => Some(s[start..end].to_string().into()),
            Value::String(s) => Some(s[start..end].to_string().into()),
            _ => panic!("Cannot perform substring on value `{}`", args[0]),
          }
        } else {
          panic!("Incorrect number of arguments when calling substring function")
        }
      }
      Self::StringCharAt => {
        let string = args[0].as_str();
        let index = args[1].as_usize();
        if index < string.len() {
          Some(Value::Char(string.chars().skip(index).next().unwrap()))
        } else {
          None // Index out of range
        }
      }
    }
  }
}
