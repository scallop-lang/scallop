use super::*;
use crate::common::type_family::*;
use crate::common::value_type::*;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub enum TypeSet {
  BaseType(ValueType, AstNodeLocation), // Concrete base type
  Numeric(AstNodeLocation),             // Contains integer and float, default i32
  Arith(AstNodeLocation),               // Numeric but with arithmetics, default integer i32
  Integer(AstNodeLocation),             // integer, default `i32`
  SignedInteger(AstNodeLocation),       // signed integer, default `i32`
  UnsignedInteger(AstNodeLocation),     // unsigned integer, default `u32`
  Float(AstNodeLocation),               // float, default `f32`
  String(AstNodeLocation),              // string, default `String`
  Any(AstNodeLocation),                 // Any type, default i32
}

impl Default for TypeSet {
  fn default() -> Self {
    Self::Any(AstNodeLocation::default())
  }
}

impl From<TypeFamily> for TypeSet {
  fn from(value: TypeFamily) -> Self {
    match value {
      TypeFamily::Bottom => panic!("There is no type set for bottom type; aborting"),
      TypeFamily::String => Self::String(Loc::default()),
      TypeFamily::Float => Self::Float(Loc::default()),
      TypeFamily::SignedInteger => Self::SignedInteger(Loc::default()),
      TypeFamily::UnsignedInteger => Self::UnsignedInteger(Loc::default()),
      TypeFamily::Integer => Self::Integer(Loc::default()),
      TypeFamily::Number => Self::Numeric(Loc::default()),
      TypeFamily::Any => Self::Any(Loc::default()),
    }
  }
}

impl std::cmp::PartialEq for TypeSet {
  fn eq(&self, other: &Self) -> bool {
    match (self, other) {
      (Self::BaseType(b1, _), Self::BaseType(b2, _)) => b1 == b2,
      (Self::Numeric(_), Self::Numeric(_)) => true,
      (Self::Arith(_), Self::Arith(_)) => true,
      (Self::Integer(_), Self::Integer(_)) => true,
      (Self::SignedInteger(_), Self::SignedInteger(_)) => true,
      (Self::Float(_), Self::Float(_)) => true,
      (Self::String(_), Self::String(_)) => true,
      (Self::Any(_), Self::Any(_)) => true,
      _ => false,
    }
  }
}

impl std::cmp::PartialOrd for TypeSet {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering::*;
    match (self, other) {
      // Base type less than anything else
      (Self::BaseType(b1, _), Self::BaseType(b2, _)) if b1 == b2 => Some(Equal),
      (Self::BaseType(b, _), Self::String(_)) if b.is_string() => Some(Less),
      (Self::String(_), Self::BaseType(b, _)) if b.is_string() => Some(Greater),
      (Self::BaseType(b, _), Self::Float(_)) if b.is_float() => Some(Less),
      (Self::Float(_), Self::BaseType(b, _)) if b.is_float() => Some(Greater),
      (Self::BaseType(b, _), Self::SignedInteger(_)) if b.is_signed_integer() => Some(Less),
      (Self::SignedInteger(_), Self::BaseType(b, _)) if b.is_signed_integer() => Some(Greater),
      (Self::BaseType(b, _), Self::Integer(_)) if b.is_integer() => Some(Less),
      (Self::Integer(_), Self::BaseType(b, _)) if b.is_integer() => Some(Greater),
      (Self::BaseType(b, _), Self::Arith(_)) if b.is_numeric() => Some(Less),
      (Self::Arith(_), Self::BaseType(b, _)) if b.is_numeric() => Some(Greater),
      (Self::BaseType(b, _), Self::Numeric(_)) if b.is_numeric() => Some(Less),
      (Self::Numeric(_), Self::BaseType(b, _)) if b.is_numeric() => Some(Greater),
      (Self::BaseType(_, _), Self::Any(_)) => Some(Less),
      (Self::Any(_), Self::BaseType(_, _)) => Some(Greater),

      // String type less than Any
      (Self::String(_), Self::String(_)) => Some(Equal),
      (Self::String(_), Self::Any(_)) => Some(Less),
      (Self::Any(_), Self::String(_)) => Some(Greater),

      // Float type less than Arith and Numeric and Any
      (Self::Float(_), Self::Float(_)) => Some(Equal),
      (Self::Float(_), Self::Arith(_)) => Some(Less),
      (Self::Arith(_), Self::Float(_)) => Some(Greater),
      (Self::Float(_), Self::Numeric(_)) => Some(Less),
      (Self::Numeric(_), Self::Float(_)) => Some(Greater),
      (Self::Float(_), Self::Any(_)) => Some(Less),
      (Self::Any(_), Self::Float(_)) => Some(Greater),

      // Signed Integer less than Integer, Arith, Numeric, and Any
      (Self::SignedInteger(_), Self::SignedInteger(_)) => Some(Equal),
      (Self::SignedInteger(_), Self::Integer(_)) => Some(Less),
      (Self::Integer(_), Self::SignedInteger(_)) => Some(Greater),
      (Self::SignedInteger(_), Self::Arith(_)) => Some(Less),
      (Self::Arith(_), Self::SignedInteger(_)) => Some(Greater),
      (Self::SignedInteger(_), Self::Numeric(_)) => Some(Less),
      (Self::Numeric(_), Self::SignedInteger(_)) => Some(Greater),
      (Self::SignedInteger(_), Self::Any(_)) => Some(Less),
      (Self::Any(_), Self::SignedInteger(_)) => Some(Greater),

      // Unsigned Integer less than Integer, Arith, Numeric, and Any
      (Self::UnsignedInteger(_), Self::UnsignedInteger(_)) => Some(Equal),
      (Self::UnsignedInteger(_), Self::Integer(_)) => Some(Less),
      (Self::Integer(_), Self::UnsignedInteger(_)) => Some(Greater),
      (Self::UnsignedInteger(_), Self::Arith(_)) => Some(Less),
      (Self::Arith(_), Self::UnsignedInteger(_)) => Some(Greater),
      (Self::UnsignedInteger(_), Self::Numeric(_)) => Some(Less),
      (Self::Numeric(_), Self::UnsignedInteger(_)) => Some(Greater),
      (Self::UnsignedInteger(_), Self::Any(_)) => Some(Less),
      (Self::Any(_), Self::UnsignedInteger(_)) => Some(Greater),

      // Integer less than Arith, Numeric, and Any
      (Self::Integer(_), Self::Integer(_)) => Some(Equal),
      (Self::Integer(_), Self::Arith(_)) => Some(Less),
      (Self::Arith(_), Self::Integer(_)) => Some(Greater),
      (Self::Integer(_), Self::Numeric(_)) => Some(Less),
      (Self::Numeric(_), Self::Integer(_)) => Some(Greater),
      (Self::Integer(_), Self::Any(_)) => Some(Less),
      (Self::Any(_), Self::Integer(_)) => Some(Greater),

      // Arith less than Numeric and Any
      (Self::Arith(_), Self::Arith(_)) => Some(Equal),
      (Self::Arith(_), Self::Numeric(_)) => Some(Less),
      (Self::Numeric(_), Self::Arith(_)) => Some(Greater),
      (Self::Arith(_), Self::Any(_)) => Some(Less),
      (Self::Any(_), Self::Arith(_)) => Some(Greater),

      // Numeric less than Any
      (Self::Numeric(_), Self::Numeric(_)) => Some(Equal),
      (Self::Numeric(_), Self::Any(_)) => Some(Less),
      (Self::Any(_), Self::Numeric(_)) => Some(Greater),

      // Any
      (Self::Any(_), Self::Any(_)) => Some(Greater),

      // All others have no ordering
      _ => None,
    }
  }
}

impl std::fmt::Display for TypeSet {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::BaseType(t, _) => t.fmt(f),
      Self::Numeric(_) => f.write_str("numeric"),
      Self::Arith(_) => f.write_str("arithmetic"),
      Self::Integer(_) => f.write_str("integer"),
      Self::SignedInteger(_) => f.write_str("signed integer"),
      Self::UnsignedInteger(_) => f.write_str("unsigned integer"),
      Self::Float(_) => f.write_str("float"),
      Self::String(_) => f.write_str("string"),
      Self::Any(_) => f.write_str("any"),
    }
  }
}

impl TypeSet {
  pub fn base(ty: ValueType) -> Self {
    Self::BaseType(ty, AstNodeLocation::default())
  }

  pub fn any() -> Self {
    Self::Any(AstNodeLocation::default())
  }

  pub fn numeric() -> Self {
    Self::Numeric(AstNodeLocation::default())
  }

  pub fn arith() -> Self {
    Self::Arith(AstNodeLocation::default())
  }

  pub fn string() -> Self {
    Self::String(AstNodeLocation::default())
  }

  pub fn from_constant(c: &Constant) -> Self {
    match &c.node {
      ConstantNode::Integer(i) => {
        if i < &0 {
          Self::SignedInteger(c.location().clone())
        } else {
          Self::Numeric(c.location().clone())
        }
      }
      ConstantNode::Float(_) => Self::Float(c.location().clone()),
      ConstantNode::Char(_) => Self::BaseType(ValueType::Char, c.location().clone()),
      ConstantNode::Boolean(_) => Self::BaseType(ValueType::Bool, c.location().clone()),
      ConstantNode::String(_) => Self::String(c.location().clone()),
      ConstantNode::DateTime(_) => Self::BaseType(ValueType::DateTime, c.location().clone()),
      ConstantNode::Duration(_) => Self::BaseType(ValueType::Duration, c.location().clone()),
      ConstantNode::Invalid(_) => panic!("[Internal Error] Should not be called with invalid constant"),
    }
  }

  pub fn can_type_cast(&self, base_ty: &ValueType) -> bool {
    // Shortcut: any type can be casted to String
    if &ValueType::String == base_ty {
      return true;
    }

    // Check inside type lattice
    match (self, base_ty) {
      (Self::BaseType(b1, _), base_ty) => b1.can_type_cast(base_ty),
      (Self::Numeric(_), base_ty) => base_ty.is_numeric(),
      (Self::Arith(_), base_ty) => base_ty.is_numeric(),
      (Self::Integer(_), base_ty) => base_ty.is_numeric(),
      (Self::SignedInteger(_), base_ty) => base_ty.is_numeric(),
      (Self::UnsignedInteger(_), base_ty) => base_ty.is_numeric(),
      (Self::Float(_), base_ty) => base_ty.is_numeric(),
      (Self::String(_), base_ty) => base_ty.is_string() || base_ty.is_numeric(),
      (Self::Any(_), base_ty) => base_ty.is_numeric(),
    }
  }

  pub fn is_boolean(&self) -> bool {
    match self {
      Self::BaseType(b, _) => b.is_boolean(),
      _ => false,
    }
  }

  pub fn location(&self) -> &AstNodeLocation {
    match self {
      Self::BaseType(_, l) => l,
      Self::Numeric(l) => l,
      Self::Arith(l) => l,
      Self::Integer(l) => l,
      Self::SignedInteger(l) => l,
      Self::UnsignedInteger(l) => l,
      Self::Float(l) => l,
      Self::String(l) => l,
      Self::Any(l) => l,
    }
  }

  pub fn to_default_value_type(&self) -> ValueType {
    match self {
      Self::BaseType(b, _) => b.clone(),
      Self::Numeric(_) => ValueType::I32,
      Self::Arith(_) => ValueType::I32,
      Self::Integer(_) => ValueType::I32,
      Self::SignedInteger(_) => ValueType::I32,
      Self::UnsignedInteger(_) => ValueType::U32,
      Self::Float(_) => ValueType::F32,
      Self::String(_) => ValueType::String,
      Self::Any(_) => ValueType::I32,
    }
  }

  pub fn has_same_kind(&self, other: &Self) -> bool {
    match (self, other) {
      (Self::BaseType(_, _), Self::BaseType(_, _)) => true,
      (Self::Numeric(_), Self::Numeric(_)) => true,
      (Self::Arith(_), Self::Arith(_)) => true,
      (Self::Integer(_), Self::Integer(_)) => true,
      (Self::SignedInteger(_), Self::SignedInteger(_)) => true,
      (Self::UnsignedInteger(_), Self::UnsignedInteger(_)) => true,
      (Self::Float(_), Self::Float(_)) => true,
      (Self::String(_), Self::String(_)) => true,
      (Self::Any(_), Self::Any(_)) => true,
      _ => false,
    }
  }

  pub fn unify_type_sets(tss: Vec<&TypeSet>) -> Result<TypeSet, TypeInferenceError> {
    let mut ty = tss[0].clone();
    for curr_ty in tss {
      match ty.unify(curr_ty) {
        Ok(new_ty) => ty = new_ty,
        Err(err) => return Err(err),
      }
    }
    Ok(ty)
  }

  pub fn unify(&self, other: &Self) -> Result<Self, TypeInferenceError> {
    use std::cmp::Ordering::*;
    match self.partial_cmp(other) {
      Some(Equal) | Some(Less) => Ok(self.clone()),
      Some(Greater) => Ok(other.clone()),
      None => Err(TypeInferenceError::CannotUnifyTypes {
        t1: self.clone(),
        t2: other.clone(),
        loc: None,
      }),
    }
  }

  pub fn contains_value_type(&self, value_type: &ValueType) -> bool {
    match self {
      Self::BaseType(b, _) => b == value_type,
      Self::Numeric(_) => value_type.is_numeric(),
      Self::Arith(_) => value_type.is_numeric(),
      Self::Integer(_) => value_type.is_integer(),
      Self::SignedInteger(_) => value_type.is_signed_integer(),
      Self::UnsignedInteger(_) => value_type.is_unsigned_integer(),
      Self::Float(_) => value_type.is_float(),
      Self::String(_) => value_type.is_string(),
      Self::Any(_) => true,
    }
  }
}
