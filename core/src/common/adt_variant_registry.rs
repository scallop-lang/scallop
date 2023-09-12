use std::collections::*;

use super::entity::*;
use super::value::*;
use super::value_type::*;

use crate::compiler::front::*;

#[derive(Debug, Clone)]
pub struct ADTVariant {
  pub relation_name: String,
  pub arg_types: Vec<ValueType>,
}

#[derive(Debug, Clone)]
pub struct ADTVariantRegistry {
  registry: HashMap<String, ADTVariant>,
}

impl ADTVariantRegistry {
  pub fn new() -> Self {
    Self {
      registry: HashMap::new(),
    }
  }

  pub fn add(&mut self, variant_name: String, relation_name: String, arg_types: Vec<ValueType>) {
    let variant = ADTVariant {
      relation_name,
      arg_types,
    };
    self.registry.insert(variant_name, variant);
  }

  pub fn iter(&self) -> std::collections::hash_map::Iter<String, ADTVariant> {
    self.registry.iter()
  }

  pub fn parse(&self, s: &str) -> Result<ADTParseResult, ADTEntityError> {
    // First parse an entity from string
    let entity = parser::str_to_entity(s).map_err(ADTEntityError::Parsing)?;

    // Completely parse the entity:
    // - check that there is no variable or expression involved
    // - check that all the mentioned ADT variants exist
    // - check all the constant type
    // - create intermediate ADTs
    // - generate a final ADT as the entity value
    let mut facts = Vec::new();
    let entity = self.parse_entity(&entity, &ValueType::Entity, &mut facts)?;

    // Return the final result
    Ok(ADTParseResult { entity, facts })
  }

  fn parse_entity(
    &self,
    entity: &ast::Entity,
    ty: &ValueType,
    facts: &mut Vec<(String, Value, Vec<Value>)>,
  ) -> Result<Value, ADTEntityError> {
    match &entity {
      ast::Entity::Expr(e) => match e {
        ast::Expr::Constant(c) => match (&c, ty) {
          (ast::Constant::Integer(i), ValueType::I8) => Ok(Value::I8(i.int().clone() as i8)),
          (ast::Constant::Integer(i), ValueType::I16) => Ok(Value::I16(i.int().clone() as i16)),
          (ast::Constant::Integer(i), ValueType::I32) => Ok(Value::I32(i.int().clone() as i32)),
          (ast::Constant::Integer(i), ValueType::I64) => Ok(Value::I64(i.int().clone() as i64)),
          (ast::Constant::Integer(i), ValueType::I128) => Ok(Value::I128(i.int().clone() as i128)),
          (ast::Constant::Integer(i), ValueType::ISize) => Ok(Value::ISize(i.int().clone() as isize)),
          (ast::Constant::Integer(i), ValueType::U8) => Ok(Value::U8(i.int().clone() as u8)),
          (ast::Constant::Integer(i), ValueType::U16) => Ok(Value::U16(i.int().clone() as u16)),
          (ast::Constant::Integer(i), ValueType::U32) => Ok(Value::U32(i.int().clone() as u32)),
          (ast::Constant::Integer(i), ValueType::U64) => Ok(Value::U64(i.int().clone() as u64)),
          (ast::Constant::Integer(i), ValueType::U128) => Ok(Value::U128(i.int().clone() as u128)),
          (ast::Constant::Integer(i), ValueType::USize) => Ok(Value::USize(i.int().clone() as usize)),
          (ast::Constant::Integer(i), ValueType::F32) => Ok(Value::F32(i.int().clone() as f32)),
          (ast::Constant::Integer(i), ValueType::F64) => Ok(Value::F64(i.int().clone() as f64)),
          (ast::Constant::Float(f), ValueType::F32) => Ok(Value::F32(f.float().clone() as f32)),
          (ast::Constant::Float(f), ValueType::F64) => Ok(Value::F64(f.float().clone() as f64)),
          (ast::Constant::Boolean(b), ValueType::Bool) => Ok(Value::Bool(b.value().clone())),
          (ast::Constant::Char(c), ValueType::Char) => Ok(Value::Char(c.parse_char())),
          (ast::Constant::String(s), ValueType::String) => Ok(Value::String(s.string().clone())),
          _ => Err(ADTEntityError::CannotUnifyType),
        },
        _ => Err(ADTEntityError::InvalidExpr),
      },
      ast::Entity::Object(o) => {
        let variant_name = o.functor().name();
        if let Some(variant) = self.registry.get(variant_name) {
          let expected_arity = variant.arg_types.len() - 1;
          let actual_arity = o.args().len();
          if expected_arity == actual_arity {
            // Compute the arguments
            let parsed_args = o
              .args()
              .iter()
              .zip(variant.arg_types.iter().skip(1))
              .map(|(arg, arg_ty)| self.parse_entity(arg, arg_ty, facts))
              .collect::<Result<Vec<_>, _>>()?;

            // Aggregate them into a hash
            let entity = Value::Entity(encode_entity(variant_name, parsed_args.iter()));

            // Create a new fact to insert
            let fact = (variant.relation_name.clone(), entity.clone(), parsed_args);
            facts.push(fact);

            // Return the entity as result
            Ok(entity)
          } else {
            Err(ADTEntityError::ArityMismatch {
              variant: variant_name.to_string(),
              expected: expected_arity,
              actual: actual_arity,
            })
          }
        } else {
          Err(ADTEntityError::UnknownVariant(variant_name.to_string()))
        }
      }
    }
  }
}

#[derive(Clone, Debug)]
pub struct ADTParseResult {
  pub entity: Value,
  pub facts: Vec<(String, Value, Vec<Value>)>,
}

#[derive(Clone, Debug)]
pub enum ADTEntityError {
  Parsing(parser::ParserError),
  InvalidExpr,
  CannotUnifyType,
  UnknownVariant(String),
  ArityMismatch {
    variant: String,
    expected: usize,
    actual: usize,
  },
}

impl std::fmt::Display for ADTEntityError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Parsing(p) => p.fmt(f),
      Self::InvalidExpr => f.write_str("Invalid Expression"),
      Self::CannotUnifyType => f.write_str("Cannot unify type"),
      Self::UnknownVariant(v) => f.write_fmt(format_args!("Unknown variant `{}`", v)),
      Self::ArityMismatch {
        variant,
        expected,
        actual,
      } => f.write_fmt(format_args!(
        "Arity mismatch for variant `{}`, expected {}, found {}",
        variant, expected, actual
      )),
    }
  }
}
