use lazy_static::lazy_static;
use std::collections::*;

use crate::common::expr;
use crate::common::foreign_predicate::*;
use crate::common::input_tag::*;
use crate::common::tuple::*;
use crate::common::unary_op;
use crate::common::value::*;
use crate::common::value_type::*;

use crate::compiler::front::*;
use crate::runtime::env::*;

lazy_static! {
  pub static ref TAG_TYPE: Vec<ValueType> = {
    use ValueType::*;
    vec![F64, F32, Bool]
  };
}

#[derive(Clone, Debug)]
pub struct TaggedRuleAnalysis {
  pub to_add_tag_predicates: HashMap<ast::NodeLocation, ToAddTagPredicate>,
  pub errors: Vec<FrontCompileErrorMessage>,
}

impl TaggedRuleAnalysis {
  pub fn new() -> Self {
    Self {
      to_add_tag_predicates: HashMap::new(),
      errors: Vec::new(),
    }
  }

  pub fn add_tag_predicate(
    &mut self,
    rule_id: ast::NodeLocation,
    name: String,
    arg_name: String,
    tag_loc: ast::NodeLocation,
  ) {
    let pred = ToAddTagPredicate::new(name, arg_name, tag_loc);
    self.to_add_tag_predicates.insert(rule_id, pred);
  }

  pub fn register_predicates(
    &mut self,
    type_inference: &super::TypeInference,
    foreign_predicate_registry: &mut ForeignPredicateRegistry,
  ) {
    for (rule_id, tag_predicate) in self.to_add_tag_predicates.drain() {
      if let Some(rule_variable_type) = type_inference.rule_variable_type.get(&rule_id) {
        if let Some(var_ty) = rule_variable_type.get(&tag_predicate.arg_name) {
          match get_target_tag_type(var_ty, &tag_predicate.tag_loc) {
            Ok(target_tag_ty) => {
              // This means that we have an okay tag that is type checked
              // Create a foreign predicate and register it
              let fp = TagPredicate::new(tag_predicate.name.clone(), target_tag_ty);
              if let Err(err) = foreign_predicate_registry.register(fp) {
                self.errors.push(FrontCompileErrorMessage::error().msg(err.to_string()));
              }
            }
            Err(err) => {
              self.errors.push(err);
            }
          }
        }
      }
    }
  }
}

fn get_target_tag_type(
  var_ty: &analyzers::type_inference::TypeSet,
  loc: &ast::NodeLocation,
) -> Result<ValueType, FrontCompileErrorMessage> {
  // Top priority: if var_ty is a base type, directly check if it is among some expected type
  if let Some(base_ty) = var_ty.get_base_type() {
    if TAG_TYPE.contains(&base_ty) {
      return Ok(base_ty);
    }
  }

  // Then we check if the value can be casted into certain types
  for tag_ty in TAG_TYPE.iter() {
    if var_ty.can_type_cast(tag_ty) {
      return Ok(var_ty.to_default_value_type());
    }
  }

  // If not, then
  return Err(
    FrontCompileErrorMessage::error()
      .msg(format!(
        "A value of type `{var_ty}` cannot be casted into a dynamic tag"
      ))
      .src(loc.clone()),
  );
}

/// The information of a helper tag predicate
///
/// Suppose we have a rule
/// ``` ignore
/// rel 1/p :: head() = body(p)
/// ```
///
/// This rule will be transformed into
/// ``` ignore
/// rel head() = body(p) and tag#head#1#var == 1 / p and tag#head#1(tag#head#1#var)
/// ```
#[derive(Clone, Debug)]
pub struct ToAddTagPredicate {
  /// The name of the predicate
  pub name: String,

  /// The main tag expression
  pub arg_name: String,

  /// Tag location
  pub tag_loc: ast::NodeLocation,
}

impl ToAddTagPredicate {
  pub fn new(name: String, arg_name: String, tag_loc: ast::NodeLocation) -> Self {
    Self {
      name,
      arg_name,
      tag_loc,
    }
  }
}

/// An actual predicate
#[derive(Clone, Debug)]
pub struct TagPredicate {
  /// The name of he predicate
  pub name: String,

  /// args
  pub arg_ty: ValueType,
}

impl TagPredicate {
  pub fn new(name: String, arg_ty: ValueType) -> Self {
    Self { name, arg_ty }
  }
}

impl ForeignPredicate for TagPredicate {
  fn name(&self) -> String {
    self.name.clone()
  }

  fn arity(&self) -> usize {
    1
  }

  fn argument_type(&self, i: usize) -> ValueType {
    assert_eq!(i, 0);
    self.arg_ty.clone()
  }

  fn num_bounded(&self) -> usize {
    1
  }

  fn evaluate_with_env(&self, env: &RuntimeEnvironment, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    // Result tuple
    let tup = vec![];

    // Create a type cast expression and evaluate it on the given values
    let tuple = Tuple::from_values(bounded.iter().cloned());
    let cast_expr = expr::Expr::unary(unary_op::UnaryOp::TypeCast(ValueType::F64), expr::Expr::access(0));
    let maybe_computed_tag = env.eval(&cast_expr, &tuple);

    // Return the value
    if let Some(Tuple::Value(Value::F64(f))) = maybe_computed_tag {
      vec![(DynamicInputTag::Float(f), tup)]
    } else {
      vec![]
    }
  }
}
