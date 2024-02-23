use super::*;

/// An aggregation operation, e.g. `n = count(p: person(p))`
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Reduce {
  pub left: Vec<VariableOrWildcard>,
  pub operator: ReduceOp,
  pub args: Vec<Variable>,
  pub bindings: Vec<VariableBinding>,
  pub body: Box<Formula>,
  pub group_by: Option<(Vec<VariableBinding>, Box<Formula>)>,
}

impl Reduce {
  pub fn iter_left_variables(&self) -> impl Iterator<Item = &Variable> {
    self.left().iter().filter_map(|i| match i {
      VariableOrWildcard::Variable(v) => Some(v),
      _ => None,
    })
  }

  pub fn iter_binding_names(&self) -> impl Iterator<Item = &String> {
    self.bindings().iter().map(|b| b.name().name())
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _NamedReduceParam {
  pub name: Identifier,
  pub value: Constant,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub enum ReduceParam {
  Positional(Constant),
  Named(NamedReduceParam),
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ReduceOp {
  pub name: Identifier,
  pub parameters: Vec<ReduceParam>,
  pub has_exclaimation_mark: bool,
}

/// An syntax sugar for forall/exists operation, e.g. `forall(p: person(p) => father(p, _))`.
/// In this case, the assigned variable is omitted for abbrevity.
#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ForallExistsReduce {
  pub negate: bool,
  pub operator: ReduceOp,
  pub bindings: Vec<VariableBinding>,
  pub body: Box<Formula>,
  pub group_by: Option<(Vec<VariableBinding>, Box<Formula>)>,
}

impl ForallExistsReduce {
  pub fn is_negated(&self) -> bool {
    self.negate().clone()
  }

  pub fn iter_binding_names(&self) -> impl Iterator<Item = &String> {
    self.iter_bindings().map(|b| b.name().name())
  }
}
