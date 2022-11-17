use crate::compiler::front::analyzers::ConstantDeclAnalysis;

use super::super::*;

#[derive(Clone, Debug)]
pub struct TransformConstVarToConst<'a> {
  const_decl_analysis: &'a ConstantDeclAnalysis,
}

impl<'a> TransformConstVarToConst<'a> {
  pub fn new(const_decl_analysis: &'a ConstantDeclAnalysis) -> Self {
    Self { const_decl_analysis }
  }
}

impl<'a> NodeVisitorMut for TransformConstVarToConst<'a> {
  fn visit_expr(&mut self, expr: &mut Expr) {
    match expr {
      Expr::Variable(v) => {
        // First make sure that we are about to change this variable
        if self.const_decl_analysis.variable_use.contains_key(v.location()) {
          // Then get the constant that the const variable maps to
          let (_, _, c) = self.const_decl_analysis.get_variable(v.name()).unwrap();

          // Update the expression to be a constant
          *expr = Expr::Constant(Constant::new(v.location().clone(), c.node.clone()));
        }
      }
      _ => {}
    }
  }

  fn visit_constant_or_variable(&mut self, cov: &mut ConstantOrVariable) {
    match cov {
      ConstantOrVariable::Variable(v) => {
        // First make sure that we are about to change this variable
        if self.const_decl_analysis.variable_use.contains_key(v.location()) {
          // Then get the constant that the const variable maps to
          let (_, _, c) = self.const_decl_analysis.get_variable(v.name()).unwrap();

          // Update the expression to be a constant
          *cov = ConstantOrVariable::Constant(Constant::new(v.location().clone(), c.node.clone()));
        }
      }
      _ => {}
    }
  }
}
