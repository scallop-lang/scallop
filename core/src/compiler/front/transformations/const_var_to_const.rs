use crate::compiler::front::analyzers::ConstantDeclAnalysis;
use crate::compiler::front::transformations::TransformAlgebraicDataType;

use super::super::*;

pub struct TransformConstVarToConst2<'a> {
  parent: TransformConstVarToConst<'a>,
}

impl<'a> TransformConstVarToConst2<'a> {
  pub fn new(const_decl_analysis: &'a ConstantDeclAnalysis) -> Self {
    Self {
      parent: TransformConstVarToConst::new(const_decl_analysis),
    }
  }
}

impl<'a> Transformation<'a> for TransformConstVarToConst2<'a> {
  fn dependencies(&self) -> Vec<&'static str> {
    vec![TransformAlgebraicDataType::name()]
  }
}

impl<'a> NodeVisitor<Expr> for TransformConstVarToConst2<'a> {
  fn visit_mut(&mut self, expr: &mut Expr) {
    self.parent.visit_mut(expr);
  }
}

impl<'a> NodeVisitor<ConstantOrVariable> for TransformConstVarToConst2<'a> {
  fn visit_mut(&mut self, cov: &mut ConstantOrVariable) {
    self.parent.visit_mut(cov);
  }
}

#[derive(Clone, Debug)]
pub struct TransformConstVarToConst<'a> {
  const_decl_analysis: &'a ConstantDeclAnalysis,
}

impl<'a> Transformation<'a> for TransformConstVarToConst<'a> {
  fn post_walking_generated_items(&mut self) -> Vec<Item> {
    self.generate_items()
  }
}

impl<'a> TransformConstVarToConst<'a> {
  pub fn new(const_decl_analysis: &'a ConstantDeclAnalysis) -> Self {
    Self { const_decl_analysis }
  }

  pub fn generate_items(&self) -> Vec<Item> {
    self
      .const_decl_analysis
      .entity_facts
      .iter()
      .map(|entity_fact| {
        Item::RelationDecl(RelationDecl::Fact(FactDecl::new(
          Attributes::new(),
          None,
          Atom::new_with_loc(
            {
              entity_fact
                .functor
                .clone_without_location_id()
                .map(|n| format!("adt#{n}"))
            },
            vec![],
            {
              std::iter::once(&entity_fact.id)
                .chain(entity_fact.args.iter())
                .cloned()
                .map(Expr::Constant)
                .collect()
            },
            entity_fact.loc.clone(),
          ),
        )))
      })
      .collect()
  }
}

impl<'a> NodeVisitor<Expr> for TransformConstVarToConst<'a> {
  fn visit_mut(&mut self, expr: &mut Expr) {
    match expr {
      Expr::Variable(v) => {
        // First make sure that we are about to change this variable
        if self.const_decl_analysis.variable_use.contains_key(v.location()) {
          // Then get the constant that the const variable maps to
          let (_, _, c) = self.const_decl_analysis.get_variable(v.name().name()).unwrap();

          // Update the expression to be a constant
          *expr = Expr::Constant(c.clone_with_loc(v.location().clone()));
        }
      }
      _ => {}
    }
  }
}

impl<'a> NodeVisitor<ConstantOrVariable> for TransformConstVarToConst<'a> {
  fn visit_mut(&mut self, cov: &mut ConstantOrVariable) {
    match cov {
      ConstantOrVariable::Variable(v) => {
        // First make sure that we are about to change this variable
        if self.const_decl_analysis.variable_use.contains_key(v.location()) {
          // Then get the constant that the const variable maps to
          let (_, _, c) = self.const_decl_analysis.get_variable(v.name().name()).unwrap();

          // Update the expression to be a constant
          *cov = ConstantOrVariable::Constant(c.clone_with_loc(v.location().clone()));
        }
      }
      _ => {}
    }
  }
}
