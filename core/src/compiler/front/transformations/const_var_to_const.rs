use crate::{common::input_tag::DynamicInputTag, compiler::front::analyzers::ConstantDeclAnalysis};

use super::super::*;

#[derive(Clone, Debug)]
pub struct TransformConstVarToConst<'a> {
  const_decl_analysis: &'a ConstantDeclAnalysis,
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
        Item::RelationDecl(
          RelationDeclNode::Fact(
            FactDeclNode {
              attrs: Attributes::new(),
              tag: TagNode(DynamicInputTag::None).into(),
              atom: Atom {
                loc: entity_fact.loc.clone(),
                node: AtomNode {
                  predicate: {
                    entity_fact
                      .functor
                      .clone_without_location_id()
                      .map(|n| format!("adt#{n}"))
                  },
                  type_args: vec![],
                  args: {
                    std::iter::once(&entity_fact.id)
                      .chain(entity_fact.args.iter())
                      .cloned()
                      .map(Expr::Constant)
                      .collect()
                  },
                },
              },
            }
            .into(),
          )
          .into(),
        )
      })
      .collect()
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
