use std::collections::*;

use super::super::ast::*;
use super::super::error::*;
use super::super::utils::*;
use super::super::*;

#[derive(Clone, Debug)]
pub struct ConstantDeclAnalysis {
  pub variables: HashMap<String, (Loc, Option<Type>, Constant)>,
  pub variable_use: HashMap<Loc, String>,
  pub errors: Vec<ConstantDeclError>,
}

impl ConstantDeclAnalysis {
  pub fn new() -> Self {
    Self {
      variables: HashMap::new(),
      variable_use: HashMap::new(),
      errors: vec![],
    }
  }

  pub fn get_variable(&self, var: &str) -> Option<&(Loc, Option<Type>, Constant)> {
    self.variables.get(var)
  }

  pub fn loc_of_const_type(&self, loc: &Loc) -> Option<Type> {
    self
      .variable_use
      .get(loc)
      .and_then(|v| self.variables.get(v))
      .and_then(|(_, ty, _)| ty.clone())
  }

  pub fn compute_typed_constants(&self) -> HashMap<Loc, Type> {
    self
      .variable_use
      .iter()
      .filter_map(|(var_loc, var_name)| {
        self
          .variables
          .get(var_name)
          .and_then(|(_, ty, _)| ty.clone())
          .and_then(|ty| Some((var_loc.clone(), ty)))
      })
      .collect()
  }
}

impl NodeVisitor for ConstantDeclAnalysis {
  fn visit_const_assignment(&mut self, ca: &ast::ConstAssignment) {
    // First check if the name is already declared
    if let Some((first_decl_loc, _, _)) = self.variables.get(ca.name()) {
      self.errors.push(ConstantDeclError::DuplicatedConstant {
        name: ca.name().to_string(),
        first_decl: first_decl_loc.clone(),
        second_decl: ca.location().clone(),
      })
    } else {
      // Then store the variable into the storage
      self.variables.insert(
        ca.name().to_string(),
        (ca.location().clone(), ca.ty().cloned(), ca.value().clone()),
      );
    }
  }

  fn visit_constant_set_tuple(&mut self, cst: &ConstantSetTuple) {
    for c in cst.iter_constants() {
      if let Some(v) = c.variable() {
        if self.variables.contains_key(v.name()) {
          self.variable_use.insert(v.location().clone(), v.name().to_string());
        } else {
          self.errors.push(ConstantDeclError::UnknownConstantVariable {
            name: v.name().to_string(),
            loc: v.location().clone(),
          })
        }
      }
    }
  }

  fn visit_fact_decl(&mut self, fact_decl: &FactDecl) {
    for arg in fact_decl.atom().iter_arguments() {
      let vars = arg.collect_used_variables();
      for v in vars {
        if self.variables.contains_key(v.name()) {
          self.variable_use.insert(v.location().clone(), v.name().to_string());
        } else {
          self.errors.push(ConstantDeclError::UnknownConstantVariable {
            name: v.name().to_string(),
            loc: v.location().clone(),
          });
        }
      }
    }
  }

  fn visit_variable(&mut self, v: &ast::Variable) {
    // Check if the variable is a constant variable
    if self.variables.contains_key(v.name()) {
      self.variable_use.insert(v.location().clone(), v.name().to_string());
    }
  }

  fn visit_variable_binding(&mut self, b: &ast::VariableBinding) {
    // Cannot occur in the variable binding
    if let Some((const_var_decl, _, _)) = self.variables.get(b.name()) {
      self.errors.push(ConstantDeclError::ConstantVarInBinding {
        name: b.name().to_string(),
        const_var_decl: const_var_decl.clone(),
        var_binding: b.location().clone(),
      });
    }
  }
}

#[derive(Debug, Clone)]
pub enum ConstantDeclError {
  DuplicatedConstant {
    name: String,
    first_decl: Loc,
    second_decl: Loc,
  },
  ConstantVarInBinding {
    name: String,
    const_var_decl: Loc,
    var_binding: Loc,
  },
  UnknownConstantVariable {
    name: String,
    loc: Loc,
  },
}

impl FrontCompileErrorClone for ConstantDeclError {
  fn clone_box(&self) -> Box<dyn FrontCompileErrorTrait> {
    Box::new(self.clone())
  }
}

impl FrontCompileErrorTrait for ConstantDeclError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::DuplicatedConstant {
        name,
        first_decl,
        second_decl,
      } => {
        format!(
          "duplicated declaration of constant `{}`. First declared here:\n{}\nduplicate definition here:\n{}",
          name,
          first_decl.report(src),
          second_decl.report(src)
        )
      }
      Self::ConstantVarInBinding {
        name,
        const_var_decl,
        var_binding,
      } => {
        format!("constant variable `{}` occurring in a variable binding. Consider changing the name of the variable binding:\n{}\nThe constant is declared here:\n{}", name, var_binding.report(src), const_var_decl.report(src))
      }
      Self::UnknownConstantVariable { name, loc } => {
        format!("unknown variable `{}`:\n{}", name, loc.report(src))
      }
    }
  }
}
