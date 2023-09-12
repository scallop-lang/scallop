use std::collections::*;

use super::super::ast::*;
use super::super::error::*;
use super::super::utils::*;
use super::super::*;

/// Constant declaration analysis
///
/// Analyzes the constant declarations coming from `ConstAssignment` and `EnumTypeDecl`.
/// After walking through AST, the analysis checks whether there is duplicated constant
/// declarations, unknown constants, and etc.
/// It stores the locations and other information where a constant is used and declared.
#[derive(Clone, Debug)]
pub struct ConstantDeclAnalysis {
  pub variables: HashMap<String, (Loc, Option<Type>, Constant)>,
  pub variable_use: HashMap<Loc, String>,
  pub entity_facts: Vec<EntityFact>,
  pub errors: Vec<ConstantDeclError>,
}

impl ConstantDeclAnalysis {
  /// Create a new analysis
  pub fn new() -> Self {
    Self {
      variables: HashMap::new(),
      variable_use: HashMap::new(),
      entity_facts: Vec::new(),
      errors: vec![],
    }
  }

  /// Get the variable information stored in the analysis, including
  /// its declaration location, its type, and the constant it is associated with.
  /// `None` is returned if such variable does not exist.
  pub fn get_variable(&self, var: &str) -> Option<&(Loc, Option<Type>, Constant)> {
    self.variables.get(var)
  }

  /// Given a location where a constant variable is used, find the type of that variable.
  /// `None` is returned if this location is not recorded or the variable is not annotated with a type.
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

  pub fn process_enum_type_decl(&mut self, etd: &ast::EnumTypeDecl) -> Result<(), ConstantDeclError> {
    let extract_value = |member: &ast::EnumTypeMember, prev_max: Option<i64>| -> Result<i64, ConstantDeclError> {
      // First check if there is an integer number assignment to the enum
      match member.assigned_num() {
        Some(c) => match c {
          // If there is, we check if the integer is greater than or equal to zero and greater than the previous maximum
          Constant::Integer(i) if i.int() >= &0 => {
            let i = *i.int();
            // Check if we have a previous number already
            if let Some(prev_max) = prev_max {
              if i > prev_max {
                // If the number is greater than previous number, then ok to directly assign the number
                return Ok(i);
              } else {
                // If the number is not greater, then this enum value ID is invalid
                return Err(ConstantDeclError::EnumIDAlreadyAssigned {
                  curr_name: member.name().to_string(),
                  id: i,
                  loc: member.location().clone(),
                });
              }
            } else {
              // If there is no previous max, then directly give it `i`.
              return Ok(i);
            }
          }
          _ => {
            // We don't care other cases
          }
        },
        _ => {}
      };

      // If the assignment is not presented, we simply increment the previous maximum value
      if let Some(prev_max) = prev_max {
        return Ok(prev_max + 1);
      } else {
        return Ok(0);
      }
    };

    let mut process_member = |member: &ast::EnumTypeMember, id: i64| -> Result<(), ConstantDeclError> {
      if let Some((first_decl_loc, _, _)) = self.variables.get(member.member_name()) {
        Err(ConstantDeclError::DuplicatedConstant {
          name: member.name().to_string(),
          first_decl: first_decl_loc.clone(),
          second_decl: member.location().clone(),
        })
      } else {
        // Then store the variable into the storage
        self.variables.insert(
          member.name().to_string(),
          (
            member.location().clone(),
            Some(Type::usize()),
            Constant::integer(IntLiteral::new(id as i64)),
          ),
        );
        Ok(())
      }
    };

    // Go through all the members
    let mut members_iterator = etd.iter_members();

    // First process the first member
    let first_member = members_iterator.next().unwrap(); // Unwrap is ok since there has to be at least two components
    let mut curr_id = extract_value(first_member, None)?;
    process_member(first_member, curr_id)?;

    // Then process the rest
    while let Some(curr_member) = members_iterator.next() {
      curr_id = extract_value(curr_member, Some(curr_id))?;
      process_member(curr_member, curr_id)?;
    }

    Ok(())
  }
}

impl NodeVisitor<ConstAssignment> for ConstantDeclAnalysis {
  fn visit(&mut self, ca: &ConstAssignment) {
    // First check if the name is already declared
    if let Some((first_decl_loc, _, _)) = self.variables.get(ca.variable_name()) {
      self.errors.push(ConstantDeclError::DuplicatedConstant {
        name: ca.name().to_string(),
        first_decl: first_decl_loc.clone(),
        second_decl: ca.location().clone(),
      })
    } else {
      let entity = ca.value();

      // Then we make sure that the entity is indeed a constant
      if let Some(var_loc) = entity.get_first_non_constant_location(&|v| self.variables.contains_key(v.variable_name())) {
        self.errors.push(ConstantDeclError::EntityContainsNonConstant {
          const_decl_loc: ca.location().clone(),
          var_loc: var_loc.clone(),
        })
      } else {
        // Annotate the type of the entity
        let ty = if entity.is_constant() {
          ca.ty().as_ref().cloned()
        } else {
          Some(Type::entity())
        };

        // Process the entity into a set of entity facts and one final constant value
        let (entity_facts, constant) =
          entity.to_facts_with_constant_variables(|v|
            self
              .variables
              .get(v.name().name())
              .map(|(_, _, c)| c.clone())
          );

        // Extend the entity facts with the storage
        self.entity_facts.extend(entity_facts);

        // Store the variable
        self.variables.insert(
          ca.name().name().to_string(),
          (ca.location().clone(), ty, constant),
        );
      }
    }
  }
}

impl NodeVisitor<EnumTypeDecl> for ConstantDeclAnalysis {
  fn visit(&mut self, etd: &EnumTypeDecl) {
    if let Err(e) = self.process_enum_type_decl(etd) {
      self.errors.push(e);
    }
  }
}

impl NodeVisitor<ConstantSetTuple> for ConstantDeclAnalysis {
  fn visit(&mut self, cst: &ConstantSetTuple) {
    for c in cst.iter_constants() {
      if let Some(v) = c.as_variable() {
        if self.variables.contains_key(v.variable_name()) {
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
}

impl NodeVisitor<FactDecl> for ConstantDeclAnalysis {
  fn visit(&mut self, fact_decl: &FactDecl) {
    for arg in fact_decl.atom().iter_args() {
      let vars = arg.collect_used_variables();
      for v in vars {
        if self.variables.contains_key(v.variable_name()) {
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
}

impl NodeVisitor<Variable> for ConstantDeclAnalysis {
  fn visit(&mut self, v: &Variable) {
    // Check if the variable is a constant variable
    if self.variables.contains_key(v.name().name()) {
      self.variable_use.insert(v.location().clone(), v.name().to_string());
    }
  }
}

impl NodeVisitor<VariableBinding> for ConstantDeclAnalysis {
  fn visit(&mut self, b: &VariableBinding) {
    // Cannot occur in the variable binding
    if let Some((const_var_decl, _, _)) = self.variables.get(b.variable_name()) {
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
  EnumIDAlreadyAssigned {
    curr_name: String,
    id: i64,
    loc: Loc,
  },
  EntityContainsNonConstant {
    const_decl_loc: Loc,
    var_loc: Loc,
  },
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
      Self::EnumIDAlreadyAssigned { curr_name, id, loc } => {
        format!(
          "the enum ID `{}` for variant `{}` has already been assigned\n{}",
          id,
          curr_name,
          loc.report(src)
        )
      }
      Self::EntityContainsNonConstant { var_loc, .. } => {
        format!(
          "non-constant expression found in constant entity:\n{}",
          var_loc.report(src)
        )
      }
    }
  }
}
