use crate::compiler::front::*;
use crate::utils::IdAllocator;

#[derive(Clone, Debug)]
pub struct DesugarCaseIs;

impl<'a> Transformation<'a> for DesugarCaseIs {}

impl DesugarCaseIs {
  pub fn new() -> Self {
    Self
  }

  pub fn transform_case_is_to_formula(&self, case: &Case) -> Formula {
    match &case.entity() {
      Entity::Expr(e) => {
        // If the entity is directly an expression, the formula is a constraint
        Formula::Constraint(Constraint::new_with_loc(
          Expr::binary(BinaryExpr::new(
            BinaryOp::new_eq(),
            Expr::Variable(case.variable().clone()),
            e.clone(),
          )),
          case.location().clone(),
        ))
      }
      Entity::Object(o) => {
        // If the entity is an object, the formula is a conjunction of atoms
        let parent_id = case.variable_name();
        let variable = case.variable().clone();
        let mut variable_counter = IdAllocator::new();
        let mut formulas = vec![];

        // Recurse through the entity to create formulas
        self.transform_object_to_formula_helper(variable, o, parent_id, &mut variable_counter, &mut formulas);

        // Return the conjunction of formulas
        Formula::conjunction(Conjunction::new(formulas))
      }
    }
  }

  fn transform_object_to_formula_helper(
    &self,
    variable: Variable,
    object: &Object,
    parent_id: &String,
    variable_counter: &mut IdAllocator,
    formulas: &mut Vec<Formula>,
  ) {
    // Obtain the predicate of the atom that we are going to generate
    let predicate = object.functor().clone_without_location_id().map(|n| format!("adt#{n}"));

    // Obtain the second-to-last arguments in the atom
    let sub_args = object.iter_args().map(|arg| {
      match &arg {
        Entity::Expr(e) => e.clone(),
        Entity::Object(o) => {
          // Obtain a variable id
          let variable_id = variable_counter.alloc();

          // Create a variable from the variable id
          let current_variable = Variable::new(Identifier::new(format!("adt#var#({parent_id})#{variable_id}")));

          // Recurse on the object
          self.transform_object_to_formula_helper(current_variable.clone(), o, parent_id, variable_counter, formulas);

          // Return the variable as the result
          Expr::Variable(current_variable)
        }
      }
    });

    // Create all arguments including the variable
    let args = std::iter::once(Expr::Variable(variable)).chain(sub_args).collect();

    // Add a formula to the formulas
    let formula = Formula::Atom(Atom::new(predicate, vec![], args));
    formulas.push(formula);
  }
}

impl NodeVisitor<Formula> for DesugarCaseIs {
  fn visit_mut(&mut self, formula: &mut Formula) {
    match formula {
      Formula::Case(c) => *formula = self.transform_case_is_to_formula(c),
      _ => {}
    }
  }
}
