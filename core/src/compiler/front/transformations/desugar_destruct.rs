use crate::compiler::front::*;
use crate::utils::IdAllocator;

#[derive(Clone, Debug)]
pub struct DesugarDestruct {}

impl<'a> Transformation<'a> for DesugarDestruct {}

impl DesugarDestruct {
  pub fn new() -> Self {
    Self {}
  }

  pub fn has_destruct_in_atom(&self, atom: &Atom) -> bool {
    atom.iter_args().any(|arg| arg.is_destruct())
  }

  pub fn transform_atom_with_destructor_to_formula(&self, atom: &Atom) -> (Atom, Vec<Formula>) {
    let mut variable_counter = IdAllocator::new();
    let mut all_desugared_formulas = vec![];
    let mut desugared_atom_args = vec![];

    for arg in atom.iter_args() {
      match arg {
        Expr::Destruct(destruct) => {
          let parent_id = destruct.location_id().expect("Destruct should have an ID");
          let variable = Variable::new(Identifier::new(format!("adt#destr#var#root#{parent_id}")));
          desugared_atom_args.push(Expr::Variable(variable.clone()));
          self.transform_destruct_to_formula_helper(
            variable,
            destruct,
            parent_id,
            &mut variable_counter,
            &mut all_desugared_formulas,
          );
        }
        _ => {
          desugared_atom_args.push(arg.clone());
        }
      }
    }

    let desugared_atom = Atom::new_with_loc(
      atom.predicate().clone(),
      atom.type_args().clone(),
      desugared_atom_args,
      atom.location().clone_without_id(),
    );

    (desugared_atom, all_desugared_formulas)
  }

  fn transform_destruct_to_formula_helper(
    &self,
    variable: Variable,
    destruct: &DestructExpr,
    parent_id: usize,
    variable_counter: &mut IdAllocator,
    formulas: &mut Vec<Formula>,
  ) {
    // Obtain the predicate of the atom that we are going to generate
    let predicate = destruct
      .functor()
      .clone_without_location_id()
      .map(|n| format!("adt#{n}"));

    // Obtain the second-to-last arguments in the atom
    let sub_args = destruct.iter_args().map(|arg| {
      match &arg {
        Expr::Destruct(o) => {
          // Obtain a variable id
          let variable_id = variable_counter.alloc();

          // Create a variable from the variable id
          let current_variable = Variable::new(Identifier::new(format!("adt#destr#var#{parent_id}#{variable_id}")));

          // Recurse on the object
          self.transform_destruct_to_formula_helper(current_variable.clone(), o, parent_id, variable_counter, formulas);

          // Return the variable as the result
          Expr::Variable(current_variable)
        }
        _ => arg.clone(),
      }
    });

    // Create all arguments including the variable
    let args = std::iter::once(Expr::Variable(variable)).chain(sub_args).collect();

    // Add a formula to the formulas
    let formula = Formula::Atom(Atom::new(predicate, vec![], args));
    formulas.push(formula);
  }
}

impl NodeVisitor<Formula> for DesugarDestruct {
  fn visit_mut(&mut self, formula: &mut Formula) {
    match formula {
      Formula::Atom(a) => {
        if self.has_destruct_in_atom(a) {
          let (atom, rest) = self.transform_atom_with_destructor_to_formula(a);
          *formula = Formula::conjunction(Conjunction::new(
            std::iter::once(Formula::atom(atom)).chain(rest.into_iter()).collect(),
          ));
        }
      }
      _ => {}
    }
  }
}

impl NodeVisitor<Rule> for DesugarDestruct {
  fn visit_mut(&mut self, rule: &mut Rule) {
    match rule.head_mut() {
      RuleHead::Atom(a) => {
        if self.has_destruct_in_atom(a) {
          let (atom, rest) = self.transform_atom_with_destructor_to_formula(a);
          *a = atom;
          *rule.body_mut() = Formula::conjunction(Conjunction::new(
            std::iter::once(rule.body().clone()).chain(rest.into_iter()).collect(),
          ));
        }
      }
      RuleHead::Conjunction(conj_head) => {
        if conj_head.iter_atoms().any(|atom| self.has_destruct_in_atom(atom)) {
          panic!("[Consider report this bug] Conjunction head should be handled by a prior transformation pass")
        }
      }
      RuleHead::Disjunction(disj_head) => {
        if disj_head.iter_atoms().any(|atom| self.has_destruct_in_atom(atom)) {
          unimplemented!()
        }
      }
    }
  }
}

impl NodeVisitor<FactDecl> for DesugarDestruct {
  fn visit(&mut self, fact_decl: &FactDecl) {
    if self.has_destruct_in_atom(fact_decl.atom()) {
      panic!(
        "[Consider report this bug] Fact declaration with destructor should be handled by a prior transformation pass"
      )
    }
  }
}
