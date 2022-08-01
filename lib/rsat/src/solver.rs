use super::*;
use std::collections::*;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum VariableStatus {
  None,
  Pos,
  Neg,
}

impl VariableStatus {
  pub fn from_literal(l: Literal) -> Self {
    if l.is_positive() {
      Self::Pos
    } else {
      Self::Neg
    }
  }

  pub fn matches_literal(&self, l: Literal) -> bool {
    match (self, l.is_positive()) {
      (Self::None, _) | (Self::Pos, true) | (Self::Neg, false) => true,
      _ => false,
    }
  }
}

enum PossibleLiteral {
  None,
  One(Literal),
  Multiple,
}

impl PossibleLiteral {
  fn is_none(&self) -> bool {
    match self {
      Self::None => true,
      _ => false,
    }
  }
}

/// A disjunctive clause (a form like `(A \/ ~B \/ C)`)
///
/// Internally, a clause stores a list of literals.
/// It also stores an ID representing itself.
/// When used in the manager, a clause in the input CNF will have a negative ID (-1, -2, ...)
/// A learned clause will have a non-negative ID (0, 1, 2, ...) which can be directly index into the learned-clause array
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clause {
  literals: Vec<Literal>,
  id: i64,
  assertion_level: usize,
}

impl Clause {
  fn new(literals: Vec<Literal>, id: i64) -> Self {
    Self {
      literals,
      id,
      assertion_level: 0,
    }
  }

  fn new_with_assertion_level(literals: Vec<Literal>, id: i64, assertion_level: usize) -> Self {
    Self {
      literals,
      id,
      assertion_level,
    }
  }

  fn clone_with_assertion_level(&self, assertion_level: usize) -> Self {
    Self {
      literals: self.literals.clone(),
      id: self.id,
      assertion_level,
    }
  }

  pub fn len(&self) -> usize {
    self.literals.len()
  }

  pub fn is_empty(&self) -> bool {
    self.literals.is_empty()
  }

  pub fn literals(&self) -> impl Iterator<Item = &Literal> {
    self.literals.iter()
  }

  pub fn clause_id(&self) -> i64 {
    self.id
  }
}

impl std::ops::Index<usize> for Clause {
  type Output = Literal;

  fn index(&self, i: usize) -> &Self::Output {
    &self.literals[i]
  }
}

impl std::ops::IndexMut<usize> for Clause {
  fn index_mut(&mut self, i: usize) -> &mut Self::Output {
    &mut self.literals[i]
  }
}

#[derive(Clone, Debug)]
pub struct VariableInfo {
  status: VariableStatus,
  implication_level: usize,
}

#[derive(Clone, Debug)]
pub struct Solver {
  // variable_count: usize,
  variable_info: Vec<VariableInfo>,
  original_clauses: Vec<Clause>,
  learned_clauses: Vec<Clause>,

  /// The sequence of decisions
  decisions: Vec<Literal>,

  /// A literal can implied by multiple decision literals
  implications: HashMap<Literal, Clause>,
}

impl Solver {
  pub fn new(cnf: CNF) -> Self {
    // Collect the number of variable id required
    let max_var_id = cnf_max_var_id(&cnf);
    let variable_count = max_var_id + 1;

    // Variable status
    let variable_info = (0..variable_count)
      .map(|_| VariableInfo {
        status: VariableStatus::None,
        implication_level: 0,
      })
      .collect::<Vec<_>>();

    // Process the CNF into `original_clauses` annotated with IDs
    let original_clauses = cnf
      .into_iter()
      .enumerate()
      .map(|(i, lits)| Clause::new(lits, -(i as i64 + 1)))
      .collect::<Vec<_>>();

    // Construct the manager
    Self {
      // variable_count,
      variable_info,
      original_clauses,
      learned_clauses: Vec::new(),
      decisions: Vec::new(),
      implications: HashMap::new(),
    }
  }

  pub fn implied_positive(&self, variable: Variable) -> bool {
    match self.variable_status(variable) {
      VariableStatus::Pos => true,
      _ => false,
    }
  }

  pub fn implied_negative(&self, variable: Variable) -> bool {
    match self.variable_status(variable) {
      VariableStatus::Neg => true,
      _ => false,
    }
  }

  pub fn variable_status(&self, variable: Variable) -> VariableStatus {
    self.variable_info[variable.variable_id()].status
  }

  pub fn variable_implication_level(&self, variable: Variable) -> usize {
    self.variable_info[variable.variable_id()].implication_level
  }

  pub fn boolean_constraint_propagation(mut self) -> Result<Self, Clause> {
    for clause in self.original_clauses.iter().chain(self.learned_clauses.iter()) {
      // First go through the whole clause
      let mut clause_sat = false;
      let mut possible_literal = PossibleLiteral::None;
      for literal in clause.literals() {
        match (literal.is_positive(), self.variable_status(literal.variable())) {
          (true, VariableStatus::Pos) | (false, VariableStatus::Neg) => {
            // Has one positive; the whole clause is now sat
            clause_sat = true;
          }
          (false, VariableStatus::Pos) | (true, VariableStatus::Neg) => {
            // Literal unsat
          }
          _ => {
            // Free variable
            if possible_literal.is_none() {
              possible_literal = PossibleLiteral::One(literal.clone());
            } else {
              possible_literal = PossibleLiteral::Multiple;
            }
          }
        }
      }

      // If the whole clause is SAT, don't need to do any propagation in this clause
      if clause_sat {
        continue;
      }

      // Check `possible_literal`...
      if possible_literal.is_none() {
        // If there is no possible literal, then this clause is an UNSAT, and we return an asserting clause
        let conflict_clause = clause.clone();
        if let Some(top) = self.decisions.last() {
          // If there is a decision made, then we check the reason of the negate of that variable
          let neg_top = top.negate();
          if let Some(reason) = self.implications.get(&neg_top) {
            let learned_clause_id = self.learned_clauses.len() as i64;
            let learned_clause_literals = conflict_clause
              .literals()
              .chain(reason.literals())
              .filter_map(|l| {
                if l.variable_id() != top.variable_id() {
                  Some(l.negate())
                } else {
                  None
                }
              })
              .collect::<Vec<_>>();
            let assertion_level = learned_clause_literals
              .iter()
              .map(|l| self.variable_implication_level(l.variable()))
              .max()
              .unwrap_or(0);
            return Err(Clause::new_with_assertion_level(
              learned_clause_literals,
              learned_clause_id,
              assertion_level,
            ));
          } else {
            let assertion_level = conflict_clause
              .literals()
              .map(|l| self.variable_implication_level(l.variable()))
              .max()
              .unwrap_or(0);
            return Err(conflict_clause.clone_with_assertion_level(assertion_level));
          }
        } else {
          // If there is no decision made, then we get an UNSAT
          return Err(conflict_clause);
        }
      } else if let PossibleLiteral::One(l) = possible_literal {
        // If there is exactly one possible literal, then we get an implication
        self.implications.insert(l, clause.clone());
        self.variable_info[l.variable_id()].status = VariableStatus::from_literal(l);
        self.variable_info[l.variable_id()].implication_level = self.decisions.len();
      } else {
        // If there are many literals, then we cannot conclude anything
      }
    }

    Ok(self)
  }

  pub fn decide_literal(mut self, l: Literal) -> Result<Self, Clause> {
    self.decisions.push(l);
    self.variable_info[l.variable_id()].status = VariableStatus::from_literal(l);
    self.variable_info[l.variable_id()].implication_level = self.decisions.len();
    self.boolean_constraint_propagation()
  }

  pub fn at_assertion_level(&self, clause: &Clause) -> bool {
    self.decisions.len() == clause.assertion_level
  }

  pub fn assert_clause(mut self, clause: Clause) -> Result<Self, Clause> {
    if clause.id < self.learned_clauses.len() as i64 {
      self.learned_clauses.push(clause);
      self.boolean_constraint_propagation()
    } else {
      Ok(self)
    }
  }

  pub fn solve_with_variable_order(self, var_order: &[Variable]) -> Result<(), Clause> {
    if var_order.len() == 0 {
      Ok(())
    } else {
      let first_variable = var_order[0].clone();
      let result = match self.variable_status(first_variable) {
        VariableStatus::None => {
          let literal = Literal::positive(first_variable);
          self
            .clone()
            .decide_literal(literal)
            .and_then(|new| new.solve_with_variable_order(&var_order[1..]))
        }
        VariableStatus::Pos | VariableStatus::Neg => self.clone().solve_with_variable_order(&var_order[1..]),
      };
      if let Err(asserting_clause) = result {
        if asserting_clause.assertion_level == self.decisions.len() {
          if self.decisions.len() == 0 {
            Err(asserting_clause)
          } else {
            self
              .assert_clause(asserting_clause)
              .and_then(|new| new.solve_with_variable_order(var_order))
          }
        } else {
          Err(asserting_clause)
        }
      } else {
        Ok(())
      }
    }
  }

  pub fn model_counting_with_variable_order(&self, var_order: &[Variable]) -> Result<usize, Clause> {
    if var_order.len() == 0 {
      Ok(1)
    } else {
      let first_variable = var_order[0].clone();
      match self.variable_status(first_variable) {
        VariableStatus::Pos | VariableStatus::Neg => self.model_counting_with_variable_order(&var_order[1..]),
        _ => {
          let plit = Literal::positive(first_variable);
          let pcount = match self.clone().decide_literal(plit) {
            Ok(new) => new.model_counting_with_variable_order(&var_order[1..])?,
            Err(asserting_clause) => {
              if self.at_assertion_level(&asserting_clause) {
                return self
                  .clone()
                  .assert_clause(asserting_clause)
                  .unwrap()
                  .model_counting_with_variable_order(var_order);
              } else {
                return Err(asserting_clause);
              }
            }
          };
          let nlit = Literal::negative(first_variable);
          let ncount = match self.clone().decide_literal(nlit) {
            Ok(new) => new.model_counting_with_variable_order(&var_order[1..])?,
            Err(asserting_clause) => {
              if self.at_assertion_level(&asserting_clause) {
                return self
                  .clone()
                  .assert_clause(asserting_clause)
                  .unwrap()
                  .model_counting_with_variable_order(var_order);
              } else {
                return Err(asserting_clause);
              }
            }
          };
          Ok(pcount + ncount)
        }
      }
    }
  }
}

fn cnf_max_var_id(cnf: &CNF) -> usize {
  cnf
    .iter()
    .map(|clause| clause.iter().map(|lit| lit.variable_id()).max().unwrap())
    .max()
    .unwrap()
}
