use std::collections::BTreeSet;

use super::Literal;

#[derive(Clone, PartialEq, PartialOrd, Eq)]
pub struct Clause {
  pub literals: Vec<Literal>,
}

impl std::fmt::Debug for Clause {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_set().entries(&self.literals).finish()
  }
}

impl std::fmt::Display for Clause {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, literal) in self.literals.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{}", literal))?;
    }
    f.write_str("}")
  }
}

impl Clause {
  /// Create a new clause using a set of literals
  ///
  /// This function assumes that the literals are sorted,
  /// and there is no conflicting fact (`pos(x)` and `neg(x)`)
  pub fn new(literals: Vec<Literal>) -> Self {
    Self { literals }
  }

  /// Create a new clause using a set of literals.
  /// If there is conflicting fact (`pos(x)` and `neg(x)`) then we return `None`
  ///
  /// Assumption: Assumes that the literals are sorted
  pub fn new_with_pos_neg_check(literals: Vec<Literal>) -> Option<Self> {
    for i in 0..literals.len() - 1 {
      let curr = &literals[i];
      let next = &literals[i + 1];
      match (curr, next) {
        (Literal::Pos(x), Literal::Neg(y)) if x == y => return None,
        (Literal::Neg(x), Literal::Pos(y)) if x == y => return None,
        _ => {}
      }
    }
    Some(Self { literals })
  }

  /// Check if there are both pos(v) and neg(v) in the same clause. If there are, return `false`
  /// representing "not valid". Otherwise, the clause is valid.
  pub fn is_valid(&self) -> bool {
    // If the clause is empty, then this is valid
    if self.literals.is_empty() {
      return true;
    }

    // Otherwise, check for contiguous pos(v)/neg(v)
    for i in 0..self.literals.len() - 1 {
      let curr = &self.literals[i];
      let next = &self.literals[i + 1];
      match (curr, next) {
        (Literal::Pos(x), Literal::Neg(y)) if x == y => return false,
        (Literal::Neg(x), Literal::Pos(y)) if x == y => return false,
        _ => {}
      }
    }
    true
  }

  pub fn empty() -> Self {
    Self { literals: Vec::new() }
  }

  pub fn len(&self) -> usize {
    self.literals.len()
  }

  pub fn is_empty(&self) -> bool {
    self.literals.is_empty()
  }

  pub fn iter(&self) -> impl Iterator<Item = &Literal> + '_ {
    self.literals.iter()
  }

  pub fn singleton(l: Literal) -> Self {
    Self { literals: vec![l] }
  }

  pub fn pos_fact_ids(&self) -> BTreeSet<usize> {
    self
      .literals
      .iter()
      .filter_map(|l| match l {
        Literal::Pos(v) => Some(v.clone()),
        Literal::Neg(_) => None,
      })
      .collect()
  }

  /// Merge two clauses in an unchecked manner
  ///
  /// A clause can represent either a conjunction over simple literals or a disjunction.
  /// We assume when the function `merge` is called, the two operands are of the same type
  /// (i.e. both are conjunctions or both are disjunctions).
  /// We also assume that the literals in the clauses are sorted in incrementing order of
  /// their `fact_id`.
  ///
  /// We don't check if pos(v) and neg(v) occur in the same clause
  pub fn merge_unchecked(&self, rhs: &Self) -> Self {
    let mut new_literals = vec![];
    let (mut i, mut j) = (0, 0);

    // The main loop is an iteration that iterates two pointers in the two vectors of
    // `Literal`s. Since the input vectors are sorted we can do an `O(n)` iteration to
    // preserve the orderedness so that the resulting vector of literal is also sorted.
    // The loop terminates when both vectors are exhausted.
    loop {
      if i < self.literals.len() && j < rhs.literals.len() {
        let fact_i = self.literals[i].fact_id();
        let fact_j = rhs.literals[j].fact_id();
        match fact_i.cmp(&fact_j) {
          std::cmp::Ordering::Equal => {
            match (self.literals[i].sign(), rhs.literals[j].sign()) {
              (true, true) => new_literals.push(Literal::Pos(fact_i)),
              (false, false) => new_literals.push(Literal::Neg(fact_i)),
              _ => {
                new_literals.push(Literal::Pos(fact_i));
                new_literals.push(Literal::Neg(fact_i));
              }
            }
            i += 1;
            j += 1;
          }
          std::cmp::Ordering::Less => {
            new_literals.push(self.literals[i].clone());
            i += 1;
          }
          std::cmp::Ordering::Greater => {
            new_literals.push(rhs.literals[j].clone());
            j += 1;
          }
        }
      } else if i < self.literals.len() {
        new_literals.push(self.literals[i].clone());
        i += 1;
      } else if j < rhs.literals.len() {
        new_literals.push(rhs.literals[j].clone());
        j += 1;
      } else {
        break;
      }
    }
    Self::new(new_literals)
  }

  /// Merge two clauses in an unchecked manner
  ///
  /// A clause can represent either a conjunction over simple literals or a disjunction.
  /// We assume when the function `merge` is called, the two operands are of the same type
  /// (i.e. both are conjunctions or both are disjunctions).
  /// We also assume that the literals in the clauses are sorted in incrementing order of
  /// their `fact_id`.
  ///
  /// We don't check if pos(v) and neg(v) occur in the same clause
  pub fn merge(&self, rhs: &Self) -> Option<Self> {
    let mut new_literals = vec![];
    let (mut i, mut j) = (0, 0);

    // The main loop is an iteration that iterates two pointers in the two vectors of
    // `Literal`s. Since the input vectors are sorted we can do an `O(n)` iteration to
    // preserve the orderedness so that the resulting vector of literal is also sorted.
    // The loop terminates when both vectors are exhausted.
    loop {
      if i < self.literals.len() && j < rhs.literals.len() {
        let fact_i = self.literals[i].fact_id();
        let fact_j = rhs.literals[j].fact_id();
        match fact_i.cmp(&fact_j) {
          std::cmp::Ordering::Equal => {
            match (self.literals[i].sign(), rhs.literals[j].sign()) {
              (true, true) => new_literals.push(Literal::Pos(fact_i)),
              (false, false) => new_literals.push(Literal::Neg(fact_i)),
              _ => {
                // There is both positive and negative in this clause; discard
                return None;
              }
            }
            i += 1;
            j += 1;
          }
          std::cmp::Ordering::Less => {
            new_literals.push(self.literals[i].clone());
            i += 1;
          }
          std::cmp::Ordering::Greater => {
            new_literals.push(rhs.literals[j].clone());
            j += 1;
          }
        }
      } else if i < self.literals.len() {
        new_literals.push(self.literals[i].clone());
        i += 1;
      } else if j < rhs.literals.len() {
        new_literals.push(rhs.literals[j].clone());
        j += 1;
      } else {
        break;
      }
    }

    Some(Self::new(new_literals))
  }
}
