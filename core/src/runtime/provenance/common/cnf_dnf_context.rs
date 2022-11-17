use itertools::Itertools;
use std::collections::*;

use super::{CNFDNFFormula, Clause, FormulaKind, Literal};

pub trait CNFDNFContextTrait {
  fn fact_probability(&self, id: &usize) -> f64;

  fn has_disjunction_conflict(&self, pos_facts: &BTreeSet<usize>) -> bool;

  fn literal_probability(&self, l: &Literal) -> f64 {
    match l {
      Literal::Pos(v) => self.fact_probability(v),
      Literal::Neg(v) => 1.0 - self.fact_probability(v),
    }
  }

  fn conjunction_probability(&self, c: &Clause) -> f64 {
    c.literals.iter().fold(1.0, |acc, l| acc * self.literal_probability(l))
  }

  fn disjunction_probability(&self, c: &Clause) -> f64 {
    1.0
      - c
        .literals
        .iter()
        .fold(1.0, |acc, l| acc * self.literal_probability(&l.negate()))
  }

  fn add_formula_top_k(&self, f1: &Vec<Clause>, f2: &Vec<Clause>, k: usize) -> CNFDNFFormula {
    let mut result_clauses = vec![];
    let (mut i, mut j) = (0, 0);
    let (mut prob_i, mut prob_j) = (0.0, 0.0);

    // Before iteration, update prob i and prob j
    if i < f1.len() {
      prob_i = self.conjunction_probability(&f1[i]);
    }
    if j < f2.len() {
      prob_j = self.conjunction_probability(&f2[j]);
    }

    // Closures for cleaner code
    let incr_and_update = |index: &mut usize, prob: &mut f64, f: &Vec<Clause>| {
      *index += 1;
      if *index < f.len() {
        *prob = self.conjunction_probability(&f[*index]);
      }
    };

    // Enter iteration finding the top k proofs
    while result_clauses.len() < k {
      if i < f1.len() && j < f2.len() {
        let (clause_i, clause_j) = (&f1[i], &f2[j]);
        if clause_i == clause_j {
          result_clauses.push(clause_i.clone());
          incr_and_update(&mut i, &mut prob_i, f1);
          incr_and_update(&mut j, &mut prob_j, f2);
        } else if prob_i > prob_j {
          result_clauses.push(clause_i.clone());
          incr_and_update(&mut i, &mut prob_i, f1);
        } else {
          /* prob_j > prob_i */
          result_clauses.push(clause_j.clone());
          incr_and_update(&mut j, &mut prob_j, f2);
        }
      } else if i < f1.len() {
        result_clauses.push(f1[i].clone());
        incr_and_update(&mut i, &mut prob_i, f1);
      } else if j < f2.len() {
        result_clauses.push(f2[j].clone());
        incr_and_update(&mut j, &mut prob_j, f2);
      } else {
        break;
      }
    }

    // Create
    CNFDNFFormula::dnf(result_clauses)
  }

  fn add_formula_bottom_k(&self, f1: &Vec<Clause>, f2: &Vec<Clause>, k: usize) -> CNFDNFFormula {
    let mut result_clauses = vec![];
    let (mut i, mut j) = (0, 0);
    let (mut prob_i, mut prob_j) = (0.0, 0.0);

    // Before iteration, update prob i and prob j
    if i < f1.len() {
      prob_i = self.disjunction_probability(&f1[i]);
    }
    if j < f2.len() {
      prob_j = self.disjunction_probability(&f2[j]);
    }

    // Closures for cleaner code
    let incr_and_update = |index: &mut usize, prob: &mut f64, f: &Vec<Clause>| {
      *index += 1;
      if *index < f.len() {
        *prob = self.disjunction_probability(&f[*index]);
      }
    };

    // Enter iteration finding the top k proofs
    while result_clauses.len() < k {
      if i < f1.len() && j < f2.len() {
        let (clause_i, clause_j) = (&f1[i], &f2[j]);
        if clause_i == clause_j {
          result_clauses.push(clause_i.clone());
          incr_and_update(&mut i, &mut prob_i, f1);
          incr_and_update(&mut j, &mut prob_j, f2);
        } else if prob_i < prob_j {
          result_clauses.push(clause_i.clone());
          incr_and_update(&mut i, &mut prob_i, f1);
        } else {
          /* prob_j < prob_i */
          result_clauses.push(clause_j.clone());
          incr_and_update(&mut j, &mut prob_j, f2);
        }
      } else if i < f1.len() {
        result_clauses.push(f1[i].clone());
        incr_and_update(&mut i, &mut prob_i, f1);
      } else if j < f2.len() {
        result_clauses.push(f2[j].clone());
        incr_and_update(&mut j, &mut prob_j, f2);
      } else {
        break;
      }
    }

    // Create
    CNFDNFFormula::cnf(result_clauses)
  }

  fn mult_formula_top_k(&self, f1: &Vec<Clause>, f2: &Vec<Clause>, k: usize) -> CNFDNFFormula {
    #[derive(PartialEq)]
    struct Element {
      prob: f64,
      i: usize,
      j: usize,
      clause: Clause,
    }

    impl std::cmp::Eq for Element {}

    impl std::cmp::PartialOrd for Element {
      fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.prob.partial_cmp(&other.prob)
      }
    }

    impl std::cmp::Ord for Element {
      fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
      }
    }

    // If any one of f1 and f2 is empty, then cartesian product is empty
    if f1.len() == 0 || f2.len() == 0 {
      return CNFDNFFormula::dnf_zero().into();
    } else {
      // Shortcut for the two formula are exactly the same
      if f1 == f2 {
        return CNFDNFFormula::dnf(f1.clone()).into();
      }

      // Setup variables
      let mut heap = BinaryHeap::<Element>::new();
      let mut visited = HashSet::<(usize, usize)>::new();
      let mut result_clauses = Vec::new();

      // Closure for cleaner code
      let create_element = |i: usize, j: usize| {
        if i < f1.len() && j < f2.len() {
          let clause = f1[i].merge_unchecked(&f2[j]);
          let prob = self.conjunction_probability(&clause);
          Some(Element { prob, i, j, clause })
        } else {
          None
        }
      };

      // First element; there must be one since f1 and f2 are non-empty
      let first_elem = create_element(0, 0).unwrap();
      heap.push(first_elem);

      // Enter the loop
      while result_clauses.len() < k {
        // Pop one from the heap
        if let Some(elem) = heap.pop() {
          if !visited.contains(&(elem.i, elem.j)) {
            visited.insert((elem.i, elem.j));

            // Check validity of this clause
            if elem.clause.is_valid() {
              // Check if there is disjunctive conflict
              if !self.has_disjunction_conflict(&elem.clause.pos_fact_ids()) {
                // if valid, put it in the result
                result_clauses.push(elem.clause);
              }
            }

            // Then push the two successors into the heap
            if let Some(s1) = create_element(elem.i + 1, elem.j) {
              heap.push(s1);
            }
            if let Some(s2) = create_element(elem.i, elem.j + 1) {
              heap.push(s2);
            }
          } else {
            // Skip the current iteration
          }
        } else {
          break;
        }
      }

      // Return the result
      CNFDNFFormula::dnf(result_clauses).into()
    }
  }

  fn mult_formula_bottom_k(&self, f1: &Vec<Clause>, f2: &Vec<Clause>, k: usize) -> CNFDNFFormula {
    #[derive(PartialEq)]
    struct Element {
      prob: f64,
      i: usize,
      j: usize,
      clause: Clause,
    }

    impl std::cmp::Eq for Element {}

    impl std::cmp::PartialOrd for Element {
      fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.prob.partial_cmp(&other.prob).map(std::cmp::Ordering::reverse)
      }
    }

    impl std::cmp::Ord for Element {
      fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
      }
    }

    // If any one of f1 and f2 is empty, then cartesian product is empty
    if f1.len() == 0 || f2.len() == 0 {
      return CNFDNFFormula::cnf_zero().into();
    } else {
      // Shortcut for the two formula are exactly the same
      if f1 == f2 {
        return CNFDNFFormula::cnf(f1.clone()).into();
      }

      // Setup variables
      let mut heap = BinaryHeap::<Element>::new();
      let mut visited = HashSet::<(usize, usize)>::new();
      let mut result_clauses = Vec::new();

      // Closure for cleaner code
      let create_element = |i: usize, j: usize| {
        if i < f1.len() && j < f2.len() {
          let clause = f1[i].merge_unchecked(&f2[j]);
          let prob = self.disjunction_probability(&clause);
          Some(Element { prob, i, j, clause })
        } else {
          None
        }
      };

      // First element; there must be one since f1 and f2 are non-empty
      let first_elem = create_element(0, 0).unwrap();
      heap.push(first_elem);

      // Enter the loop
      while result_clauses.len() < k {
        // Pop one from the heap
        if let Some(elem) = heap.pop() {
          if !visited.contains(&(elem.i, elem.j)) {
            visited.insert((elem.i, elem.j));

            // Check validity of this clause; if valid, put it in the result
            if elem.clause.is_valid() {
              result_clauses.push(elem.clause);
            }

            // Then push the two successors into the heap
            if let Some(s1) = create_element(elem.i + 1, elem.j) {
              heap.push(s1);
            }
            if let Some(s2) = create_element(elem.i, elem.j + 1) {
              heap.push(s2);
            }
          } else {
            // Skip the current iteration
          }
        } else {
          break;
        }
      }

      // Return the result
      CNFDNFFormula::cnf(result_clauses)
    }
  }

  fn cnf2dnf_k(&self, f: &Vec<Clause>, k: usize) -> CNFDNFFormula {
    #[derive(PartialEq)]
    struct Element {
      prob: f64,
      indices: Vec<usize>,
      clause: Clause,
    }

    impl std::cmp::Eq for Element {}

    impl std::cmp::PartialOrd for Element {
      fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.prob.partial_cmp(&other.prob)
      }
    }

    impl std::cmp::Ord for Element {
      fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
      }
    }

    // Create a list of cnf clauses
    let cnf_clauses = f
      .iter()
      .map(|clause| {
        clause
          .literals
          .iter()
          .sorted_by(|x, y| {
            let prob_x = self.literal_probability(*x);
            let prob_y = self.literal_probability(*y);
            prob_x.partial_cmp(&prob_y).unwrap().reverse()
          })
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();

    let create_element = |indices: Vec<usize>| {
      // Collect the new set of literals from the `new_indices`. Note that there might not exist such a set
      // since the index can go over bound.
      //
      // We need to make sure that the literals are sorted by their variable id
      let mut literals: Vec<Literal> = (0..cnf_clauses.len())
        .map(|j| {
          if indices[j] < cnf_clauses[j].len() {
            Some(cnf_clauses[j][indices[j]].clone())
          } else {
            None
          }
        })
        .collect::<Option<Vec<_>>>()?;
      literals.sort_by_key(|l| l.fact_id() * 2 + if l.sign() { 0 } else { 1 });
      literals.dedup();

      // Get the result
      let clause = Clause::new(literals);
      let prob = self.conjunction_probability(&clause);
      Some(Element { prob, indices, clause })
    };

    // A closure for getting the next elements of an element
    let next_elements = |indices: Vec<usize>| {
      // There will be `k` next elements if there are `k` clauses
      (0..indices.len()).filter_map(move |i| {
        // The `i`-th element will have an indices same as the current with only the `i`-th index incremented by 1.
        let new_indices = indices
          .iter()
          .enumerate()
          .map(|(j, x)| if i == j { *x + 1 } else { *x })
          .collect::<Vec<_>>();

        // Create the element using the generated indices
        create_element(new_indices)
      })
    };

    // Check corner cases
    if cnf_clauses.is_empty() {
      // If there is no clause, this is a "true"
      CNFDNFFormula::dnf_one().into()
    } else if cnf_clauses.iter().any(|c| c.is_empty()) {
      // If there is a clause with no literal, this is a "false"
      CNFDNFFormula::dnf_zero().into()
    } else {
      // Otherwise, perform full cartesian product
      let mut heap = BinaryHeap::<Element>::new();
      let mut visited = HashSet::<Vec<usize>>::new();
      let mut result_clauses = Vec::new();

      // Get the first element
      let first_indices = vec![0; cnf_clauses.len()];
      let first_elem = create_element(first_indices).unwrap();

      // Push the element into the heap
      heap.push(first_elem);

      // Enter the main loop
      while result_clauses.len() < k {
        if let Some(elem) = heap.pop() {
          if !visited.contains(&elem.indices) {
            visited.insert(elem.indices.clone());

            // Check validity of the clause; if valid, put it in the result
            if elem.clause.is_valid() && !self.has_disjunction_conflict(&elem.clause.pos_fact_ids()) {
              result_clauses.push(elem.clause);
            }

            // Insert the successors into the heap
            for elem in next_elements(elem.indices) {
              heap.push(elem);
            }
          } else {
            // Skip this clause
          }
        } else {
          break;
        }
      }

      // Return the result
      CNFDNFFormula::dnf(result_clauses).into()
    }
  }

  fn dnf2cnf_k(&self, f: &Vec<Clause>, k: usize) -> CNFDNFFormula {
    #[derive(PartialEq)]
    struct Element {
      prob: f64,
      indices: Vec<usize>,
      clause: Clause,
    }

    impl std::cmp::Eq for Element {}

    impl std::cmp::PartialOrd for Element {
      fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.prob.partial_cmp(&other.prob).map(std::cmp::Ordering::reverse)
      }
    }

    impl std::cmp::Ord for Element {
      fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
      }
    }

    // Create a list of dnf clauses
    let dnf_clauses = f
      .iter()
      .map(|clause| {
        clause
          .literals
          .iter()
          .sorted_by(|x, y| {
            let prob_x = self.literal_probability(*x);
            let prob_y = self.literal_probability(*y);
            prob_x.partial_cmp(&prob_y).unwrap()
          })
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();

    let create_element = |indices: Vec<usize>| {
      // Collect the new set of literals from the `new_indices`. Note that there might not exist such a set
      // since the index can go over bound.
      //
      // We need to make sure that the literals are sorted by their variable id
      let mut literals: Vec<Literal> = (0..dnf_clauses.len())
        .map(|j| {
          if indices[j] < dnf_clauses[j].len() {
            Some(dnf_clauses[j][indices[j]].clone())
          } else {
            None
          }
        })
        .collect::<Option<Vec<_>>>()?;
      literals.sort_by_key(|l| l.fact_id() * 2 + if l.sign() { 0 } else { 1 });
      literals.dedup();

      // Get the result
      let clause = Clause::new(literals);
      let prob = self.disjunction_probability(&clause);
      Some(Element { prob, indices, clause })
    };

    // A closure for getting the next elements of an element
    let next_elements = |indices: Vec<usize>| {
      // There will be `k` next elements if there are `k` clauses
      (0..indices.len()).filter_map(move |i| {
        // The `i`-th element will have an indices same as the current with only the `i`-th index incremented by 1.
        let new_indices = indices
          .iter()
          .enumerate()
          .map(|(j, x)| if i == j { *x + 1 } else { *x })
          .collect::<Vec<_>>();

        // Create the element using the generated indices
        create_element(new_indices)
      })
    };

    // Check corner cases
    if dnf_clauses.is_empty() {
      // If there is no clause, this is a "false"
      CNFDNFFormula::cnf_zero().into()
    } else if dnf_clauses.iter().any(|c| c.is_empty()) {
      // If there is a clause with no literal, this is a "true"
      CNFDNFFormula::cnf_one().into()
    } else {
      // Otherwise, perform full cartesian product
      let mut heap = BinaryHeap::<Element>::new();
      let mut visited = HashSet::<Vec<usize>>::new();
      let mut result_clauses = Vec::new();

      // Get the first element
      let first_indices = vec![0; dnf_clauses.len()];
      let first_elem = create_element(first_indices).unwrap();

      // Push the element into the heap
      heap.push(first_elem);

      // Enter the main loop
      while result_clauses.len() < k {
        if let Some(elem) = heap.pop() {
          if !visited.contains(&elem.indices) {
            visited.insert(elem.indices.clone());

            // Check validity of the clause; if valid, put it in the result
            if elem.clause.is_valid() {
              result_clauses.push(elem.clause);
            }

            // Insert the successors into the heap
            for elem in next_elements(elem.indices) {
              heap.push(elem);
            }
          } else {
            // Skip this clause
          }
        } else {
          break;
        }
      }

      // Return the result
      CNFDNFFormula::cnf(result_clauses).into()
    }
  }

  fn top_bottom_k_add(&self, t1: &CNFDNFFormula, t2: &CNFDNFFormula, k: usize) -> CNFDNFFormula {
    use FormulaKind::*;
    match (&t1.kind, &t2.kind) {
      (CNF, CNF) => self.mult_formula_bottom_k(&t1.clauses, &t2.clauses, k),
      (DNF, DNF) => self.add_formula_top_k(&t1.clauses, &t2.clauses, k),
      (CNF, DNF) => self.add_formula_top_k(&self.cnf2dnf_k(&t1.clauses, k).clauses, &t2.clauses, k),
      (DNF, CNF) => self.add_formula_top_k(&t1.clauses, &self.cnf2dnf_k(&t2.clauses, k).clauses, k),
    }
  }

  fn top_bottom_k_mult(&self, t1: &CNFDNFFormula, t2: &CNFDNFFormula, k: usize) -> CNFDNFFormula {
    use FormulaKind::*;
    match (&t1.kind, &t2.kind) {
      (CNF, CNF) => self.add_formula_bottom_k(&t1.clauses, &t2.clauses, k),
      (DNF, DNF) => self.mult_formula_top_k(&t1.clauses, &t2.clauses, k),
      (CNF, DNF) => self.add_formula_bottom_k(&t1.clauses, &self.dnf2cnf_k(&t2.clauses, k).clauses, k),
      (DNF, CNF) => self.add_formula_bottom_k(&self.dnf2cnf_k(&t1.clauses, k).clauses, &t2.clauses, k),
    }
  }

  fn base_negate(&self, t: &CNFDNFFormula) -> CNFDNFFormula {
    t.negate()
  }

  fn top_bottom_k_tag_of_chosen_set<'a, I>(&self, all: I, chosen_ids: &Vec<usize>, k: usize) -> CNFDNFFormula
  where
    I: Iterator<Item = &'a CNFDNFFormula>,
  {
    all
      .enumerate()
      .map(|(id, f)| {
        if chosen_ids.contains(&id) {
          f.clone()
        } else {
          self.base_negate(f)
        }
      })
      .fold(CNFDNFFormula::dnf_one(), |a, b| self.top_bottom_k_mult(&a, &b, k))
  }
}

pub struct BasicCNFDNFClausesContext {
  pub probabilities: Vec<f64>,
}

impl BasicCNFDNFClausesContext {
  pub fn new() -> Self {
    Self { probabilities: vec![] }
  }
}

impl CNFDNFContextTrait for BasicCNFDNFClausesContext {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.probabilities[*id]
  }

  fn has_disjunction_conflict(&self, _: &BTreeSet<usize>) -> bool {
    false
  }
}
