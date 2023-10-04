use std::collections::*;
pub use std::iter::FromIterator;

#[derive(Clone, Debug)]
pub struct Disjunction {
  pub facts: BTreeSet<usize>,
}

impl Disjunction {
  pub fn add_fact_id(&mut self, fact_id: usize) {
    self.facts.insert(fact_id);
  }

  /// Note: Assumes that the #facts > 2
  pub fn has_conflict(&self, facts: &BTreeSet<usize>) -> bool {
    // Short cut 1
    if self.facts.len() < 2 {
      return false;
    }

    // Short cut 2
    let j_last = self.facts.last().unwrap(); // Note: disj.len >= 2
    let j_first = self.facts.first().unwrap(); // Note: disj.len >= 2
    let f_last = facts.last().unwrap(); // Note: facts.len >= 2
    let f_first = facts.first().unwrap(); // Note: facts.len >= 2
    if j_last < f_first || f_last < j_first {
      return false;
    }

    // Start iteration
    let mut j_iter = self.facts.iter();
    let mut f_iter = facts.iter();
    let mut has_same = false;
    let mut j_curr = j_iter.next();
    let mut f_curr = f_iter.next();
    loop {
      match (j_curr, f_curr) {
        (Some(j_elem), Some(f_elem)) => {
          if j_elem == f_elem {
            if has_same {
              return true;
            } else {
              has_same = true;
            }
            j_curr = j_iter.next();
            f_curr = f_iter.next();
          } else if j_elem < f_elem {
            j_curr = j_iter.next();
          } else {
            f_curr = f_iter.next();
          }
        }
        _ => break,
      }
    }
    false
  }
}

impl FromIterator<usize> for Disjunction {
  fn from_iter<T>(iter: T) -> Self
  where
    T: IntoIterator<Item = usize>,
  {
    Self {
      facts: iter.into_iter().collect(),
    }
  }
}

#[derive(Clone, Debug, Default)]
pub struct Disjunctions {
  pub id_map: HashMap<usize, usize>,
  pub disjunctions: Vec<Disjunction>,
}

impl Disjunctions {
  pub fn new() -> Self {
    Self {
      id_map: HashMap::new(),
      disjunctions: Vec::new(),
    }
  }

  pub fn is_empty(&self) -> bool {
    self.disjunctions.is_empty()
  }

  pub fn len(&self) -> usize {
    self.disjunctions.len()
  }

  pub fn has_conflict(&self, facts: &BTreeSet<usize>) -> bool {
    // Short hand
    if facts.len() < 2 {
      return false;
    }

    // Check conflict for each disjunction
    for disj in &self.disjunctions {
      if disj.has_conflict(facts) {
        return true;
      }
    }
    false
  }

  pub fn add_disjunction(&mut self, disj_id: usize, fact_id: usize) {
    if let Some(internal_disj_id) = self.id_map.get(&disj_id) {
      self.disjunctions[*internal_disj_id].add_fact_id(fact_id)
    } else {
      let new_disj = Disjunction::from_iter(std::iter::once(fact_id));
      let internal_disj_id = self.disjunctions.len();
      self.disjunctions.push(new_disj);
      self.id_map.insert(disj_id, internal_disj_id);
    }
  }
}
