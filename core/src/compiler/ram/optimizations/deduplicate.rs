use std::collections::*;

use super::*;

pub fn deduplicate(ram: &mut Program) -> bool {
  let mut has_update = false;
  for stratum in &mut ram.strata {
    has_update |= deduplicate_within_stratum(stratum);
  }
  has_update
}

pub fn deduplicate_within_stratum(stratum: &mut Stratum) -> bool {
  let mut has_update = false;
  let mut equivalent_sets = EquivalentRelations::new();

  // 1. Find temporary relations with the same update
  for (i, update_i) in stratum.updates.iter().enumerate() {
    for (j, update_j) in stratum.updates.iter().enumerate() {

      // Check if the dataflow of the update looks exactly the same
      if i != j && update_i.dataflow == update_j.dataflow {
        let head_i = &update_i.target;
        let head_j = &update_j.target;
        if head_i == head_j {
          // They are both the same;
        } else {
          let i_is_temp = head_i.starts_with("#temp#");
          let j_is_temp = head_j.starts_with("#temp#");
          if i_is_temp && j_is_temp {
            equivalent_sets.add_equivalence(head_i, head_j);
          } else if i_is_temp {
            // Do nothing
          } else if j_is_temp {
            // Do nothing
          } else {
            // Do nothing
          }
        }
      }
    }
  }

  // 2. Perform the substitution
  for set in equivalent_sets.sets {
    if set.len() < 2 {
      continue;
    }

    // 2.1. Obtaining a pivot
    let mut iterator = set.into_iter();
    let pivot = iterator.next().unwrap();

    // 2.2. Substitute all relations in the equivalent sets to pivot
    while let Some(to_substitute) = iterator.next() {
      // 2.2.1. Remove the to_substitue relation
      stratum.relations.remove(&to_substitute);
      has_update |= true;

      stratum.updates.retain_mut(|update| {
        if update.target == to_substitute {
          // 2.2.2. Remove all the updates with the to_substitute head
          false
        } else {
          // 2.2.3. Substitute the body dataflow
          has_update |= update.dataflow.substitute_relation(&to_substitute, &pivot);
          true
        }
      })
    }
  }

  has_update
}

struct EquivalentRelations {
  eq_map: HashMap<String, usize>,
  sets: Vec<HashSet<String>>,
}

impl EquivalentRelations {
  pub fn new() -> Self {
    Self {
      eq_map: HashMap::new(),
      sets: Vec::new(),
    }
  }

  pub fn add_equivalence(&mut self, r1: &str, r2: &str) {
    let contains_r1 = self.eq_map.get(r1).cloned();
    let contains_r2 = self.eq_map.get(r2).cloned();
    if contains_r1.is_some() && contains_r2.is_some() {
      if contains_r1 == contains_r2 {
        // They are already in the equivalent set. Do nothing
      } else {
        let eq_set_id_1 = contains_r1.unwrap();
        let eq_set_id_2 = contains_r2.unwrap();

        // We drain the content of the second set
        let relations = self.sets[eq_set_id_2].drain().collect::<Vec<_>>();

        // We rewire their equivalence set to the first set
        for relation in &relations {
          self.eq_map.insert(relation.clone(), eq_set_id_1.clone());
        }

        // We extend the first equivalence set with everything in the second
        self.sets[eq_set_id_1].extend(relations.into_iter());
      }
    } else if contains_r1.is_some() {

      // Insert the second into the equivalence set
      let eq_set_id = contains_r1.unwrap();
      self.eq_map.insert(r2.to_string(), eq_set_id);
      self.sets[eq_set_id].insert(r2.to_string());

    } else if contains_r2.is_some() {

      // Insert the first into the equivalence set
      let eq_set_id = contains_r2.unwrap();
      self.eq_map.insert(r1.to_string(), eq_set_id);
      self.sets[eq_set_id].insert(r1.to_string());

    } else {

      // Create a new equivalence set
      let eq_set_id = self.sets.len();
      let eq_set = vec![r1.to_string(), r2.to_string()].into_iter().collect();
      self.sets.push(eq_set);
      self.eq_map.insert(r1.to_string(), eq_set_id);
      self.eq_map.insert(r2.to_string(), eq_set_id);
    }
  }
}
