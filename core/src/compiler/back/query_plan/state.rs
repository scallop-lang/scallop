use std::collections::*;

use itertools::Itertools;

use super::*;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct State {
  pub atom_relations: HashMap<usize, String>,
  pub visited_atoms: Vec<usize>,
  pub arcs: Vec<Arc>,
}

impl std::cmp::PartialOrd for State {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.aggregated_weight().partial_cmp(&other.aggregated_weight())
  }
}

impl std::cmp::Ord for State {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.aggregated_weight().cmp(&other.aggregated_weight())
  }
}

impl State {
  pub fn new(atom_relations: HashMap<usize, String>) -> Self {
    Self {
      atom_relations,
      visited_atoms: vec![],
      arcs: vec![],
    }
  }

  pub fn aggregated_weight(&self) -> i32 {
    self.arcs.iter().map(|a| a.weight()).sum()
  }

  pub fn next_states(&self, ctx: &QueryPlanContext, beam_size: usize) -> Vec<Self> {
    if self.visited_atoms.len() >= ctx.pos_atoms.len() {
      vec![]
    } else {
      let mut next_states: Vec<Self> = vec![];
      for (id, atom) in ctx
        .pos_atoms
        .iter()
        .enumerate()
        .filter(|(i, _)| !self.visited_atoms.contains(i))
      {
        for set in self.visited_atoms.iter().powerset() {
          let all_bounded_args = ctx.bounded_args_from_pos_atoms_set(&set, false);
          let bounded_vars = atom
            .variable_args()
            .filter(|a| all_bounded_args.contains(a))
            .cloned()
            .collect();
          let is_edb = !ctx.stratum.predicates.contains(&atom.predicate);
          let arc = Arc {
            left: set.iter().map(|i| **i).collect::<Vec<_>>(),
            right: id,
            left_relations: set.iter().map(|id| self.atom_relations[id].clone()).collect(),
            bounded_vars,
            is_edb,
          };
          next_states.push(State {
            atom_relations: self.atom_relations.clone(),
            visited_atoms: vec![self.visited_atoms.clone(), vec![id]].concat(),
            arcs: vec![self.arcs.clone(), vec![arc]].concat(),
          });
        }
      }
      next_states.sort_by_key(|s| -s.aggregated_weight());
      next_states[0..beam_size.min(next_states.len())].to_vec()
    }
  }

  pub fn bounded_all(&self, ctx: &QueryPlanContext) -> bool {
    self.visited_atoms.len() == ctx.pos_atoms.len()
  }
}
