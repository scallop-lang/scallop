use std::collections::*;

use super::*;
use crate::common::tuple::Tuple;
use crate::compiler::ram;
use crate::runtime::monitor::Monitor;
use crate::runtime::provenance::*;
use crate::utils::{PointerFamily, RcFamily};

pub struct DynamicExecutionContext<Prov: Provenance, P: PointerFamily = RcFamily> {
  pub program: ram::Program,
  pub facts: HashMap<String, Vec<(Option<InputTagOf<Prov>>, Tuple)>>,
  pub disjunctions: HashMap<String, Vec<Vec<usize>>>,
  pub results: HashMap<String, P::Pointer<DynamicCollection<Prov>>>,
}

impl<Prov: Provenance, P: PointerFamily> Clone for DynamicExecutionContext<Prov, P> {
  fn clone(&self) -> Self {
    Self {
      program: self.program.clone(),
      facts: self.facts.clone(),
      disjunctions: self.disjunctions.clone(),
      results: self.results.iter().map(|(r, c)| (r.clone(), P::clone_ptr(c))).collect(),
    }
  }
}

impl<Prov: Provenance, P: PointerFamily> DynamicExecutionContext<Prov, P> {
  pub fn new() -> Self {
    Self {
      program: ram::Program::new(),
      facts: HashMap::new(),
      disjunctions: HashMap::new(),
      results: HashMap::new(),
    }
  }

  pub fn execute(&mut self, program: ram::Program, ctx: &mut Prov) -> Result<(), RuntimeError> {
    self.execute_with_iter_limit(program, ctx, None)
  }

  pub fn execute_with_iter_limit(
    &mut self,
    program: ram::Program,
    ctx: &mut Prov,
    iter_limit: Option<usize>,
  ) -> Result<(), RuntimeError> {
    let mut curr_result = HashMap::new();
    std::mem::swap(&mut self.results, &mut curr_result);

    // Persistent relations
    let pers = self.program.persistent_relations(&program);
    curr_result.retain(|k, _| pers.contains(k));

    // Go through each stratum
    for stratum in &program.strata {
      let inputs = curr_result.iter().map(|(p, c)| (p.clone(), c)).collect();
      let result = self.execute_stratum(stratum, inputs, ctx, iter_limit)?;
      curr_result.extend(result.into_iter().map(|(p, c)| (p, P::new(c))));
    }

    // Store the result
    self.results = curr_result;

    // Update the program
    self.program = program;

    // Success!
    Ok(())
  }

  fn execute_stratum(
    &mut self,
    stratum: &ram::Stratum,
    inputs: HashMap<String, &P::Pointer<DynamicCollection<Prov>>>,
    ctx: &mut Prov,
    iter_limit: Option<usize>,
  ) -> Result<HashMap<String, DynamicCollection<Prov>>, RuntimeError> {
    let dyn_relas = stratum
      .relations
      .iter()
      .filter(|(r, _)| !inputs.contains_key(*r))
      .map(|(r, _)| r.clone())
      .collect::<HashSet<_>>();

    // Check if we need to compute anything new
    if dyn_relas.is_empty() {
      return Ok(HashMap::new());
    }

    // Otherwise, do computation
    let mut iter = DynamicIteration::<Prov>::new();

    // Add input collections
    for (rel, col) in inputs {
      iter.add_input_dynamic_collection(&rel, &*col);
    }

    // Create dynamic relations; all of them will be in the output
    for rela in &dyn_relas {
      iter.create_dynamic_relation(rela);
      iter.add_output_relation(rela);

      // Add facts
      let relation = &stratum.relations[rela];
      let facts = &relation.facts;
      if !facts.is_empty() {
        iter
          .get_dynamic_relation_unsafe(rela)
          .insert_dynamically_tagged(ctx, facts.iter().map(|f| (f.tag.clone(), f.tuple.clone())).collect());
      }

      // Load Inputs
      if let Some(input_file) = &relation.input_file {
        let inp = io::load(input_file, &relation.tuple_type).map_err(RuntimeError::IO)?;
        iter
          .get_dynamic_relation_unsafe(rela)
          .insert_dynamically_tagged(ctx, inp);
      }

      // Load external facts
      if let Some(facts) = self.facts.get(rela) {
        // Check if we need to process disjunction
        match self.disjunctions.get(rela) {
          Some(disjunctions) if !disjunctions.is_empty() => {
            // Process disjunctions if presented
            let mut all_indices = (0..facts.len()).collect::<HashSet<_>>();

            // Go through each disjunction and insert disjunction
            for disjunction in disjunctions {
              for id in disjunction {
                all_indices.remove(id);
              }
              let data = disjunction.iter().map(|i| facts[*i].clone()).collect();
              iter
                .get_dynamic_relation_unsafe(rela)
                .insert_annotated_disjunction(ctx, data);
            }

            // At the end, insert things that are not disjunctions
            if all_indices.len() > 0 {
              let other_facts = all_indices.into_iter().map(|i| facts[i].clone()).collect();
              iter.get_dynamic_relation_unsafe(rela).insert_tagged(ctx, other_facts);
            }
          }
          _ => {
            // If there is no disjunction
            iter.get_dynamic_relation_unsafe(rela).insert_tagged(ctx, facts.clone());
          }
        }
      }
    }

    // Add updates
    for update in &stratum.updates {
      if dyn_relas.contains(&update.target) {
        iter.add_update(update.clone());
      }
    }

    // Run!
    let result = iter.run_with_iter_limit(ctx, iter_limit);

    // Success!
    Ok(result)
  }

  pub fn execute_with_monitor<M>(&mut self, program: ram::Program, ctx: &mut Prov, m: &M) -> Result<(), RuntimeError>
  where
    M: Monitor<Prov>,
  {
    self.execute_with_iter_limit_and_monitor(program, ctx, None, m)
  }

  pub fn execute_with_iter_limit_and_monitor<M>(
    &mut self,
    program: ram::Program,
    ctx: &mut Prov,
    iter_limit: Option<usize>,
    m: &M,
  ) -> Result<(), RuntimeError>
  where
    M: Monitor<Prov>,
  {
    let mut curr_result = HashMap::new();
    std::mem::swap(&mut self.results, &mut curr_result);

    // Persistent relations
    let pers = self.program.persistent_relations(&program);
    curr_result.retain(|k, _| pers.contains(k));

    // Go through each stratum
    for stratum in &program.strata {
      let inputs = curr_result.iter().map(|(p, c)| (p.clone(), c)).collect();
      let result = self.execute_stratum_with_monitor(stratum, inputs, ctx, iter_limit, m)?;
      curr_result.extend(result.into_iter().map(|(p, c)| (p, P::new(c))));
    }

    // Store the result
    self.results = curr_result;

    // Update the program
    self.program = program;

    // Success!
    Ok(())
  }

  fn execute_stratum_with_monitor<M>(
    &mut self,
    stratum: &ram::Stratum,
    inputs: HashMap<String, &P::Pointer<DynamicCollection<Prov>>>,
    ctx: &mut Prov,
    iter_limit: Option<usize>,
    m: &M,
  ) -> Result<HashMap<String, DynamicCollection<Prov>>, RuntimeError>
  where
    M: Monitor<Prov>,
  {
    let dyn_relas = stratum
      .relations
      .iter()
      .filter(|(r, _)| !inputs.contains_key(*r))
      .map(|(r, _)| r.clone())
      .collect::<HashSet<_>>();

    // Check if we need to compute anything new
    if dyn_relas.is_empty() {
      return Ok(HashMap::new());
    }

    // Otherwise, do computation
    let mut iter = DynamicIteration::<Prov>::new();

    // Add input collections
    for (rel, col) in inputs {
      iter.add_input_dynamic_collection(&rel, &*col);
    }

    // Create dynamic relations; all of them will be in the output
    for rela in &dyn_relas {
      iter.create_dynamic_relation(rela);
      iter.add_output_relation(rela);

      // Monitor loading of relation
      m.observe_loading_relation(rela);

      // Add facts
      let relation = &stratum.relations[rela];
      let facts = &relation.facts;
      if !facts.is_empty() {
        let fs = facts.iter().map(|f| (f.tag.clone(), f.tuple.clone())).collect();
        iter
          .get_dynamic_relation_unsafe(rela)
          .insert_dynamically_tagged_with_monitor(ctx, fs, m);
      }

      // Load Inputs
      if let Some(input_file) = &relation.input_file {
        let inp = io::load(input_file, &relation.tuple_type).map_err(RuntimeError::IO)?;
        iter
          .get_dynamic_relation_unsafe(rela)
          .insert_dynamically_tagged_with_monitor(ctx, inp, m);
      }

      // Load external facts
      if let Some(facts) = self.facts.get(rela) {
        // Check if we need to process disjunction
        match self.disjunctions.get(rela) {
          Some(disjunctions) if !disjunctions.is_empty() => {
            // Process disjunctions if presented
            let mut all_indices = (0..facts.len()).collect::<HashSet<_>>();

            // Go through each disjunction and insert disjunction
            for disjunction in disjunctions {
              for id in disjunction {
                all_indices.remove(id);
              }
              let data = disjunction.iter().map(|i| facts[*i].clone()).collect();
              iter
                .get_dynamic_relation_unsafe(rela)
                .insert_annotated_disjunction_with_monitor(ctx, data, m);
            }

            // At the end, insert things that are not disjunctions
            if all_indices.len() > 0 {
              let other_facts = all_indices.into_iter().map(|i| facts[i].clone()).collect();
              iter
                .get_dynamic_relation_unsafe(rela)
                .insert_tagged_with_monitor(ctx, other_facts, m);
            }
          }
          _ => {
            // If there is no disjunction
            iter
              .get_dynamic_relation_unsafe(rela)
              .insert_tagged_with_monitor(ctx, facts.clone(), m);
          }
        }
      }
    }

    // Add updates
    for update in &stratum.updates {
      if dyn_relas.contains(&update.target) {
        iter.add_update(update.clone());
      }
    }

    // Run!
    let result = iter.run_with_iter_limit(ctx, iter_limit);

    // Success!
    Ok(result)
  }

  pub fn add_facts(&mut self, relation: &str, facts: Vec<(Option<InputTagOf<Prov>>, Tuple)>) {
    self.add_facts_with_disjunction(relation, facts, None);
  }

  pub fn add_facts_with_disjunction(
    &mut self,
    relation: &str,
    facts: Vec<(Option<InputTagOf<Prov>>, Tuple)>,
    disjunctions: Option<Vec<Vec<usize>>>,
  ) {
    // For incremental: when new fact is added, we need to recompute the relation
    if !facts.is_empty() {
      self.results.remove(relation);
    }

    // Cache number of facts
    let num_facts = facts.len();

    // Add facts back
    self.facts.entry(relation.to_string()).or_default().extend(facts);

    // Add disjunctions if presented
    if let Some(disjunctions) = disjunctions {
      let last_index = self.facts[relation].len();
      for disjunction in disjunctions {
        let remapped_disjunction = disjunction.into_iter().map(|i| last_index - num_facts + i).collect();
        self
          .disjunctions
          .entry(relation.to_string())
          .or_default()
          .push(remapped_disjunction);
      }
    }
  }

  pub fn internal_relation(&self, r: &str) -> Option<&DynamicCollection<Prov>> {
    self.results.get(r).map(|c| &**c)
  }

  pub fn internal_rc_relation(&self, r: &str) -> Option<P::Pointer<DynamicCollection<Prov>>> {
    self.results.get(r).map(|c| P::clone_ptr(c))
  }

  pub fn relation(&self, r: &str, ctx: &Prov) -> Option<DynamicOutputCollection<Prov>> {
    self.internal_relation(r).map(|c| c.clone().recover(ctx))
  }

  pub fn relation_with_monitor<M>(&self, r: &str, ctx: &Prov, m: &M) -> Option<DynamicOutputCollection<Prov>>
  where
    M: Monitor<Prov>,
  {
    self
      .internal_relation(r)
      .map(|c| c.clone().recover_with_monitor(ctx, m))
  }

  pub fn is_computed(&self, r: &str) -> bool {
    self.results.contains_key(r)
  }

  pub fn num_relations(&self) -> usize {
    self.program.relation_to_stratum.len()
  }

  pub fn relations(&self) -> Vec<String> {
    self
      .program
      .relation_to_stratum
      .iter()
      .map(|(n, _)| n.clone())
      .collect()
  }
}
