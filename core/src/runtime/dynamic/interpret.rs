use std::collections::*;

use super::super::provenance::*;
use super::*;
use crate::common::output_option::OutputOption;
use crate::common::predicate_set::PredicateSet;
use crate::compiler;
use crate::runtime::monitor::Monitor;

pub type Output<T> = BTreeMap<String, DynamicOutputCollection<T>>;

pub type InterpretResult<T> = Result<Output<T>, RuntimeError>;

#[derive(Default)]
pub struct InterpretOptions {
  /// The set of predicates that we expect to return from interpretation
  pub return_relations: PredicateSet,

  /// The set of predicates that we expect to be printed from interpretation
  pub print_relations: PredicateSet,
}

pub fn interpret<C>(ram: compiler::ram::Program, ctx: &mut C) -> InterpretResult<C::Tag>
where
  C: ProvenanceContext,
{
  let opt = InterpretOptions::default();
  interpret_with_options(ram, ctx, &opt)
}

pub fn interpret_with_options<C>(
  ram: compiler::ram::Program,
  ctx: &mut C,
  opt: &InterpretOptions,
) -> InterpretResult<C::Tag>
where
  C: ProvenanceContext,
{
  let (stratum_inputs, stratum_outputs) = stratum_inputs_outputs(&ram);

  // Store the results
  let mut results: Vec<HashMap<String, DynamicCollection<C::Tag>>> = vec![];

  // Iterate through stratum
  for (i, stratum) in ram.strata.iter().enumerate() {
    let mut iter = DynamicIteration::<C::Tag>::new();

    // Create dynamic relations
    let mut dyn_relas = HashSet::new();
    for (predicate, relation) in &stratum.relations {
      dyn_relas.insert(predicate.clone());
      iter.create_dynamic_relation(predicate);

      // Load input file
      if let Some(input_file) = &relation.input_file {
        let tuples = io::load(input_file, &relation.tuple_type)?;
        iter
          .get_dynamic_relation_unsafe(predicate)
          .insert_dynamically_tagged(ctx, tuples);
      }

      // Add facts
      if !relation.facts.is_empty() {
        iter.get_dynamic_relation_unsafe(predicate).insert_dynamically_tagged(
          ctx,
          relation
            .facts
            .iter()
            .map(|f| (f.tag.clone(), f.tuple.clone()))
            .collect(),
        );
      }

      // Add disjunctive facts
      if !relation.disjunctive_facts.is_empty() {
        for disjunction in &relation.disjunctive_facts {
          iter.get_dynamic_relation_unsafe(predicate).insert_dynamically_tagged(
            ctx,
            disjunction.iter().map(|f| (f.tag.clone(), f.tuple.clone())).collect(),
          );
        }
      }
    }

    // Add input collections
    if let Some(inputs) = stratum_inputs.get(&i) {
      for (dep, dep_stratum) in inputs {
        iter.add_input_dynamic_collection(dep, &results[*dep_stratum][dep]);
      }
    }

    // Add output relations
    if let Some(outputs) = stratum_outputs.get(&i) {
      for relation in outputs {
        iter.add_output_relation(relation);
      }
    }

    // Add updates
    for update in &stratum.updates {
      iter.add_update(update.clone());
    }

    // Run!
    let result = iter.run(ctx);
    results.push(result);
  }

  // Output
  let mut final_result = BTreeMap::new();
  for (i, stratum_result) in results.into_iter().enumerate() {
    for (r, c) in stratum_result.into_iter() {
      match &ram.strata[i].relations[&r].output {
        OutputOption::Hidden => {}
        OutputOption::Default => {
          let rc = c.recover(ctx);
          if opt.print_relations.contains(&r) {
            println!("{}: {}", r, rc);
          }
          if opt.return_relations.contains(&r) {
            final_result.insert(r, rc);
          }
        }
        OutputOption::File(f) => {
          io::store(f, c.iter().map(|e| &e.tuple))?;
        }
      }
    }
  }
  Ok(final_result)
}

pub fn interpret_with_options_and_monitor<C, M>(
  ram: compiler::ram::Program,
  ctx: &mut C,
  opt: &InterpretOptions,
  monitor: &M,
) -> InterpretResult<C::Tag>
where
  C: ProvenanceContext,
  M: Monitor<C>,
{
  let (stratum_inputs, stratum_outputs) = stratum_inputs_outputs(&ram);

  // Store the results
  let mut results: Vec<HashMap<String, DynamicCollection<C::Tag>>> = vec![];

  // Iterate through stratum
  for (i, stratum) in ram.strata.iter().enumerate() {
    let mut iter = DynamicIteration::<C::Tag>::new();

    // Create dynamic relations
    let mut dyn_relas = HashSet::new();
    for (predicate, relation) in &stratum.relations {
      dyn_relas.insert(predicate.clone());
      iter.create_dynamic_relation(predicate);

      // Monitor loading of relation
      monitor.observe_loading_relation(predicate);

      // Load input file
      if let Some(input_file) = &relation.input_file {
        let tuples = io::load(input_file, &relation.tuple_type)?;
        iter
          .get_dynamic_relation_unsafe(predicate)
          .insert_dynamically_tagged_with_monitor(ctx, tuples, monitor);
      }

      // Add facts
      if !relation.facts.is_empty() {
        iter
          .get_dynamic_relation_unsafe(predicate)
          .insert_dynamically_tagged_with_monitor(
            ctx,
            relation
              .facts
              .iter()
              .map(|f| (f.tag.clone(), f.tuple.clone()))
              .collect(),
            monitor,
          );
      }

      // Add disjunctive facts
      if !relation.disjunctive_facts.is_empty() {
        for disjunction in &relation.disjunctive_facts {
          iter
            .get_dynamic_relation_unsafe(predicate)
            .insert_dynamically_tagged_with_monitor(
              ctx,
              disjunction.iter().map(|f| (f.tag.clone(), f.tuple.clone())).collect(),
              monitor,
            );
        }
      }
    }

    // Add input collections
    if let Some(inputs) = stratum_inputs.get(&i) {
      for (dep, dep_stratum) in inputs {
        iter.add_input_dynamic_collection(dep, &results[*dep_stratum][dep]);
      }
    }

    // Add output relations
    if let Some(outputs) = stratum_outputs.get(&i) {
      for relation in outputs {
        iter.add_output_relation(relation);
      }
    }

    // Add updates
    for update in &stratum.updates {
      iter.add_update(update.clone());
    }

    // Run!
    let result = iter.run(ctx);
    results.push(result);
  }

  // Output
  let mut final_result = BTreeMap::new();
  for (i, stratum_result) in results.into_iter().enumerate() {
    for (r, c) in stratum_result.into_iter() {
      match &ram.strata[i].relations[&r].output {
        OutputOption::Hidden => {}
        OutputOption::Default => {
          monitor.observe_recovering_relation(&r);
          let rc = c.recover_with_monitor(ctx, monitor);
          if opt.print_relations.contains(&r) {
            println!("{}: {}", r, rc);
          }
          if opt.return_relations.contains(&r) {
            final_result.insert(r, rc);
          }
        }
        OutputOption::File(_) => {
          unimplemented!("Cannot output into file for now")
        }
      }
    }
  }
  Ok(final_result)
}

type StratumInputs = HashMap<usize, HashSet<(String, usize)>>;

type StratumOutputs = HashMap<usize, HashSet<String>>;

fn stratum_inputs_outputs(ram: &compiler::ram::Program) -> (StratumInputs, StratumOutputs) {
  // Cache stratum input output
  let mut stratum_inputs = HashMap::<usize, HashSet<(String, usize)>>::new();
  let mut stratum_outputs = HashMap::<usize, HashSet<String>>::new();

  // Iterate through the strata
  for (i, stratum) in ram.strata.iter().enumerate() {
    // With dependency, add to input/output
    for dep in stratum.dependency() {
      let dep_stratum = ram.relation_to_stratum[&dep];
      stratum_outputs.entry(dep_stratum).or_default().insert(dep.clone());
      stratum_inputs.entry(i).or_default().insert((dep.clone(), dep_stratum));
    }

    // Regular user requested outputs
    for (predicate, relation) in &stratum.relations {
      if relation.output.is_not_hidden() {
        stratum_outputs.entry(i).or_default().insert(predicate.clone());
      }
    }
  }

  (stratum_inputs, stratum_outputs)
}
