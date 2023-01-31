use std::collections::*;

use super::*;
use crate::compiler::ram;
use crate::compiler::CompileOptions;

impl Program {
  /// Apply basic optimizations on the back program.
  ///
  /// Such optimizations include:
  /// - equality propagation: `R(a, b), a == b` would become `R(a, a)`
  /// - constant folding/constant propagation: `R(a, b), a == 3` would become `R(3, b)`
  /// - true literals extraction: `a == a` would become `true`
  /// - removal of rules containing `false` literal
  /// - convert empty rule to a fact
  pub fn apply_optimizations(&mut self, opt: &CompileOptions) -> Result<(), BackCompileError> {
    // Perform rule level optimizations
    for rule in &mut self.rules {
      // First propagate equality
      optimizations::propagate_equality(rule);

      // Enter the loop of constant folding/propagation
      loop {
        let cloned = rule.clone();
        optimizations::constant_fold(rule, &self.function_registry);
        optimizations::constant_prop(rule);
        if &cloned == rule {
          break;
        }
      }

      // Remove the true literals in the rules
      optimizations::remove_true_literals(rule);
    }

    // Filter out the rules with a False inside of it
    optimizations::remove_false_rules(&mut self.rules);

    // Demand Transformation
    if !opt.do_not_demand_transform {
      self.demand_transform()?;
    }

    // Turn empty rules into facts
    optimizations::empty_rule_to_fact(&mut self.rules, &mut self.facts);

    // Return
    Ok(())
  }

  /// Remove unused relation according to the dependency graph.
  /// All the relations that are isolated from the queries will be removed, along
  /// with their rules
  pub fn remove_unused_relations(&mut self, dep_graph: &mut DependencyGraph) {
    // First collect all the relations to remove
    let to_remove = dep_graph.unused_relations(&self.output_relations());

    // Remove relations
    self.relations.retain(|r| !to_remove.contains(&r.predicate));

    // Remove rules
    self.rules.retain(|r| !to_remove.contains(r.head_predicate()));

    // Remove facts
    self.facts.retain(|f| !to_remove.contains(&f.predicate));

    // Update dep_graph by re-computing it
    *dep_graph = self.dependency_graph();
  }

  pub fn to_ram_program(&mut self, opt: &CompileOptions) -> Result<ram::Program, BackCompileError> {
    // Compute the dependency
    let mut dep_graph = self.dependency_graph();

    // Remove unused relations and strata; this process will modify the dependency graph
    if !opt.output_all && !opt.do_not_remove_unused_relations {
      self.remove_unused_relations(&mut dep_graph);
    }

    // Construct strata from the dependency graph
    dep_graph.compute_scc();
    let strata = dep_graph.stratify().map_err(BackCompileError::from)?;

    // For each strata, generate a query plan
    let mut ram_strata = self.strata_to_ram_strata(strata);

    // If output all, modify ram strata
    if opt.output_all {
      ram_strata.iter_mut().for_each(|s| s.output_all());
    }

    // Cache relation to stratum map
    let relation_to_stratum = ram_strata
      .iter()
      .enumerate()
      .flat_map(|(i, stratum)| {
        stratum
          .relations
          .iter()
          .map(move |(relation, _)| (relation.clone(), i.clone()))
      })
      .collect::<HashMap<_, _>>();

    // Create
    Ok(ram::Program {
      strata: ram_strata,
      function_registry: self.function_registry.clone(),
      relation_to_stratum,
    })
  }

  pub fn output_relations(&self) -> HashSet<String> {
    self
      .outputs
      .iter()
      .filter_map(|(n, o)| if o.is_not_hidden() { Some(n.clone()) } else { None })
      .collect()
  }
}
