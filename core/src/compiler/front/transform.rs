// use std::collections::*;
// use petgraph::graph::{Graph, NodeIndex};

// use crate::common::foreign_function::*;
// use crate::common::foreign_predicate::*;

use super::transformations::*;
use super::*;

/// A transformation pass
///
/// The lifetime constraint `'a` is based on the existing analysis results
pub trait Transformation<'a> {
  /// The dependent transformation passes
  fn dependencies(&self) -> Vec<&'static str> {
    vec![]
  }

  /// After running this pass, what transformation is invalidated and need to be rerun
  fn invalidates(&self) -> Vec<&'static str> {
    vec![]
  }

  /// After running this pass, does an item need to be removed
  #[allow(unused)]
  fn post_walking_removes_item(&self, item: &Item) -> bool {
    false
  }

  /// After running this pass, is there newly generated items
  fn post_walking_generated_items(&mut self) -> Vec<Item> {
    vec![]
  }
}

pub trait TransformationName {
  fn name() -> &'static str {
    std::any::type_name::<Self>()
  }
}

impl<'a, T> TransformationName for T where T: Transformation<'a> {}

pub fn apply_transformations(ast: &mut Vec<Item>, analysis: &mut Analysis) {
  // let transf_passes = Transformations::std(analysis);
  // transf_passes.run(ast);
  let mut transform_adt = TransformAlgebraicDataType::new(&mut analysis.adt_analysis);
  let mut transform_const_var_to_const = TransformConstVarToConst::new(&analysis.constant_decl_analysis);
  let mut transform_atomic_query = TransformAtomicQuery::new();
  let mut transform_conjunctive_head = TransformConjunctiveHead::new();
  let mut desugar_reduce_rule = DesugarReduceRule::new(&analysis.type_inference.foreign_aggregate_type_registry);
  let mut transform_tagged_rule = TransformTaggedRule::new(
    /* &mut analysis.type_inference, */ &mut analysis.tagged_rule_analysis,
  );
  let mut transform_non_const_fact = TransformNonConstantFactToRule;
  let mut desugar_arg_type_adornment = DesugarArgTypeAdornment::new();
  let mut desugar_destruct = DesugarDestruct::new();
  let mut desugar_case_is = DesugarCaseIs::new();
  let mut desugar_forall_exists = DesugarForallExists::new();
  let mut desugar_range = DesugarRange::new();
  let mut forall_to_not_exists = TransformForall;
  let mut implies_to_disjunction = TransformImplies;
  let mut visitors = (
    &mut transform_adt,
    &mut transform_atomic_query,
    &mut transform_conjunctive_head,
    &mut transform_const_var_to_const,
    &mut transform_tagged_rule,
    &mut transform_non_const_fact,
    &mut desugar_arg_type_adornment,
    &mut desugar_destruct,
    &mut desugar_case_is,
    &mut desugar_forall_exists,
    &mut desugar_range,
    &mut desugar_reduce_rule,
    &mut forall_to_not_exists, // Note: forall needs to go before implies transformation
    &mut implies_to_disjunction,
  );
  ast.walk_mut(&mut visitors);

  // Post-transformation; remove items
  ast.retain(|item| transform_conjunctive_head.retain(item) && transform_adt.retain(item));

  // Post-transformation; annotate node ids afterwards
  let mut new_items = vec![];
  new_items.extend(transform_adt.generate_items());
  new_items.extend(transform_const_var_to_const.generate_items());
  new_items.extend(transform_atomic_query.drain_items());
  new_items.extend(transform_conjunctive_head.generate_items());

  // Some of the transformations need to be applied to new items as well
  let mut transform_const_var_to_const_2 = TransformConstVarToConst2::new(&analysis.constant_decl_analysis);
  new_items.walk_mut(&mut transform_const_var_to_const_2);

  // Extend the ast to incorporate these new items
  ast.extend(new_items);
}

// pub struct DynTransformation<'a> {
//   transf: Box<dyn Transformation<'a> + 'a>
// }

// impl<'a> Transformation<'a> for DynTransformation<'a> {
//   fn dependencies(&self) -> Vec<&'static str> {
//     self.transf.dependencies()
//   }

//   fn invalidates(&self) -> Vec<&'static str> {
//     self.transf.invalidates()
//   }

//   fn post_walking_removes_item(&self, item: &Item) -> bool {
//     self.transf.post_walking_removes_item(item)
//   }

//   /// After running this pass, is there newly generated items
//   fn post_walking_generated_items(&mut self) -> Vec<Item> {
//     self.transf.post_walking_generated_items()
//   }
// }

// impl<'a, V> NodeVisitor<V> for DynTransformation<'a> {
//   fn visit(&mut self, node: &V) {
//     (&mut *self.transf).visit(node)
//   }

//   fn visit_mut(&mut self, node: &mut V) {
//     (&mut *self.transf).visit_mut(node)
//   }
// }

// /// A manager of all the transformation passes
// pub struct Transformations<'a> {
//   transformations: Graph<DynTransformation<'a>, ()>,
//   transformation_ids: HashMap<&'static str, NodeIndex>,
// }

// impl<'a> Transformations<'a> {
//   pub fn empty() -> Self {
//     Self {
//       transformations: Graph::new(),
//       transformation_ids: HashMap::new(),
//     }
//   }

//   pub fn std(analysis: &'a mut Analysis) -> Self {
//     let mut passes = Self::empty();

//     passes.add(TransformAlgebraicDataType::new(&mut analysis.adt_analysis));
//     passes.add(TransformConstVarToConst::new(&analysis.constant_decl_analysis));
//     passes.add(TransformAtomicQuery::new());
//     passes.add(TransformConjunctiveHead::new());
//     passes.add(TransformTaggedRule::new());
//     passes.add(TransformNonConstantFactToRule);
//     passes.add(DesugarArgTypeAdornment::new());
//     passes.add(DesugarCaseIs::new());
//     passes.add(DesugarForallExists::new());
//     passes.add(DesugarRange::new());
//     passes.add(DesugarReduceRule::new(&analysis.type_inference.foreign_aggregate_type_registry));
//     passes.add(TransformForall);
//     passes.add(TransformImplies);

//     passes.add(TransformConstVarToConst2::new(&analysis.constant_decl_analysis));
//     // passes.add(DesugarDestruct::new());

//     passes

//   }

//   pub fn add<T: Transformation<'a> + 'a>(&mut self, transformation: T) {
//     // Get the name and dependencies
//     let name = T::name();
//     let dependencies = transformation.dependencies();

//     // Add the transformation into the graph and the id map
//     let id = self.transformations.add_node(DynTransformation { transf: Box::new(transformation) });
//     self.transformation_ids.insert(name, id.clone());

//     // Add the dependency edges
//     for dep_name in dependencies {
//       let dep_id = self.transformation_ids.get(dep_name).expect(&format!("When adding front-compile pass `{name}`, dependent pass `{dep_name}` does not exist"));
//       self.transformations.add_edge(dep_id.clone(), id, ());
//     }
//   }

//   fn stratumize_transformations(&self) -> Vec<Vec<NodeIndex>> {
//     use petgraph::visit::EdgeRef;
//     let sorted_nodes = petgraph::algo::toposort(&self.transformations, None).expect("Cycle found in front-compile passes. Aborting");
//     let mut stratums: Vec<Vec<NodeIndex>> = Vec::new();
//     let mut node_to_stratum_map: HashMap<NodeIndex, usize> = HashMap::new();
//     for node in sorted_nodes {
//       // Compute what stratum to put the pass in
//       let to_put_stratum = self
//         .transformations
//         .edges_directed(node, petgraph::Direction::Incoming)
//         .map(|edge| {
//           let source_stratum_id = node_to_stratum_map.get(&edge.source()).expect("Should contain edge.source");
//           source_stratum_id + 1
//         })
//         .max()
//         .unwrap_or(0);

//       // Put the pass into the stratum
//       if to_put_stratum >= stratums.len() {
//         stratums.push(vec![node]);
//       } else {
//         stratums[to_put_stratum].push(node);
//       }

//       // Record the information in `node_to_stratum_map`
//       node_to_stratum_map.insert(node, to_put_stratum);
//     }

//     stratums
//   }

//   /// Run all the transformations
//   pub fn run(mut self, ast: &mut Vec<Item>) {
//     // Stratumize transformations so that transformations can be ran in as minimum iterations as possible
//     let stratums = self.stratumize_transformations();

//     // Construct a node index to transformation mapping; we are doing this because it is not possible to directly
//     // access this information through graph API
//     let mut pass_id_to_pass_map = HashMap::new();
//     for pass_id in self.transformations.node_indices().rev() {
//       pass_id_to_pass_map.insert(pass_id, self.transformations.remove_node(pass_id).expect("Should be expected"));
//     }

//     // Iterate through stratums
//     for stratum in stratums {
//       // Construct all the passes in this stratum
//       let mut to_run_passes = vec![];
//       for pass_id in stratum {
//         let pass = pass_id_to_pass_map.remove(&pass_id).expect("Should be expected");
//         to_run_passes.push(pass);
//       }

//       // Walk the AST with the set of passes
//       ast.walk_mut(&mut to_run_passes);

//       // Check if any item needs to be removed
//       ast.retain(|item| !to_run_passes.iter().any(|pass| pass.post_walking_removes_item(item)));

//       // Check if any item needs to be added
//       for pass in &mut to_run_passes {
//         ast.extend(pass.post_walking_generated_items());
//       }
//     }
//   }
// }

// pub fn apply_transformations(ast: &mut Vec<Item>, analysis: &mut Analysis) {
//   let transf_passes = Transformations::std(analysis);
//   transf_passes.run(ast);
// }
