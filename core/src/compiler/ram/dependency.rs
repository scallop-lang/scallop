use petgraph::graph::{EdgeIndex, Graph, NodeIndex};
use std::collections::*;

use super::*;

pub type RamDependencyGraph<'a> = Graph<&'a Relation, BTreeSet<&'a Update>>;

pub type RamDependencySCCGraph<'a> = Graph<Vec<&'a Relation>, BTreeSet<&'a Update>>;

impl Program {
  pub fn dependency_graph<'a>(&'a self) -> RamDependencyGraph<'a> {
    let mut graph = RamDependencyGraph::new();

    // Iterate through all the relations and create their node ids
    let mut relation_node_ids = HashMap::new();
    for stratum in &self.strata {
      for (predicate, relation) in &stratum.relations {
        let relation_node_id = graph.add_node(relation);
        relation_node_ids.insert(predicate.clone(), relation_node_id.clone());
      }
    }

    // Iterate through all the updates and link relations
    let mut edge_ids = HashMap::<(NodeIndex, NodeIndex), EdgeIndex>::new();
    for stratum in &self.strata {
      for update in &stratum.updates {
        let target = relation_node_ids[&update.target];
        for dep_pred in update.dependency() {
          let source = relation_node_ids[&dep_pred];
          if let Some(edge_id) = edge_ids.get(&(source.clone(), target.clone())) {
            graph.edge_weight_mut(edge_id.clone()).unwrap().insert(update);
          } else {
            let edge_id = graph.add_edge(source, target, std::iter::once(update).collect::<BTreeSet<_>>());
            edge_ids.insert((source, target), edge_id);
          }
        }
      }
    }

    graph
  }

  pub fn scc<'a>(&'a self) -> RamDependencySCCGraph<'a> {
    let g = self.dependency_graph();
    petgraph::algo::condensation(g, true)
  }
}

impl Stratum {
  pub fn dependency(&self) -> HashSet<String> {
    self
      .updates
      .iter()
      .flat_map(Update::dependency)
      .filter(|i| !self.relations.contains_key(i))
      .collect()
  }
}

impl Update {
  pub fn dependency(&self) -> HashSet<String> {
    let mut preds = HashSet::new();
    self.dataflow.collect_dependency(&mut preds);
    preds
  }
}

impl Dataflow {
  fn collect_dependency(&self, preds: &mut HashSet<String>) {
    match self {
      Self::Unit(_) => {}
      Self::Relation(r) => {
        preds.insert(r.clone());
      }
      Self::Reduce(r) => {
        preds.insert(r.predicate.clone());
        if let ReduceGroupByType::Join(group_by_predicate) = &r.group_by {
          preds.insert(group_by_predicate.clone());
        }
      }
      Self::OverwriteOne(d) => {
        d.collect_dependency(preds);
      }
      Self::Find(d, _) => {
        d.collect_dependency(preds);
      }
      Self::Filter(d, _) => {
        d.collect_dependency(preds);
      }
      Self::Project(d, _) => {
        d.collect_dependency(preds);
      }
      Self::Difference(d1, d2) => {
        d1.collect_dependency(preds);
        d2.collect_dependency(preds);
      }
      Self::Antijoin(d1, d2) => {
        d1.collect_dependency(preds);
        d2.collect_dependency(preds);
      }
      Self::Product(d1, d2) => {
        d1.collect_dependency(preds);
        d2.collect_dependency(preds);
      }
      Self::Intersect(d1, d2) => {
        d1.collect_dependency(preds);
        d2.collect_dependency(preds);
      }
      Self::Join(d1, d2) => {
        d1.collect_dependency(preds);
        d2.collect_dependency(preds);
      }
      Self::Union(d1, d2) => {
        d1.collect_dependency(preds);
        d2.collect_dependency(preds);
      }
    }
  }
}
