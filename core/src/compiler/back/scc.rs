use colored::*;
use petgraph::{
  graph::{EdgeIndex, Graph, NodeIndex},
  visit::*,
  EdgeDirection::{Incoming, Outgoing},
};
use std::collections::*;

use super::{ast::*, BackCompileError};

/// The type of a dependency graph edge: positive, negative, or aggregation
#[derive(Debug, Clone)]
pub enum DependencyGraphEdge {
  Positive,
  Negative,
  Aggregation,
}

impl std::fmt::Display for DependencyGraphEdge {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Positive => f.write_str("positive"),
      Self::Negative => f.write_str("negative"),
      Self::Aggregation => f.write_str("aggregation"),
    }
  }
}

impl DependencyGraphEdge {
  /// Check if the edge needs to be stratified
  pub fn needs_stratification(&self) -> bool {
    match self {
      Self::Positive => false,
      _ => true,
    }
  }
}

/// A graph storing the dependencies between predicates
#[derive(Clone)]
pub struct DependencyGraph {
  graph: Graph<String, DependencyGraphEdge>,
  predicate_to_node_id: HashMap<String, NodeIndex>,
  sccs: Vec<Vec<NodeIndex>>,
}

impl DependencyGraph {
  /// Create an empty dependency graph
  pub fn new() -> Self {
    Self {
      graph: Graph::new(),
      predicate_to_node_id: HashMap::new(),
      sccs: Vec::new(),
    }
  }

  /// Add a predicate to the dependency graph
  pub fn add_predicate(&mut self, predicate: &String) -> NodeIndex {
    if let Some(ni) = self.predicate_to_node_id.get(predicate) {
      ni.clone()
    } else {
      let node_id = self.graph.add_node(predicate.clone());
      self.predicate_to_node_id.insert(predicate.clone(), node_id);
      node_id
    }
  }

  /// Get the node id of a predicate
  pub fn predicate_node(&self, predicate: &String) -> NodeIndex {
    self.predicate_to_node_id.get(predicate).unwrap().clone()
  }

  /// Add a dependency between two predicates
  pub fn add_dependency(&mut self, src: &String, dst: &String, edge: DependencyGraphEdge) -> EdgeIndex {
    let src_id = self.predicate_to_node_id[src];
    let dst_id = self.predicate_to_node_id[dst];
    self.graph.add_edge(src_id, dst_id, edge)
  }

  pub fn unused_relations(&self, targets: &HashSet<String>) -> HashSet<String> {
    let mut unrelated = HashSet::new();
    let output_ids = targets.iter().map(|r| self.predicate_node(r)).collect::<Vec<_>>();
    for (predicate, rid) in &self.predicate_to_node_id {
      if !targets.contains(predicate) {
        let connected_to_output = output_ids
          .iter()
          .any(|oid| petgraph::algo::has_path_connecting(&self.graph, *oid, *rid, None));
        if !connected_to_output {
          unrelated.insert(predicate.clone());
        }
      }
    }
    unrelated
  }

  pub fn compute_scc(&mut self) {
    self.sccs = petgraph::algo::kosaraju_scc(&self.graph);
  }

  pub fn stratify(&self) -> Result<Vec<Stratum>, SCCError> {
    // First check if it is possible to stratify
    for scc in &self.sccs {
      for node_id in scc {
        for edge in self.graph.edges(*node_id) {
          if edge.weight().needs_stratification() && scc.contains(&edge.target()) {
            return Err(SCCError::CannotStratify {
              pred_1: self.graph[edge.source()].clone(),
              pred_2: self.graph[edge.target()].clone(),
              edge: edge.weight().clone(),
            });
          }
        }
      }
    }

    // Then create strata for each, using topological sorting
    let scc_graph = petgraph::algo::condensation(self.graph.clone(), true);

    // Initialize the empty (sorted) list of stratass
    let mut visited = HashSet::new();
    let mut stratas = Vec::new();

    // Compute the nodes without incoming edges
    let source_nodes = scc_graph
      .node_indices()
      .filter(|n| scc_graph.edges(*n).next().is_none())
      .collect::<Vec<_>>();

    // Add source nodes into visited
    visited.extend(source_nodes.clone());

    // Add sccs with no incoming edges into the fringe
    let mut fringe = source_nodes.into_iter().collect::<VecDeque<_>>();

    // Iterate until fringe is empty
    while !fringe.is_empty() {
      // First get the current scc to process
      let scc = fringe.pop_front().unwrap();
      let scc_predicates = scc_graph[scc].clone();
      let scc_is_recursive = self.has_cycle(&scc_graph[scc]);

      // Push an initial strata
      stratas.push(Stratum {
        predicates: scc_predicates,
        is_recursive: scc_is_recursive,
      });

      // Finally add the connected component to the end
      for edge in scc_graph.edges_directed(scc, Incoming) {
        let next_node = edge.source();
        if !visited.contains(&next_node) {
          if scc_graph
            .edges_directed(next_node, Outgoing)
            .all(|e| visited.contains(&e.target()))
          {
            visited.insert(edge.source());
            fringe.push_back(edge.source());
          }
        }
      }
    }

    Ok(stratas)
  }

  fn has_cycle(&self, domain: &Vec<String>) -> bool {
    if domain.is_empty() {
      false
    } else if domain.len() == 1 {
      let node_id = self.predicate_to_node_id[&domain[0]];
      self.graph.contains_edge(node_id, node_id)
    } else {
      true
    }
  }
}

#[derive(Debug, Clone)]
pub struct Stratum {
  pub predicates: Vec<String>,
  pub is_recursive: bool,
}

#[derive(Debug, Clone)]
pub enum SCCError {
  CannotStratify {
    pred_1: String,
    pred_2: String,
    edge: DependencyGraphEdge,
  },
}

impl std::fmt::Display for SCCError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::CannotStratify { pred_1, pred_2, edge } => f.write_fmt(format_args!(
        "{} Cannot stratify program: {} cycle detected between predicate `{}` and `{}`",
        "[Error]".red(),
        edge,
        pred_1,
        pred_2
      )),
    }
  }
}

impl From<SCCError> for BackCompileError {
  fn from(e: SCCError) -> Self {
    Self::SCCError(e)
  }
}

impl std::fmt::Debug for DependencyGraph {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for (i, scc) in self.sccs.iter().enumerate() {
      f.write_fmt(format_args!("SCC#{}: {{", i))?;
      f.write_str(
        &scc
          .iter()
          .map(|j| self.graph[*j].clone())
          .collect::<Vec<_>>()
          .join(", "),
      )?;
      f.write_str("}")?;
      if i < self.sccs.len() - 1 {
        f.write_str(", ")?;
      }
    }
    Ok(())
  }
}

impl Program {
  pub fn dependency_graph(&self) -> DependencyGraph {
    type E = DependencyGraphEdge;

    // The graph to work on
    let mut graph = DependencyGraph::new();

    // First add all the nodes
    for relation in &self.relations {
      graph.add_predicate(&relation.predicate);
    }

    // Then add all the dependencies by going through rules
    for rule in &self.rules {
      let head_predicate = rule.head_predicate();
      for atom in rule.body_literals() {
        match atom {
          Literal::Atom(a) => {
            let atom_predicate = &a.predicate;
            if !self.predicate_registry.contains(atom_predicate) {
              graph.add_dependency(head_predicate, atom_predicate, E::Positive);
            }
          }
          Literal::NegAtom(a) => {
            let atom_predicate = &a.atom.predicate;
            if !self.predicate_registry.contains(atom_predicate) {
              graph.add_dependency(head_predicate, atom_predicate, E::Negative);
            }
          }
          Literal::Reduce(r) => {
            let reduce_predicate = &r.body_formula.predicate;
            graph.add_dependency(head_predicate, reduce_predicate, E::Aggregation);

            // Add group by predicate also as an aggregation dependency
            if let Some(group_by_atom) = &r.group_by_formula {
              let group_by_predicate = &group_by_atom.predicate;
              graph.add_dependency(head_predicate, group_by_predicate, E::Aggregation);
            }
          }
          _ => {}
        }
      }
    }

    // Return the graph
    graph
  }
}
