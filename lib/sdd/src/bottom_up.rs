use std::collections::*;

use super::*;

pub struct SDDBuilderConfig {
  vtree: VTree,
  garbage_collect: bool,
}

impl SDDBuilderConfig {
  pub fn new(vars: Vec<usize>, vtree_type: VTreeType, garbage_collect: bool) -> Self {
    let vtree = VTree::new_with_type(vars, vtree_type);
    Self {
      vtree,
      garbage_collect,
    }
  }

  pub fn vars(&self) -> &Vec<usize> {
    &self.vtree.vars
  }

  pub fn with_formula(form: &BooleanFormula) -> Self {
    let vars = form.collect_vars();
    let vtree = VTree::new_with_type(vars, VTreeType::default());
    Self {
      vtree,
      garbage_collect: true,
    }
  }

  pub fn disable_garbage_collect(mut self) -> Self {
    self.garbage_collect = false;
    self
  }

  pub fn enable_garbage_collect(mut self) -> Self {
    self.garbage_collect = true;
    self
  }
}

pub struct SDDBuilder {
  config: SDDBuilderConfig,

  // Core
  sdd_nodes: SDDNodes,
  roots: Vec<SDDNodeIndex>,

  // Helper caches
  false_node: SDDNodeIndex,
  true_node: SDDNodeIndex,
  pos_var_nodes: HashMap<usize, SDDNodeIndex>,
  neg_var_nodes: HashMap<usize, SDDNodeIndex>,
  negation_map: HashMap<SDDNodeIndex, SDDNodeIndex>,
  sdd_node_to_vtree_node_map: HashMap<SDDNodeIndex, VTreeNodeIndex>,
  apply_cache: HashMap<(SDDNodeIndex, SDDNodeIndex, ApplyOp), SDDNodeIndex>,

  // Builder states
  apply_depth: usize,

  // Statistics
  apply_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ApplyOp {
  Conjoin,
  Disjoin,
}

impl SDDBuilder {
  pub fn with_config(config: SDDBuilderConfig) -> Self {
    // Generate new graph
    let mut sdd_nodes = SDDNodes::new();

    // False and True Nodes
    let false_node = sdd_nodes.add_node(SDDNode::Literal {
      literal: SDDLiteral::False,
    });
    let true_node = sdd_nodes.add_node(SDDNode::Literal {
      literal: SDDLiteral::True,
    });

    // Positive and Negative variable Nodes
    let pos_var_nodes = config
      .vtree
      .vars
      .iter()
      .map(|var_id| {
        (
          var_id.clone(),
          sdd_nodes.add_node(SDDNode::Literal {
            literal: SDDLiteral::PosVar {
              var_id: var_id.clone(),
            },
          }),
        )
      })
      .collect::<HashMap<_, _>>();
    let neg_var_nodes = config
      .vtree
      .vars
      .iter()
      .map(|var_id| {
        (
          var_id.clone(),
          sdd_nodes.add_node(SDDNode::Literal {
            literal: SDDLiteral::NegVar {
              var_id: var_id.clone(),
            },
          }),
        )
      })
      .collect::<HashMap<_, _>>();

    // Negation map:
    // - pos variables are negation of neg variables
    // - neg variables are negation of pos variables
    // - true is negation of false
    // - false is negation of true
    let negation_map = pos_var_nodes
      .iter()
      .map(|(var_id, pos_node_id)| (pos_node_id.clone(), neg_var_nodes[var_id]))
      .chain(
        neg_var_nodes
          .iter()
          .map(|(var_id, neg_node_id)| (neg_node_id.clone(), pos_var_nodes[var_id])),
      )
      .chain(vec![(false_node, true_node), (true_node, false_node)])
      .collect::<HashMap<_, _>>();

    // Mapping from SDD node to VTree node
    // - All the Pos/Neg var nodes are ampped to their VTree leaf nodes
    let sdd_node_to_vtree_node_map = pos_var_nodes
      .iter()
      .map(|(var_id, pos_node_id)| (pos_node_id.clone(), config.vtree.var_to_node_id_map[var_id]))
      .chain(neg_var_nodes.iter().map(|(var_id, neg_node_id)| {
        (neg_node_id.clone(), config.vtree.var_to_node_id_map[var_id])
      }))
      .collect::<HashMap<_, _>>();
    let apply_cache = HashMap::new();

    // Roots; initialized to empty
    let roots = Vec::new();

    // Construct the builder
    Self {
      config,
      sdd_nodes,
      roots,

      // Helper nodes
      false_node,
      true_node,
      pos_var_nodes,
      neg_var_nodes,
      negation_map,
      sdd_node_to_vtree_node_map,
      apply_cache,

      // States
      apply_depth: 0,

      // Statistics
      apply_count: 0,
    }
  }

  pub fn vars(&self) -> &Vec<usize> {
    self.config.vars()
  }

  pub fn build(mut self, formula: &BooleanFormula) -> SDD {
    // Build SDD
    let root = self.build_sdd(formula);
    self.roots.push(root);

    // Do garbage collection if presented
    if self.config.garbage_collect {
      self.garbage_collect();
    }

    // Create an SDD
    SDD {
      sdd_nodes: self.sdd_nodes,
      roots: self.roots,
    }
  }

  pub fn add_formula(&mut self, formula: &BooleanFormula) -> usize {
    let num_roots = self.roots.len();
    let new_root = self.build_sdd(formula);
    self.roots.push(new_root);
    num_roots
  }

  pub fn build_arena(mut self) -> SDD {
    if self.config.garbage_collect {
      self.garbage_collect();
    }

    SDD {
      sdd_nodes: self.sdd_nodes,
      roots: self.roots,
    }
  }

  fn mark_visited(sdd_nodes: &SDDNodes, node: SDDNodeIndex, visited: &mut HashSet<SDDNodeIndex>) {
    visited.insert(node);
    match &sdd_nodes[node] {
      SDDNode::Literal { .. } => {}
      SDDNode::Or { children } => {
        for SDDElement { prime, sub } in children {
          Self::mark_visited(sdd_nodes, prime.clone(), visited);
          Self::mark_visited(sdd_nodes, sub.clone(), visited);
        }
      }
    }
  }

  fn remove_not_visited(sdd_nodes: &mut SDDNodes, visited: &HashSet<SDDNodeIndex>) {
    sdd_nodes.retain(|n| visited.contains(&n))
  }

  pub fn garbage_collect(&mut self) {
    let mut visited = HashSet::new();
    if self.roots.len() > 0 {
      for root in &self.roots {
        Self::mark_visited(&self.sdd_nodes, root.clone(), &mut visited);
      }
      Self::remove_not_visited(&mut self.sdd_nodes, &visited);
    }
  }

  pub fn build_sdd(&mut self, formula: &BooleanFormula) -> SDDNodeIndex {
    match formula {
      BooleanFormula::True => self.true_node,
      BooleanFormula::False => self.false_node,
      BooleanFormula::Pos { var_id } => self.pos_var_nodes[var_id],
      BooleanFormula::Neg { var_id } => self.neg_var_nodes[var_id],
      BooleanFormula::Not { form } => {
        let form_id = self.build_sdd(form);
        self.negate_node(form_id)
      }
      BooleanFormula::And { left, right } => {
        let left_id = self.build_sdd(left);
        let right_id = self.build_sdd(right);
        self.apply(left_id, right_id, ApplyOp::Conjoin)
      }
      BooleanFormula::Or { left, right } => {
        let left_id = self.build_sdd(left);
        let right_id = self.build_sdd(right);
        self.apply(left_id, right_id, ApplyOp::Disjoin)
      }
    }
  }

  fn negation_of(&mut self, node: SDDNodeIndex) -> Option<SDDNodeIndex> {
    self.negation_map.get(&node).map(SDDNodeIndex::clone)
  }

  fn zero(&self, op: ApplyOp) -> SDDNodeIndex {
    match op {
      ApplyOp::Conjoin => self.false_node,
      ApplyOp::Disjoin => self.true_node,
    }
  }

  #[allow(dead_code)]
  fn one(&self, op: ApplyOp) -> SDDNodeIndex {
    match op {
      ApplyOp::Conjoin => self.true_node,
      ApplyOp::Disjoin => self.false_node,
    }
  }

  fn is_zero(&self, node: SDDNodeIndex, op: ApplyOp) -> bool {
    match op {
      ApplyOp::Conjoin => node == self.false_node,
      ApplyOp::Disjoin => node == self.true_node,
    }
  }

  fn is_false(&self, node: SDDNodeIndex) -> bool {
    self.false_node == node
  }

  #[allow(dead_code)]
  fn is_true(&self, node: SDDNodeIndex) -> bool {
    self.true_node == node
  }

  fn is_one(&self, node: SDDNodeIndex, op: ApplyOp) -> bool {
    match op {
      ApplyOp::Conjoin => node == self.true_node,
      ApplyOp::Disjoin => node == self.false_node,
    }
  }

  fn vtree_node(&self, sdd_node: SDDNodeIndex) -> VTreeNodeIndex {
    self.sdd_node_to_vtree_node_map[&sdd_node]
  }

  fn add_or_node(&mut self, children: Vec<SDDElement>, vtree_node: VTreeNodeIndex) -> SDDNodeIndex {
    // Apply shortcuts
    if children.len() == 2 {
      if Some(children[0].prime) == self.negation_of(children[1].prime) {
        if children[0].sub == self.false_node && children[1].sub == self.true_node {
          return children[1].prime;
        } else if children[0].sub == self.true_node && children[1].sub == self.false_node {
          return children[0].prime;
        } else if children[0].sub == children[1].sub {
          return children[0].sub;
        }
      }
    }

    // Create node
    let node = SDDNode::Or { children };
    let node_id = self.sdd_nodes.add_node(node);

    // Update vtree link
    self.sdd_node_to_vtree_node_map.insert(node_id, vtree_node);

    // Return node id
    return node_id;
  }

  fn cache_apply_result(
    &mut self,
    lhs: SDDNodeIndex,
    rhs: SDDNodeIndex,
    op: ApplyOp,
    result_node: SDDNodeIndex,
  ) {
    self.apply_cache.insert((lhs, rhs, op), result_node);
  }

  fn lookup_apply_cache(
    &self,
    lhs: SDDNodeIndex,
    rhs: SDDNodeIndex,
    op: ApplyOp,
  ) -> Option<SDDNodeIndex> {
    self
      .apply_cache
      .get(&(lhs, rhs, op))
      .map(SDDNodeIndex::clone)
  }

  fn negate_node(&mut self, n: SDDNodeIndex) -> SDDNodeIndex {
    // Check if there is
    if let Some(neg) = self.negation_of(n) {
      return neg;
    }

    // Prime-sub stack
    let mut neg_children = Vec::new();
    if let SDDNode::Or { children } = self.sdd_nodes[n].clone() {
      for SDDElement { prime, sub } in children {
        let sub_neg = self.negate_node(sub.clone());
        neg_children.push(SDDElement {
          prime,
          sub: sub_neg,
        });
      }
    }

    // Insert negated node
    let neg = self.add_or_node(neg_children, self.sdd_node_to_vtree_node_map[&n]);

    // Update negation
    self.negation_map.insert(n, neg);
    self.negation_map.insert(neg, n);

    neg
  }

  fn apply_equal(
    &mut self,
    n1: SDDNodeIndex,
    n2: SDDNodeIndex,
    op: ApplyOp,
    lca: VTreeNodeIndex,
  ) -> SDDNodeIndex {
    let mut new_children = Vec::new();

    // Get the children; they should both have children
    let n1_sdd = self.sdd_nodes[n1].clone();
    let n2_sdd = self.sdd_nodes[n2].clone();
    let (c1, c2) = match (n1_sdd, n2_sdd) {
      (SDDNode::Or { children: c1 }, SDDNode::Or { children: c2 }) => (c1, c2),
      _ => panic!("Should not happen"),
    };

    // Do cartesian product
    for SDDElement { prime: p1, sub: s1 } in &c1 {
      for SDDElement { prime: p2, sub: s2 } in &c2 {
        // Generate prime
        let new_prime = self.apply(p1.clone(), p2.clone(), ApplyOp::Conjoin);

        // Shortcut for prime
        if self.is_false(new_prime) {
          continue;
        }

        // Generate sub
        let new_sub = self.apply(s1.clone(), s2.clone(), op);
        new_children.push(SDDElement {
          prime: new_prime,
          sub: new_sub,
        });
      }
    }

    // Add the node
    self.add_or_node(new_children, lca)
  }

  fn apply_left(
    &mut self,
    n1: SDDNodeIndex,
    n2: SDDNodeIndex,
    op: ApplyOp,
    lca: VTreeNodeIndex,
  ) -> SDDNodeIndex {
    let n1_neg = self.negate_node(n1);
    let n = match op {
      ApplyOp::Conjoin => n1,
      ApplyOp::Disjoin => n1_neg,
    };

    // Create the set of new elements
    let mut new_children = Vec::new();
    new_children.push(SDDElement {
      prime: self.negation_of(n).unwrap(), // Unwrap as we just created negated node of n1
      sub: self.zero(op),
    });

    // n2 has to be an OR node as n1 vtree is a subtree of n2 vtree
    match self.sdd_nodes[n2].clone() {
      SDDNode::Or { children } => {
        for SDDElement { prime, sub } in children {
          let new_prime = self.apply(prime, n, ApplyOp::Conjoin);
          if !self.is_false(new_prime) {
            new_children.push(SDDElement {
              prime: new_prime,
              sub: sub,
            });
          }
        }
      }
      _ => panic!("Should not happen"),
    }

    // Construct new or node
    self.add_or_node(new_children, lca)
  }

  fn apply_right(
    &mut self,
    n1: SDDNodeIndex,
    n2: SDDNodeIndex,
    op: ApplyOp,
    lca: VTreeNodeIndex,
  ) -> SDDNodeIndex {
    // n1 has to be an OR node as n2 tree is a subtree of n1 tree
    match self.sdd_nodes[n1].clone() {
      SDDNode::Or { children } => {
        let mut new_children = Vec::new();
        for SDDElement { prime, sub } in children {
          let new_sub = self.apply(sub.clone(), n2, op);
          new_children.push(SDDElement {
            prime: prime.clone(),
            sub: new_sub,
          });
        }

        // Construct new or node
        self.add_or_node(new_children, lca)
      }
      _ => panic!("Should not happen"),
    }
  }

  fn apply_disjoint(
    &mut self,
    n1: SDDNodeIndex,
    n2: SDDNodeIndex,
    op: ApplyOp,
    lca: VTreeNodeIndex,
  ) -> SDDNodeIndex {
    let n1_neg = self.negate_node(n1);
    let n1_sub = self.apply(n2, self.true_node, op);
    let n1_neg_sub = self.apply(n2, self.false_node, op);

    // Construct the new OR node
    let e1 = SDDElement {
      prime: n1,
      sub: n1_sub,
    };
    let e2 = SDDElement {
      prime: n1_neg,
      sub: n1_neg_sub,
    };

    // Add new node
    self.add_or_node(vec![e1, e2], lca)
  }

  fn apply(&mut self, lhs: SDDNodeIndex, rhs: SDDNodeIndex, op: ApplyOp) -> SDDNodeIndex {
    // If they are the same node, return the node itself
    if lhs == rhs {
      return lhs;
    }

    // If A == ~B, simplify A & B to false or A | B to true
    if Some(lhs) == self.negation_of(rhs) {
      return self.zero(op);
    }

    // If A or B is false, then A & B is false
    // If A or B is true, then A | B is true
    if self.is_zero(lhs, op) || self.is_zero(rhs, op) {
      return self.zero(op);
    }

    // If A is true, then A & B is B
    // If A is false, then A | B is B
    if self.is_one(lhs, op) {
      return rhs;
    }

    // The same applies for B
    if self.is_one(rhs, op) {
      return lhs;
    }

    // Check if there is cached computation result
    if let Some(cached_node_id) = self.lookup_apply_cache(lhs, rhs, op) {
      return cached_node_id;
    }

    // Increment depth
    self.apply_depth += 1;

    // Statistics
    self.apply_count += 1;

    // Swap the two nodes if their respective position is out of order
    let lhs_v = self.vtree_node(lhs);
    let rhs_v = self.vtree_node(rhs);
    let lhs_vpos = self.config.vtree.position(lhs_v);
    let rhs_vpos = self.config.vtree.position(rhs_v);
    let ((lhs, lhs_v), (rhs, rhs_v)) = if lhs_vpos > rhs_vpos {
      ((rhs, rhs_v), (lhs, lhs_v))
    } else {
      ((lhs, lhs_v), (rhs, rhs_v))
    };

    // Get the lowest common ancestor
    let (anc_type, lca) = self.config.vtree.lowest_common_ancestor(lhs_v, rhs_v);

    // Real apply
    let result_node = match anc_type {
      AncestorType::Equal => self.apply_equal(lhs, rhs, op, lca),
      AncestorType::Left => self.apply_left(lhs, rhs, op, lca),
      AncestorType::Right => self.apply_right(lhs, rhs, op, lca),
      AncestorType::Disjoint => self.apply_disjoint(lhs, rhs, op, lca),
    };

    // Cache
    self.cache_apply_result(lhs, rhs, op, result_node);

    // Decrement depth
    self.apply_depth -= 1;

    // Return the node
    result_node
  }
}
