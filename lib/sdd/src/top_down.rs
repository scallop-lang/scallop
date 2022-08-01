use itertools::iproduct;
use std::collections::*;

use rsat::Solver as SATState;

use super::*;

pub struct SDDBuilderConfig {
  vtree: VTree,
}

impl SDDBuilderConfig {
  pub fn new(vars: Vec<usize>, vtree_type: VTreeType) -> Self {
    let vtree = VTree::new_with_type(vars, vtree_type);
    Self { vtree }
  }

  pub fn vars(&self) -> &Vec<usize> {
    &self.vtree.vars
  }

  pub fn with_formula(form: &BooleanFormula) -> Self {
    let vars = form.collect_vars();
    let vtree = VTree::new_with_type(vars, VTreeType::default());
    Self { vtree }
  }
}

pub struct SDDBuilder {
  config: SDDBuilderConfig,

  // Core
  sdd_nodes: SDDNodes,

  // Helper caches
  false_node: SDDNodeIndex,
  true_node: SDDNodeIndex,
  pos_var_nodes: HashMap<usize, SDDNodeIndex>,
  neg_var_nodes: HashMap<usize, SDDNodeIndex>,
  negation_map: HashMap<SDDNodeIndex, SDDNodeIndex>,
  sdd_node_to_vtree_node_map: HashMap<SDDNodeIndex, VTreeNodeIndex>,
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
            literal: SDDLiteral::PosVar { var_id: var_id.clone() },
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
            literal: SDDLiteral::NegVar { var_id: var_id.clone() },
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
      .chain(
        neg_var_nodes
          .iter()
          .map(|(var_id, neg_node_id)| (neg_node_id.clone(), config.vtree.var_to_node_id_map[var_id])),
      )
      .collect::<HashMap<_, _>>();

    // Result
    Self {
      config,
      sdd_nodes,
      false_node,
      true_node,
      pos_var_nodes,
      neg_var_nodes,
      negation_map,
      sdd_node_to_vtree_node_map,
    }
  }

  pub fn vars(&self) -> &Vec<usize> {
    self.config.vars()
  }

  pub fn build(mut self, form: BooleanFormula) -> SDD {
    let cnf = boolean_formula_to_rsat_cnf(form);
    let state = SATState::new(cnf);
    match self.c2s(self.config.vtree.root_id(), state) {
      Ok((_, sdd_root)) => SDD {
        sdd_nodes: self.sdd_nodes,
        roots: vec![sdd_root],
      },
      Err(_) => SDD {
        sdd_nodes: self.sdd_nodes,
        roots: vec![self.false_node],
      },
    }
  }

  pub fn c2s(&mut self, v: VTreeNodeIndex, state: SATState) -> Result<(SATState, SDDNodeIndex), rsat::Clause> {
    // TODO: see if there is cached result

    let result = if let Some(var_id) = self.config.vtree.leaf(v) {
      self.c2s_leaf(var_id, state)
    } else if self.config.vtree.is_decomposition(v) {
      self.c2s_decomposition(v, state)
    } else {
      self.c2s_shannon(v, state)
    };

    // TODO: Cache result

    result
  }

  fn c2s_leaf(&mut self, var_id: usize, state: SATState) -> Result<(SATState, SDDNodeIndex), rsat::Clause> {
    match state.variable_status(rsat::Variable::new(var_id)) {
      rsat::VariableStatus::None => Ok((state, self.true_node.clone())),
      rsat::VariableStatus::Pos => Ok((state, self.pos_var_nodes[&var_id])),
      rsat::VariableStatus::Neg => Ok((state, self.neg_var_nodes[&var_id])),
    }
  }

  fn c2s_decomposition(
    &mut self,
    v: VTreeNodeIndex,
    state: SATState,
  ) -> Result<(SATState, SDDNodeIndex), rsat::Clause> {
    let (p_state, p) = match self.c2s(self.config.vtree.left(v).unwrap(), state) {
      Ok(n) => n,
      Err(clause) => {
        // TODO: Drop cache
        return Err(clause);
      }
    };
    let (s_state, s) = match self.c2s(self.config.vtree.right(v).unwrap(), p_state) {
      Ok(n) => n,
      Err(clause) => {
        // TODO: Drop cache
        return Err(clause);
      }
    };
    let children = vec![
      SDDElement { prime: p, sub: s },
      SDDElement {
        prime: self.negate_node(p),
        sub: self.false_node,
      },
    ];
    return Ok((s_state, self.add_or_node(children, v)));
  }

  fn c2s_shannon(&mut self, v: VTreeNodeIndex, state: SATState) -> Result<(SATState, SDDNodeIndex), rsat::Clause> {
    let shannon_var = self.config.vtree.shannon_variable(v).unwrap();
    let var = rsat::Variable::new(shannon_var);

    // Create the pos/neg of the shannon var
    let (has_plit, has_nlit) = (state.implied_positive(var), state.implied_negative(var));
    let (pn, nn) = (self.pos_var_nodes[&shannon_var], self.neg_var_nodes[&shannon_var]);
    if has_plit || has_nlit {
      let primes = if has_plit { (pn, nn) } else { (nn, pn) };
      match self.c2s(self.config.vtree.right(v).unwrap(), state) {
        Err(c) => return Err(c),
        Ok((next_state, n)) => {
          let children = vec![
            SDDElement {
              prime: primes.0,
              sub: n,
            },
            SDDElement {
              prime: primes.1,
              sub: self.false_node,
            },
          ];
          return Ok((next_state, self.add_or_node(children, v)));
        }
      }
    }

    // First check positive shannon variable
    let (l_state, s1) = match state.clone().decide_literal(rsat::Literal::positive(var)) {
      Ok(new_state) => (
        state,
        self.c2s(self.config.vtree.right(v).unwrap(), new_state).map(|r| r.1)?,
      ),
      Err(c) => {
        if state.at_assertion_level(&c) {
          return state.assert_clause(c).and_then(|new| self.c2s_shannon(v, new));
        } else {
          return Err(c);
        }
      }
    };

    // Then check negative shannon variable
    let (r_state, s2) = match l_state.clone().decide_literal(rsat::Literal::negative(var)) {
      Ok(new_state) => (
        l_state,
        self.c2s(self.config.vtree.right(v).unwrap(), new_state).map(|r| r.1)?,
      ),
      Err(c) => {
        if l_state.at_assertion_level(&c) {
          return l_state.assert_clause(c).and_then(|new| self.c2s_shannon(v, new));
        } else {
          return Err(c);
        }
      }
    };

    // Return an or node
    let children = vec![SDDElement { prime: pn, sub: s1 }, SDDElement { prime: nn, sub: s2 }];
    Ok((r_state, self.add_or_node(children, v)))
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

  fn negation_of(&mut self, node: SDDNodeIndex) -> Option<SDDNodeIndex> {
    self.negation_map.get(&node).map(SDDNodeIndex::clone)
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
        neg_children.push(SDDElement { prime, sub: sub_neg });
      }
    }

    // Insert negated node
    let neg = self.add_or_node(neg_children, self.sdd_node_to_vtree_node_map[&n]);

    // Update negation
    self.negation_map.insert(n, neg);
    self.negation_map.insert(neg, n);

    neg
  }
}

pub fn boolean_formula_to_rsat_cnf(form: BooleanFormula) -> rsat::CNF {
  match form {
    BooleanFormula::True => vec![],
    BooleanFormula::False => vec![vec![]],
    BooleanFormula::Pos { var_id } => {
      vec![vec![rsat::Literal::positive(rsat::Variable::new(var_id))]]
    }
    BooleanFormula::Neg { var_id } => {
      vec![vec![rsat::Literal::negative(rsat::Variable::new(var_id))]]
    }
    BooleanFormula::And { left, right } => {
      let lhs = boolean_formula_to_rsat_cnf(*left);
      let rhs = boolean_formula_to_rsat_cnf(*right);
      [lhs, rhs].concat()
    }
    BooleanFormula::Or { left, right } => {
      let lhs = boolean_formula_to_rsat_cnf(*left);
      let rhs = boolean_formula_to_rsat_cnf(*right);
      iproduct!(lhs, rhs)
        .map(|(dl, dr)| {
          let mut merged = [dl, dr].concat();
          merged.sort_by_key(|l| l.raw_id());
          merged.dedup();
          merged
        })
        .collect()
    }
    BooleanFormula::Not { .. } => {
      unimplemented!()
    }
  }
}
