use std::collections::*;

use super::*;

#[derive(Clone, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct SDDElement {
  pub prime: SDDNodeIndex,
  pub sub: SDDNodeIndex,
}

impl std::fmt::Debug for SDDElement {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "Elem {{ p: {:?}, s: {:?} }}",
      self.prime, self.sub
    ))
  }
}

#[derive(Clone)]
pub enum SDDLiteral {
  PosVar { var_id: usize },
  NegVar { var_id: usize },
  True,
  False,
}

impl std::fmt::Debug for SDDLiteral {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::PosVar { var_id } => f.write_fmt(format_args!("Pos({})", var_id)),
      Self::NegVar { var_id } => f.write_fmt(format_args!("Neg({})", var_id)),
      Self::True => f.write_str("True"),
      Self::False => f.write_str("False"),
    }
  }
}

#[derive(Clone)]
pub enum SDDNode {
  Or { children: Vec<SDDElement> },
  Literal { literal: SDDLiteral },
}

impl std::fmt::Debug for SDDNode {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Literal { literal } => f.write_fmt(format_args!("{:?}", literal)),
      Self::Or { children } => f.write_fmt(format_args!("Or {{ {:?} }}", children)),
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SDDNodeIndex(usize);

impl std::fmt::Debug for SDDNodeIndex {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("SDD({})", self.0))
  }
}

#[derive(Clone)]
pub struct SDDNodes {
  nodes: HashMap<usize, SDDNode>,
  max_id: usize,
}

impl SDDNodes {
  pub fn new() -> Self {
    Self {
      nodes: HashMap::new(),
      max_id: 0,
    }
  }

  pub fn add_node(&mut self, node: SDDNode) -> SDDNodeIndex {
    self.nodes.insert(self.max_id, node);
    let id = SDDNodeIndex(self.max_id);
    self.max_id += 1;
    id
  }

  pub fn remove_node(&mut self, id: SDDNodeIndex) {
    self.nodes.remove(&id.0);
  }

  pub fn len(&self) -> usize {
    self.nodes.len()
  }

  pub fn retain<F>(&mut self, mut f: F)
  where
    F: FnMut(SDDNodeIndex) -> bool,
  {
    self.nodes.retain(|n, _| f(SDDNodeIndex(n.clone())))
  }
}

impl std::ops::Index<SDDNodeIndex> for SDDNodes {
  type Output = SDDNode;

  fn index(&self, id: SDDNodeIndex) -> &Self::Output {
    &self.nodes[&id.0]
  }
}

impl std::fmt::Debug for SDDNodes {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "SDDNodes {{ num_nodes: {}, nodes: {:?} }}",
      self.nodes.len(),
      self.nodes
    ))
  }
}

#[derive(Clone, Debug)]
pub struct SDD {
  pub sdd_nodes: SDDNodes,
  pub roots: Vec<SDDNodeIndex>,
}

impl SDD {
  pub fn eval(&self, var_assign: &HashMap<usize, bool>) -> bool {
    self.eval_node(self.roots[0], var_assign)
  }

  pub fn eval_i<T: IntoIterator<Item = (usize, bool)>>(&self, var_assign: T) -> bool {
    self.eval_node(self.roots[0], &var_assign.into_iter().collect())
  }

  fn eval_node(&self, node_id: SDDNodeIndex, var_assign: &HashMap<usize, bool>) -> bool {
    match &self.sdd_nodes[node_id] {
      SDDNode::Or { children } => {
        for child in children {
          // First evaluate the prime
          let result = self.eval_node(child.prime, var_assign);

          // If prime holds, return the evaluated value of the sub
          if result {
            return self.eval_node(child.sub, var_assign);
          }
        }
        panic!("Mutual exclusion violated")
      }
      SDDNode::Literal { literal } => match literal {
        SDDLiteral::PosVar { var_id } => var_assign[var_id],
        SDDLiteral::NegVar { var_id } => !var_assign[var_id],
        SDDLiteral::True => true,
        SDDLiteral::False => false,
      },
    }
  }

  pub fn eval_t<T, V>(&self, var_assign: &V, semiring: &T) -> T::Element
  where
    T: Semiring,
    V: Fn(&usize) -> T::Element,
  {
    self.eval_node_t(self.roots[0], var_assign, semiring)
  }

  fn eval_node_t<T, V>(&self, node_id: SDDNodeIndex, var_assign: &V, semiring: &T) -> T::Element
  where
    T: Semiring,
    V: Fn(&usize) -> T::Element,
  {
    match &self.sdd_nodes[node_id] {
      SDDNode::Or { children } => {
        children
          .iter()
          .fold(semiring.zero(), |acc, SDDElement { prime, sub }| {
            let prime_res = self.eval_node_t(prime.clone(), var_assign, semiring);
            let sub_res = self.eval_node_t(sub.clone(), var_assign, semiring);
            semiring.add(acc, semiring.mult(prime_res, sub_res))
          })
      }
      SDDNode::Literal { literal } => match literal {
        SDDLiteral::PosVar { var_id } => var_assign(var_id),
        SDDLiteral::NegVar { var_id } => semiring.negate(var_assign(var_id).clone()),
        SDDLiteral::True => semiring.one(),
        SDDLiteral::False => semiring.zero(),
      },
    }
  }

  pub fn dot(&self) -> String {
    fn literal_label(literal: &SDDLiteral) -> String {
      match literal {
        SDDLiteral::True => format!("⊤"),
        SDDLiteral::False => format!("⊥"),
        SDDLiteral::PosVar { var_id } => format!("V{}", var_id),
        SDDLiteral::NegVar { var_id } => format!("¬V{}", var_id),
      }
    }

    fn node_identifier(node_id: SDDNodeIndex) -> String {
      format!("N{}", node_id.0)
    }

    fn literal_label_or_node_identifier(nodes: &SDDNodes, node_id: SDDNodeIndex) -> String {
      match &nodes[node_id] {
        SDDNode::Or { .. } => node_identifier(node_id),
        SDDNode::Literal { literal } => literal_label(literal),
      }
    }

    fn traverse(
      nodes: &SDDNodes,
      curr_node: SDDNodeIndex,
      node_strs: &mut Vec<String>,
      edge_strs: &mut Vec<String>,
      elem_id: &mut usize,
    ) {
      let curr_label = node_identifier(curr_node);
      match &nodes[curr_node.clone()] {
        SDDNode::Or { children } => {
          node_strs.push(format!("{} [label=\"OR\", shape=circle];", curr_label));
          for SDDElement { prime, sub } in children {
            // Get element label
            let elem_label = format!("E{}", elem_id);
            *elem_id += 1;

            // Get child label
            let prime_label = literal_label_or_node_identifier(nodes, prime.clone());
            let sub_label = literal_label_or_node_identifier(nodes, sub.clone());

            // Add nodes
            node_strs.push(format!(
              "{} [label=\"<prime>{} | <sub>{}\"];",
              elem_label, prime_label, sub_label
            ));

            // Add Or to Elem edge
            edge_strs.push(format!("{} -> {};", curr_label, elem_label));

            // Add Elem to Child edge and continue traverse
            match &nodes[prime.clone()] {
              SDDNode::Or { .. } => {
                edge_strs.push(format!("{}:prime -> {};", elem_label, prime_label));
                traverse(nodes, prime.clone(), node_strs, edge_strs, elem_id);
              }
              _ => {}
            }
            match &nodes[sub.clone()] {
              SDDNode::Or { .. } => {
                edge_strs.push(format!("{}:sub -> {};", elem_label, sub_label));
                traverse(nodes, sub.clone(), node_strs, edge_strs, elem_id);
              }
              _ => {}
            }
          }
        }
        SDDNode::Literal { literal } => {
          let node_str = format!("{} [label=\"{}\"]", curr_label, literal_label(literal));
          node_strs.push(node_str);
        }
      }
    }

    let mut node_strs = vec![];
    let mut edge_strs = vec![];
    let mut elem_id = 0;

    for root in &self.roots {
      traverse(
        &self.sdd_nodes,
        root.clone(),
        &mut node_strs,
        &mut edge_strs,
        &mut elem_id,
      );
    }

    format!(
      "digraph sdd {{ node [shape=record margin=0.03 width=0 height=0]; {} {} }}",
      node_strs.join(" "),
      edge_strs.join(" ")
    )
  }

  pub fn save_dot(&self, file_name: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::prelude::*;
    let mut file = File::create(file_name)?;
    file.write_all(self.dot().as_bytes())?;
    Ok(())
  }
}
