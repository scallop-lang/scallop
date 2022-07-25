use petgraph::{graph::NodeIndex, Direction, Graph};
use std::collections::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VTreeNode {
  Leaf {
    var_id: usize,
  },
  Branch {
    left: VTreeNodeIndex,
    right: VTreeNodeIndex,
  },
}

impl VTreeNode {
  pub fn is_branch(&self) -> bool {
    match self {
      Self::Branch { .. } => true,
      _ => false,
    }
  }

  pub fn is_leaf(&self) -> bool {
    match self {
      Self::Leaf { .. } => true,
      _ => false,
    }
  }

  pub fn is_left(&self, node: VTreeNodeIndex) -> bool {
    match self {
      Self::Branch { left, .. } => left == &node,
      _ => false,
    }
  }

  pub fn is_right(&self, node: VTreeNodeIndex) -> bool {
    match self {
      Self::Branch { right, .. } => right == &node,
      _ => false,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VTreeNodeIndex(NodeIndex);

impl std::fmt::Debug for VTreeNodeIndex {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("VTree({})", self.0.index()))
  }
}

impl VTreeNodeIndex {
  pub fn index(&self) -> usize {
    self.0.index()
  }
}

#[derive(Debug, Clone)]
pub struct VTree {
  pub(crate) tree: Graph<VTreeNode, ()>,
  pub(crate) vars: Vec<usize>,
  pub(crate) root: NodeIndex,

  // Caches
  pub(crate) var_to_node_id_map: HashMap<usize, VTreeNodeIndex>,
  pub(crate) in_order_positions: HashMap<VTreeNodeIndex, usize>,
  pub(crate) first_in_subtree_map: HashMap<VTreeNodeIndex, VTreeNodeIndex>,
  pub(crate) last_in_subtree_map: HashMap<VTreeNodeIndex, VTreeNodeIndex>,
}

#[derive(Debug, Clone, Copy)]
pub enum VTreeType {
  Left,
  Right,
  Balanced,
}

impl Default for VTreeType {
  fn default() -> Self {
    Self::Balanced
  }
}

#[derive(Clone, Copy, Debug)]
pub enum AncestorType {
  Equal,    // LHS is RHS
  Left,     // LHS is a subtree of RHS
  Right,    // RHS is a subtree of LHS
  Disjoint, // LHS and RHS are disjoint
}

impl VTree {
  pub fn new(vars: Vec<usize>) -> Self {
    Self::new_with_type(vars, VTreeType::default())
  }

  pub fn new_with_type(vars: Vec<usize>, ty: VTreeType) -> Self {
    match ty {
      VTreeType::Left => Self::create_left_linear_from_vars(vars),
      VTreeType::Right => Self::create_right_linear_from_vars(vars),
      VTreeType::Balanced => Self::create_balanced_from_vars(vars),
    }
  }

  pub fn create_left_linear(num_vars: usize) -> Self {
    Self::create_left_linear_from_vars((0..num_vars).collect())
  }

  pub fn create_left_linear_from_vars(vars: Vec<usize>) -> Self {
    let mut tree = Graph::new();
    if vars.len() > 0 {
      let mut curr_left = tree.add_node(VTreeNode::Leaf { var_id: vars[0] });
      for i in 1..vars.len() {
        let curr_right_leaf = tree.add_node(VTreeNode::Leaf { var_id: vars[i] });
        let parent_node = tree.add_node(VTreeNode::Branch {
          left: VTreeNodeIndex(curr_left),
          right: VTreeNodeIndex(curr_right_leaf),
        });
        tree.add_edge(parent_node, curr_left, ());
        tree.add_edge(parent_node, curr_right_leaf, ());
        curr_left = parent_node;
      }
      Self::post_process(tree, vars, curr_left)
    } else {
      panic!("Must have at least one variable");
    }
  }

  pub fn create_right_linear(num_vars: usize) -> Self {
    Self::create_right_linear_from_vars((0..num_vars).collect())
  }

  pub fn create_right_linear_from_vars(vars: Vec<usize>) -> Self {
    let mut tree = Graph::new();
    if vars.len() > 0 {
      let mut curr_right = tree.add_node(VTreeNode::Leaf {
        var_id: vars[vars.len() - 1],
      });
      for i in 2..=vars.len() {
        let curr_left_leaf = tree.add_node(VTreeNode::Leaf {
          var_id: vars[vars.len() - i],
        });
        let parent_node = tree.add_node(VTreeNode::Branch {
          left: VTreeNodeIndex(curr_left_leaf),
          right: VTreeNodeIndex(curr_right),
        });
        tree.add_edge(parent_node, curr_left_leaf, ());
        tree.add_edge(parent_node, curr_right, ());
        curr_right = parent_node;
      }
      Self::post_process(tree, vars, curr_right)
    } else {
      panic!("Must have at least one variable");
    }
  }

  pub fn create_balanced(num_vars: usize) -> Self {
    Self::create_balanced_from_vars((0..num_vars).collect())
  }

  pub fn create_balanced_from_vars(vars: Vec<usize>) -> Self {
    let mut tree = Graph::new();

    if vars.is_empty() {
      return Self {
        tree,
        vars,
        root: NodeIndex::new(0),
        var_to_node_id_map: HashMap::new(),
        in_order_positions: HashMap::new(),
        first_in_subtree_map: HashMap::new(),
        last_in_subtree_map: HashMap::new(),
      };
    }

    fn recurse(
      tree: &mut Graph<VTreeNode, ()>,
      vars: &Vec<usize>,
      low: usize,
      high: usize,
    ) -> NodeIndex {
      if low == high - 1 {
        tree.add_node(VTreeNode::Leaf { var_id: vars[low] })
      } else {
        let mid = (high + low) / 2;
        let left_id = recurse(tree, vars, low, mid);
        let right_id = recurse(tree, vars, mid, high);
        let mid_id = tree.add_node(VTreeNode::Branch {
          left: VTreeNodeIndex(left_id),
          right: VTreeNodeIndex(right_id),
        });
        tree.add_edge(mid_id, left_id, ());
        tree.add_edge(mid_id, right_id, ());
        mid_id
      }
    }

    let root = recurse(&mut tree, &vars, 0, vars.len());

    Self::post_process(tree, vars, root)
  }

  fn post_process(tree: Graph<VTreeNode, ()>, vars: Vec<usize>, root: NodeIndex) -> Self {
    // In order position
    fn in_order_traversal<F: FnMut(NodeIndex, &VTreeNode)>(
      tree: &Graph<VTreeNode, ()>,
      node_id: NodeIndex,
      f: &mut F,
    ) {
      match &tree[node_id] {
        VTreeNode::Branch {
          left: VTreeNodeIndex(left),
          right: VTreeNodeIndex(right),
        } => {
          in_order_traversal(tree, left.clone(), f);
          f(node_id, &tree[node_id]);
          in_order_traversal(tree, right.clone(), f);
        }
        VTreeNode::Leaf { .. } => {
          f(node_id, &tree[node_id]);
        }
      }
    }
    let mut var_to_node_id_map = HashMap::new();
    let mut count = 0;
    let mut in_order_positions = HashMap::new();
    in_order_traversal(&tree, root, &mut |node_index, node| {
      in_order_positions.insert(VTreeNodeIndex(node_index), count);
      count += 1;

      if let VTreeNode::Leaf { var_id } = node {
        var_to_node_id_map.insert(var_id.clone(), VTreeNodeIndex(node_index));
      }
    });

    // First and Last in subtree
    fn cache_min_max_in_subtree(
      tree: &Graph<VTreeNode, ()>,
      node_id: NodeIndex,
      min_map: &mut HashMap<VTreeNodeIndex, VTreeNodeIndex>,
      max_map: &mut HashMap<VTreeNodeIndex, VTreeNodeIndex>,
    ) {
      match &tree[node_id] {
        VTreeNode::Branch { left, right } => {
          cache_min_max_in_subtree(tree, left.0, min_map, max_map);
          cache_min_max_in_subtree(tree, right.0, min_map, max_map);
          min_map.insert(VTreeNodeIndex(node_id), min_map[left]);
          max_map.insert(VTreeNodeIndex(node_id), max_map[right]);
        }
        VTreeNode::Leaf { .. } => {
          min_map.insert(VTreeNodeIndex(node_id), VTreeNodeIndex(node_id));
          max_map.insert(VTreeNodeIndex(node_id), VTreeNodeIndex(node_id));
        }
      }
    }
    let mut first_in_subtree_map = HashMap::new();
    let mut last_in_subtree_map = HashMap::new();
    cache_min_max_in_subtree(
      &tree,
      root,
      &mut first_in_subtree_map,
      &mut last_in_subtree_map,
    );

    Self {
      tree,
      vars,
      root,
      var_to_node_id_map,
      in_order_positions,
      first_in_subtree_map,
      last_in_subtree_map,
    }
  }

  pub fn leaf(&self, node: VTreeNodeIndex) -> Option<usize> {
    match &self.tree[node.0] {
      VTreeNode::Leaf { var_id } => Some(*var_id),
      _ => None,
    }
  }

  pub fn left(&self, node: VTreeNodeIndex) -> Option<VTreeNodeIndex> {
    match &self.tree[node.0] {
      VTreeNode::Leaf { .. } => None,
      VTreeNode::Branch { left, .. } => Some(left.clone()),
    }
  }

  pub fn right(&self, node: VTreeNodeIndex) -> Option<VTreeNodeIndex> {
    match &self.tree[node.0] {
      VTreeNode::Leaf { .. } => None,
      VTreeNode::Branch { right, .. } => Some(right.clone()),
    }
  }

  /// A node is a decomposition node if
  /// 1. it is an internal node
  /// 2. its left branch is also an internal node
  pub fn is_decomposition(&self, node: VTreeNodeIndex) -> bool {
    match &self.tree[node.0] {
      VTreeNode::Leaf { .. } => false,
      VTreeNode::Branch { left, .. } => match &self.tree[left.0] {
        VTreeNode::Leaf { .. } => false,
        _ => true,
      },
    }
  }

  pub fn shannon_variable(&self, node: VTreeNodeIndex) -> Option<usize> {
    match &self.tree[node.0] {
      VTreeNode::Leaf { .. } => None,
      VTreeNode::Branch { left, .. } => match &self.tree[left.0] {
        VTreeNode::Leaf { var_id } => Some(var_id.clone()),
        _ => None,
      },
    }
  }

  pub fn position(&self, node: VTreeNodeIndex) -> usize {
    self.in_order_positions[&node]
  }

  pub fn first_in_subtree(&self, node: VTreeNodeIndex) -> VTreeNodeIndex {
    self.first_in_subtree_map[&node]
  }

  pub fn last_in_subtree(&self, node: VTreeNodeIndex) -> VTreeNodeIndex {
    self.last_in_subtree_map[&node]
  }

  pub fn parent(&self, node: VTreeNodeIndex) -> Option<VTreeNodeIndex> {
    // Should be at most one of them
    self
      .tree
      .neighbors_directed(node.0, Direction::Incoming)
      .next()
      .map(VTreeNodeIndex)
  }

  pub fn lowest_common_ancestor(
    &self,
    n1: VTreeNodeIndex,
    n2: VTreeNodeIndex,
  ) -> (AncestorType, VTreeNodeIndex) {
    if n1 == n2 {
      return (AncestorType::Equal, n1);
    }

    let p1 = self.position(n1);
    let p2 = self.position(n2);
    assert!(p1 < p2, "n1 should be to the left of n2");
    if p1 >= self.position(self.first_in_subtree(n2)) {
      (AncestorType::Left, n2)
    } else if p2 <= self.position(self.last_in_subtree(n1)) {
      (AncestorType::Right, n1)
    } else {
      let mut lca = self.parent(n1).unwrap(); // unwrap because we know n1 and n2 has common ancestor
      while p2 > self.position(self.last_in_subtree(lca)) {
        lca = self.parent(lca).unwrap();
      }
      (AncestorType::Disjoint, lca)
    }
  }

  pub fn num_vars(&self) -> usize {
    self.vars.len()
  }

  pub fn root_id(&self) -> VTreeNodeIndex {
    VTreeNodeIndex(self.root)
  }

  pub fn root_node(&self) -> &VTreeNode {
    &self.tree[self.root]
  }

  pub fn is_root(&self, node: VTreeNodeIndex) -> bool {
    self.root == node.0
  }

  pub fn branch_nodes(&self) -> Vec<VTreeNodeIndex> {
    self
      .tree
      .node_indices()
      .filter_map(|node_id| {
        if self.tree[node_id].is_branch() {
          Some(VTreeNodeIndex(node_id))
        } else {
          None
        }
      })
      .collect::<Vec<_>>()
  }

  /// Rotate left mutation given a node `x`, as depicted in the graph below
  ///
  /// ``` txt
  ///       |                   |
  ///       y                   x
  ///     /   \               /   \
  ///    x     c    <====    a     y
  ///   / \                      /  \
  ///  a   b                    b    c
  /// ```
  pub fn rotate_left(&mut self, x: VTreeNodeIndex) -> Result<(), VTreeMutationError> {
    // First gather information about the related nodes
    let (_a, y) = match self.tree[x.0] {
      VTreeNode::Branch { left, right } => (left, right),
      _ => return Err(VTreeMutationError::NodeIsNotBranch),
    };
    let (b, _c) = match self.tree[y.0] {
      VTreeNode::Branch { left, right } => (left, right),
      _ => return Err(VTreeMutationError::LeftNodeIsNotBranch),
    };

    // Modify the connectivities
    match &mut self.tree[x.0] {
      VTreeNode::Branch { right, .. } => {
        *right = b;
      }
      _ => panic!("Should not happen"),
    }
    match &mut self.tree[y.0] {
      VTreeNode::Branch { left, .. } => {
        *left = x;
      }
      _ => panic!("Should not happen"),
    }

    // Modify the graph edges
    let y_to_b = self.tree.find_edge(y.0, b.0).unwrap(); // unwrap because there has to be an edge
    let x_to_y = self.tree.find_edge(x.0, y.0).unwrap(); // unwrap because there has to be an edge
    self.tree.remove_edge(y_to_b);
    self.tree.remove_edge(x_to_y);
    self.tree.add_edge(y.0, x.0, ());
    self.tree.add_edge(x.0, b.0, ());

    // Update the parent
    if self.is_root(x) {
      self.root = y.0;
    } else {
      let parent = self.parent(x).unwrap(); // Since y is not root, y has a parent
      let parent_to_x = self.tree.find_edge(parent.0, x.0).unwrap(); // There has to be an edge
      self.tree.remove_edge(parent_to_x);
      self.tree.add_edge(parent.0, y.0, ());
    }

    Ok(())
  }

  /// Rotate right mutation given a node `y`, as depicted in the graph below
  ///
  /// ``` txt
  ///       |                   |
  ///       y                   x
  ///     /   \               /   \
  ///    x     c    ====>    a     y
  ///   / \                      /  \
  ///  a   b                    b    c
  /// ```
  pub fn rotate_right(&mut self, y: VTreeNodeIndex) -> Result<(), VTreeMutationError> {
    // First gather information about the related nodes
    let (x, _c) = match self.tree[y.0] {
      VTreeNode::Branch { left, right } => (left, right),
      _ => return Err(VTreeMutationError::NodeIsNotBranch),
    };
    let (_a, b) = match self.tree[x.0] {
      VTreeNode::Branch { left, right } => (left, right),
      _ => return Err(VTreeMutationError::LeftNodeIsNotBranch),
    };

    // Modify the connectivities
    match &mut self.tree[y.0] {
      VTreeNode::Branch { left, .. } => {
        *left = b;
      }
      _ => panic!("Should not happen"),
    }
    match &mut self.tree[x.0] {
      VTreeNode::Branch { right, .. } => {
        *right = y;
      }
      _ => panic!("Should not happen"),
    }

    // Modify the graph edges
    let x_to_b = self.tree.find_edge(x.0, b.0).unwrap(); // unwrap because there has to be an edge
    let y_to_x = self.tree.find_edge(y.0, x.0).unwrap(); // unwrap because there has to be an edge
    self.tree.remove_edge(x_to_b);
    self.tree.remove_edge(y_to_x);
    self.tree.add_edge(x.0, y.0, ());
    self.tree.add_edge(y.0, b.0, ());

    // Update the parent
    if self.is_root(y) {
      self.root = x.0;
    } else {
      let parent = self.parent(y).unwrap(); // Since y is not root, y has a parent
      let parent_to_y = self.tree.find_edge(parent.0, y.0).unwrap(); // There has to be an edge
      self.tree.remove_edge(parent_to_y);
      self.tree.add_edge(parent.0, x.0, ());
    }

    Ok(())
  }

  /// Swap the left and right node
  ///
  /// ``` txt
  ///    x              x
  ///  /   \   ===>   /   \
  /// a     b        b     a
  /// ```
  pub fn swap(&mut self, node_id: VTreeNodeIndex) -> Result<(), VTreeMutationError> {
    match &mut self.tree[node_id.0] {
      VTreeNode::Branch { left, right } => {
        let tmp = *left;
        *left = *right;
        *right = tmp;
        Ok(())
      }
      VTreeNode::Leaf { .. } => Err(VTreeMutationError::NodeIsNotBranch),
    }
  }

  pub fn dot(&self) -> String {
    fn traverse(
      nodes: &Graph<VTreeNode, ()>,
      curr_node: NodeIndex,
      node_strs: &mut Vec<String>,
      edge_strs: &mut Vec<String>,
    ) {
      let node_id = curr_node.index();
      match &nodes[curr_node.clone()] {
        VTreeNode::Leaf { var_id } => {
          let node_str = format!("{} [label=\"V{}\"]", node_id, var_id);
          node_strs.push(node_str);
        }
        VTreeNode::Branch { left, right } => {
          let node_str = format!("{} [label=\"{}\", shape=circle];", node_id, node_id);
          node_strs.push(node_str);

          edge_strs.push(format!("{} -> {};", node_id, left.index()));
          edge_strs.push(format!("{} -> {};", node_id, right.index()));

          traverse(nodes, left.0, node_strs, edge_strs);
          traverse(nodes, right.0, node_strs, edge_strs);
        }
      }
    }

    let mut node_strs = vec![];
    let mut edge_strs = vec![];

    traverse(
      &self.tree,
      self.root.clone(),
      &mut node_strs,
      &mut edge_strs,
    );

    format!(
      "digraph vtree {{ node [shape=record margin=0.03 width=0 height=0]; {} {} }}",
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

#[derive(Debug, Clone)]
pub enum VTreeMutationError {
  NodeIsNotBranch,
  RightNodeIsNotBranch,
  LeftNodeIsNotBranch,
}
