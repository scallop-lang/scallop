use std::collections::*;

use itertools::Itertools;

use super::*;

#[derive(Clone, Debug)]
pub struct QueryPlanContext<'a> {
  pub head_vars: HashSet<Variable>,
  pub reduces: Vec<Reduce>,
  pub pos_atoms: Vec<Atom>,
  pub neg_atoms: Vec<NegAtom>,
  pub assigns: Vec<Assign>,
  pub constraints: Vec<Constraint>,
  pub stratum: &'a Stratum,
}

impl<'a> QueryPlanContext<'a> {
  pub fn from_rule(stratum: &'a Stratum, rule: &Rule) -> Self {
    let mut ctx = Self {
      head_vars: rule.head.variable_args().cloned().collect(),
      reduces: vec![],
      pos_atoms: vec![],
      neg_atoms: vec![],
      assigns: vec![],
      constraints: vec![],
      stratum,
    };
    for literal in rule.body_literals() {
      match literal {
        Literal::Reduce(r) => ctx.reduces.push(r.clone()),
        Literal::Atom(a) => ctx.pos_atoms.push(a.clone()),
        Literal::NegAtom(n) => ctx.neg_atoms.push(n.clone()),
        Literal::Assign(a) => ctx.assigns.push(a.clone()),
        Literal::Constraint(c) => ctx.constraints.push(c.clone()),
        Literal::True => {}
        Literal::False => {
          panic!("[Internal Error] Rule with false literal not removed")
        }
      }
    }
    ctx
  }

  pub fn bounded_args_from_pos_atoms_set(&self, set: &Vec<&usize>) -> HashSet<Variable> {
    let mut base_bounded_args = HashSet::new();
    for atom in self
      .pos_atoms
      .iter()
      .enumerate()
      .filter_map(|(i, m)| if set.contains(&&i) { Some(m) } else { None })
    {
      for var in atom.args.iter().filter_map(|a| match a {
        Term::Variable(a) => Some(a),
        _ => None,
      }) {
        base_bounded_args.insert(var.clone());
      }
    }

    // Fix point iteration
    loop {
      let mut new_bounded_args = base_bounded_args.clone();
      for assign in &self.assigns {
        let can_bound = match &assign.right {
          AssignExpr::Binary(b) => {
            let op1_bounded = term_is_bounded(&new_bounded_args, &b.op1);
            let op2_bounded = term_is_bounded(&new_bounded_args, &b.op1);
            op1_bounded && op2_bounded
          }
          AssignExpr::Unary(u) => term_is_bounded(&new_bounded_args, &u.op1),
          AssignExpr::IfThenElse(i) => {
            let cond_bounded = term_is_bounded(&new_bounded_args, &i.cond);
            let then_br_bounded = term_is_bounded(&new_bounded_args, &i.then_br);
            let else_br_bounded = term_is_bounded(&new_bounded_args, &i.else_br);
            cond_bounded && then_br_bounded && else_br_bounded
          }
          AssignExpr::Call(c) => c.args.iter().all(|a| term_is_bounded(&new_bounded_args, a)),
        };
        if can_bound {
          new_bounded_args.insert(assign.left.clone());
        }
      }

      if new_bounded_args == base_bounded_args {
        break new_bounded_args;
      } else {
        base_bounded_args = new_bounded_args;
      }
    }
  }

  fn pos_atom_arcs(&self, beam_size: usize) -> State {
    if self.pos_atoms.is_empty() {
      return State::new();
    }

    let mut priority_queue = BinaryHeap::new();
    priority_queue.push(State::new());
    let mut final_states = BinaryHeap::new();
    while !priority_queue.is_empty() {
      let mut temp_queue = BinaryHeap::new();

      // Find the next states and push them into the queue
      while !priority_queue.is_empty() {
        let top = priority_queue.pop().unwrap();
        for next_state in top.next_states(self, beam_size) {
          if next_state.bounded_all(self) {
            final_states.push(next_state);
          } else {
            temp_queue.push(next_state);
          }
        }
      }

      // Retain only top `beam_size` inside of the priority_queue
      for _ in 0..beam_size {
        if let Some(elem) = temp_queue.pop() {
          priority_queue.push(elem);
        } else {
          break;
        }
      }
    }

    // At the end, use the best state in `final_states`
    final_states.pop().unwrap()
  }

  /// The main entry function that computes a query plan from a sequence of arcs
  fn get_query_plan(&self, arcs: &Vec<Arc>) -> Plan {
    // Stage 1: Helper Functions (Closures)

    // Store the applied constraints
    let mut applied_constraints = HashSet::new();
    let mut try_apply_constraint = |fringe: Plan| -> Plan {
      // Apply as many constraints as possible
      let node = fringe;
      let mut to_apply_constraints = vec![];
      for (i, constraint) in self.constraints.iter().enumerate() {
        if !applied_constraints.contains(&i) {
          let can_apply = constraint.variable_args().iter().all(|v| node.bounded_vars.contains(v));
          if can_apply {
            applied_constraints.insert(i);
            to_apply_constraints.push(constraint.clone());
          }
        }
      }
      if to_apply_constraints.is_empty() {
        node
      } else {
        let new_bounded_vars = node.bounded_vars.clone();

        Plan {
          bounded_vars: new_bounded_vars,
          ram_node: HighRamNode::Filter(Box::new(node), to_apply_constraints),
        }
      }
    };

    // Store the applied assigns
    let mut applied_assigns = HashSet::new();
    let mut try_apply_assigns = |mut fringe: Plan| -> Plan {
      // Find all the assigns that are needed to bound the need_projection_vars
      let mut bounded_vars = fringe.bounded_vars.clone();
      loop {
        let mut new_projections = Vec::new();

        // Check if we can apply more assigns
        for (i, assign) in self.assigns.iter().enumerate() {
          if !applied_assigns.contains(&i)
            && !bounded_vars.contains(&assign.left)
            && assign.variable_args().into_iter().all(|v| bounded_vars.contains(v))
          {
            applied_assigns.insert(i);
            new_projections.push(assign.clone());
          }
        }

        // Create projected left node
        if new_projections.is_empty() {
          break fringe;
        } else {
          bounded_vars.extend(new_projections.iter().map(|i| i.left.clone()));
          fringe = Plan {
            bounded_vars: bounded_vars.clone(),
            ram_node: HighRamNode::Project(Box::new(fringe), new_projections),
          };
        }
      }
    };

    // Stage 2: Building the RAM tree bottom-up, starting with reduces

    // Build the first fringe
    let (mut fringe, start_id) = if self.reduces.is_empty() {
      if arcs.is_empty() {
        let node = Plan {
          bounded_vars: HashSet::new(),
          ram_node: HighRamNode::Unit,
        };
        (node, 0)
      } else {
        // If there is no reduce, get it from the first arc
        let first_arc = &arcs[0];
        let node = Plan {
          bounded_vars: self.pos_atoms[first_arc.right].variable_args().cloned().collect(),
          ram_node: HighRamNode::Ground(self.pos_atoms[first_arc.right].clone()),
        };

        // Note: We always apply constraint first and then assigns
        (try_apply_constraint(try_apply_assigns(node)), 1)
      }
    } else {
      // If there is reduce, create a joined reduce
      let first_reduce = &self.reduces[0];
      let mut node = Plan {
        bounded_vars: first_reduce.variable_args().cloned().collect(),
        ram_node: HighRamNode::Reduce(first_reduce.clone()),
      };

      // Get more reduces
      for reduce in &self.reduces[1..] {
        let left = node;
        let right_bounded_vars = reduce.variable_args().cloned().collect::<HashSet<_>>();
        let right = Plan {
          bounded_vars: right_bounded_vars.clone(),
          ram_node: HighRamNode::Reduce(reduce.clone()),
        };
        node = Plan {
          bounded_vars: left.bounded_vars.union(&right_bounded_vars).cloned().collect(),
          ram_node: HighRamNode::Join(Box::new(left), Box::new(right)),
        };
      }

      // Note: We always apply constraint first and then assigns
      (try_apply_constraint(try_apply_assigns(node)), 0)
    };

    // Stage 3. Iterate through all the arcs, build the tree from bottom-up
    for arc in &arcs[start_id..] {
      // Build the simple tree
      if arc.left.is_empty() {
        // A node that is not related to any of the node before; need product
        let left = fringe;
        let right = Plan {
          bounded_vars: self.pos_atoms[arc.right].variable_args().cloned().collect(),
          ram_node: HighRamNode::Ground(self.pos_atoms[arc.right].clone()),
        };
        let new_bounded_vars = left.bounded_vars.union(&right.bounded_vars).cloned().collect();
        let new_ram_node = HighRamNode::Join(Box::new(left), Box::new(right));
        fringe = Plan {
          bounded_vars: new_bounded_vars,
          ram_node: new_ram_node,
        };
      } else {
        let left = fringe;

        // Create right node
        let right_bounded_vars = self.pos_atoms[arc.right]
          .variable_args()
          .cloned()
          .collect::<HashSet<_>>();
        let right = Plan {
          bounded_vars: right_bounded_vars,
          ram_node: HighRamNode::Ground(self.pos_atoms[arc.right].clone()),
        };

        // Create joined node
        fringe = Plan {
          bounded_vars: left.bounded_vars.union(&right.bounded_vars).cloned().collect(),
          ram_node: HighRamNode::Join(Box::new(left), Box::new(right)),
        };
      }

      // Note: We always apply constraint first and then assigns
      fringe = try_apply_constraint(try_apply_assigns(fringe));
    }

    // Apply negative atoms
    for neg_atom in &self.neg_atoms {
      let neg_node = Plan {
        bounded_vars: neg_atom.atom.variable_args().cloned().collect(),
        ram_node: HighRamNode::Ground(neg_atom.atom.clone()),
      };
      fringe = Plan {
        bounded_vars: fringe.bounded_vars.clone(),
        ram_node: HighRamNode::Antijoin(Box::new(fringe), Box::new(neg_node)),
      };
    }

    fringe
  }

  pub fn query_plan(&self) -> Plan {
    let beam_size = 5;
    let state = self.pos_atom_arcs(beam_size);
    self.get_query_plan(&state.arcs)
  }
}

fn term_is_bounded(bounded_vars: &HashSet<Variable>, term: &Term) -> bool {
  match term {
    Term::Variable(v) => bounded_vars.contains(v),
    _ => true,
  }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct State {
  visited_atoms: Vec<usize>,
  arcs: Vec<Arc>,
}

impl std::cmp::PartialOrd for State {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.aggregated_weight().partial_cmp(&other.aggregated_weight())
  }
}

impl std::cmp::Ord for State {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.aggregated_weight().cmp(&other.aggregated_weight())
  }
}

impl State {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn aggregated_weight(&self) -> i32 {
    self.arcs.iter().map(|a| a.weight()).sum()
  }

  pub fn next_states(&self, ctx: &QueryPlanContext, beam_size: usize) -> Vec<Self> {
    if self.visited_atoms.len() >= ctx.pos_atoms.len() {
      vec![]
    } else {
      let mut next_states: Vec<Self> = vec![];
      for (id, atom) in (0..ctx.pos_atoms.len())
        .filter(|i| !self.visited_atoms.contains(i))
        .map(|i| (i, &ctx.pos_atoms[i]))
      {
        for set in self.visited_atoms.iter().powerset() {
          let all_bounded_args = ctx.bounded_args_from_pos_atoms_set(&set);
          let bounded_vars = atom
            .variable_args()
            .filter(|a| all_bounded_args.contains(a))
            .cloned()
            .collect();
          let is_edb = !ctx.stratum.predicates.contains(&atom.predicate);
          let arc = Arc {
            left: set.iter().map(|i| **i).collect::<Vec<_>>(),
            right: id,
            bounded_vars,
            is_edb,
          };
          next_states.push(State {
            visited_atoms: vec![self.visited_atoms.clone(), vec![id]].concat(),
            arcs: vec![self.arcs.clone(), vec![arc]].concat(),
          });
        }
      }
      next_states.sort_by_key(|s| -s.aggregated_weight());
      next_states[0..beam_size.min(next_states.len())].to_vec()
    }
  }

  pub fn bounded_all(&self, ctx: &QueryPlanContext) -> bool {
    self.visited_atoms.len() == ctx.pos_atoms.len()
  }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Arc {
  left: Vec<usize>,
  right: usize,
  bounded_vars: HashSet<Variable>,
  is_edb: bool,
}

impl Arc {
  pub fn weight(&self) -> i32 {
    let num_bounded_vars = self.bounded_vars.len() as i32;
    let edb_weight = if self.left.is_empty() && self.is_edb { 1 } else { 0 };
    num_bounded_vars + edb_weight
  }
}

#[derive(Clone, Debug)]
pub struct Plan {
  pub bounded_vars: HashSet<Variable>,
  pub ram_node: HighRamNode,
}

impl Plan {
  pub fn pretty_print(&self) {
    self.pretty_print_helper(0);
  }

  fn pretty_print_helper(&self, depth: usize) {
    let prefix = vec!["  "; depth].join("");
    match &self.ram_node {
      HighRamNode::Unit => {
        println!("Unit");
      }
      HighRamNode::Reduce(r) => {
        println!("{}Reduce {{{}}}", prefix, r);
      }
      HighRamNode::Ground(a) => {
        println!("{}{}", prefix, a);
      }
      HighRamNode::Project(x, y) => {
        println!(
          "{}Project {{{}}}",
          prefix,
          y.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join(", ")
        );
        x.pretty_print_helper(depth + 1);
      }
      HighRamNode::Filter(x, y) => {
        println!(
          "{}Filter {{{}}}",
          prefix,
          y.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join(", ")
        );
        x.pretty_print_helper(depth + 1);
      }
      HighRamNode::Join(x, y) => {
        println!("{}Join", prefix);
        x.pretty_print_helper(depth + 1);
        y.pretty_print_helper(depth + 1);
      }
      HighRamNode::Antijoin(x, y) => {
        println!("{}Antijoin", prefix);
        x.pretty_print_helper(depth + 1);
        y.pretty_print_helper(depth + 1);
      }
    }
  }
}

#[derive(Clone, Debug)]
pub enum HighRamNode {
  Unit,
  Reduce(Reduce),
  Ground(Atom),
  Project(Box<Plan>, Vec<Assign>),
  Filter(Box<Plan>, Vec<Constraint>),
  Join(Box<Plan>, Box<Plan>),
  Antijoin(Box<Plan>, Box<Plan>),
}

impl HighRamNode {
  pub fn direct_atom(&self) -> Option<&Atom> {
    match self {
      Self::Ground(a) => Some(a),
      Self::Filter(d, _) => d.ram_node.direct_atom(),
      _ => None,
    }
  }
}
