use std::collections::*;

use crate::common::foreign_predicate::*;
use crate::compiler::back;
use crate::compiler::front::analyzers::*;
use crate::compiler::front::utils::*;
use crate::compiler::front::*;
use crate::utils::IdAllocator;

pub struct FlattenExprContext<'a> {
  pub type_inference: &'a TypeInference,
  pub foreign_predicate_registry: &'a ForeignPredicateRegistry,
  pub id_allocator: IdAllocator,
  pub ignore_exprs: HashSet<Loc>,
  pub internal: HashMap<Loc, FlattenedNode>,
  pub leaf: HashMap<Loc, FlattenedLeaf>,
}

#[derive(Clone, Debug)]
pub enum FlattenedNode {
  Binary {
    left: back::Variable,
    op: back::BinaryExprOp,
    op1: Loc,
    op2: Loc,
  },
  Unary {
    left: back::Variable,
    op: back::UnaryExprOp,
    op1: Loc,
  },
  IfThenElse {
    left: back::Variable,
    cond: Loc,
    then_br: Loc,
    else_br: Loc,
  },
  Call {
    left: back::Variable,
    function: String,
    args: Vec<Loc>,
  },
  New {
    left: back::Variable,
    functor: String,
    args: Vec<Loc>,
  },
}

impl FlattenedNode {
  pub fn back_var(&self) -> back::Variable {
    match self {
      Self::Binary { left, .. }
      | Self::Unary { left, .. }
      | Self::IfThenElse { left, .. }
      | Self::Call { left, .. }
      | Self::New { left, .. } => left.clone(),
    }
  }
}

#[doc(hidden)]
pub type FlattenedLeaf = back::Term;

impl<'a> FlattenExprContext<'a> {
  pub fn new(type_inference: &'a TypeInference, foreign_predicate_registry: &'a ForeignPredicateRegistry) -> Self {
    Self {
      type_inference,
      foreign_predicate_registry,
      id_allocator: IdAllocator::default(),
      ignore_exprs: HashSet::new(),
      internal: HashMap::new(),
      leaf: HashMap::new(),
    }
  }

  pub fn allocate_tmp_var(&mut self) -> String {
    format!("flat#{}", self.id_allocator.alloc())
  }

  pub fn allocate_wildcard_var(&mut self) -> String {
    format!("wc#{}", self.id_allocator.alloc())
  }

  pub fn get_loc_term(&self, loc: &Loc) -> back::Term {
    if let Some(node) = self.internal.get(loc) {
      back::Term::Variable(node.back_var())
    } else if let Some(leaf) = self.leaf.get(loc) {
      leaf.clone()
    } else {
      panic!(
        "[Internal Error] Cannot find loc {:?} from the context, should not happen",
        loc
      )
    }
  }

  pub fn get_expr_term(&self, expr: &Expr) -> back::Term {
    let loc = expr.location();
    self.get_loc_term(loc)
  }

  pub fn collect_flattened_literals(&self, expr_loc: &Loc) -> Vec<back::Literal> {
    if let Some(node) = self.internal.get(expr_loc) {
      match node {
        FlattenedNode::Binary { left, op, op1, op2 } => {
          self.collect_flattened_literals_of_binary_op(left, op, op1, op2)
        }
        FlattenedNode::Unary { left, op, op1 } => self.collect_flattened_literals_of_unary_op(left, op, op1),
        FlattenedNode::IfThenElse {
          left,
          cond,
          then_br,
          else_br,
        } => self.collect_flattened_literals_of_if_then_else_op(left, cond, then_br, else_br),
        FlattenedNode::Call { left, function, args } => {
          self.collect_flattened_literals_of_call_op(left, function, args)
        }
        FlattenedNode::New { left, functor, args } => self.collect_flattened_literals_of_new_op(left, functor, args),
      }
    } else {
      vec![]
    }
  }

  pub fn collect_flattened_literals_of_binary_op(
    &self,
    left: &back::Variable,
    op: &back::BinaryExprOp,
    op1: &Loc,
    op2: &Loc,
  ) -> Vec<back::Literal> {
    let mut curr_literals = vec![];

    // First generate the `left = op1 (*) op2` literal
    let op1_term = self.get_loc_term(op1);
    let op2_term = self.get_loc_term(op2);
    let literal = back::Literal::binary_expr(left.clone(), op.clone(), op1_term.clone(), op2_term.clone());
    curr_literals.push(literal);

    // For +/-, generate op1 = left (-/+) op2, if op1 is a variable
    if op.is_add_sub() {
      let inv_op = op.add_sub_inv_op().unwrap();
      if let back::Term::Variable(op1_var) = &op1_term {
        let op1_literal = back::Literal::binary_expr(
          op1_var.clone(),
          inv_op.clone(),
          back::Term::Variable(left.clone()),
          op2_term.clone(),
        );
        curr_literals.push(op1_literal);
      }
      if let back::Term::Variable(op2_var) = &op2_term {
        let op2_literal =
          back::Literal::binary_expr(op2_var.clone(), inv_op, back::Term::Variable(left.clone()), op1_term);
        curr_literals.push(op2_literal);
      }
    }

    // Collect flattened literals from op1 and op2
    curr_literals.extend(self.collect_flattened_literals(op1));
    curr_literals.extend(self.collect_flattened_literals(op2));

    // Return all of them
    curr_literals
  }

  pub fn collect_flattened_literals_of_unary_op(
    &self,
    left: &back::Variable,
    op: &back::UnaryExprOp,
    op1: &Loc,
  ) -> Vec<back::Literal> {
    let mut curr_literals = vec![];

    // The unary expression literal
    let op1_term = self.get_loc_term(op1);
    let literal = back::Literal::unary_expr(left.clone(), op.clone(), op1_term);
    curr_literals.push(literal);

    // Collect flattend literals from op1
    curr_literals.extend(self.collect_flattened_literals(op1));

    // Return all of them
    curr_literals
  }

  pub fn collect_flattened_literals_of_if_then_else_op(
    &self,
    left: &back::Variable,
    cond: &Loc,
    then_br: &Loc,
    else_br: &Loc,
  ) -> Vec<back::Literal> {
    let mut curr_literals = vec![];

    // The if-then-else expression literal
    let cond_term = self.get_loc_term(cond);
    let then_br_term = self.get_loc_term(then_br);
    let else_br_term = self.get_loc_term(else_br);
    let literal = back::Literal::if_then_else_expr(left.clone(), cond_term, then_br_term, else_br_term);
    curr_literals.push(literal);

    // Collect flattened literals from cond, then_br, and else_br
    curr_literals.extend(self.collect_flattened_literals(cond));
    curr_literals.extend(self.collect_flattened_literals(then_br));
    curr_literals.extend(self.collect_flattened_literals(else_br));

    // Return all of them
    curr_literals
  }

  pub fn collect_flattened_literals_of_call_op(
    &self,
    left: &back::Variable,
    function: &String,
    args: &Vec<Loc>,
  ) -> Vec<back::Literal> {
    let mut curr_literals = vec![];

    // The call expression literal
    let arg_terms = args.iter().map(|a| self.get_loc_term(a)).collect::<Vec<_>>();
    let literal = back::Literal::call_expr(left.clone(), function.clone(), arg_terms);
    curr_literals.push(literal);

    // Collect flattened literals from args
    for arg in args {
      curr_literals.extend(self.collect_flattened_literals(arg));
    }

    // Return all of them
    curr_literals
  }

  pub fn collect_flattened_literals_of_new_op(
    &self,
    left: &back::Variable,
    functor: &String,
    args: &Vec<Loc>,
  ) -> Vec<back::Literal> {
    let mut curr_literals = vec![];

    // The call expression literal
    let arg_terms = args.iter().map(|a| self.get_loc_term(a)).collect::<Vec<_>>();
    let literal = back::Literal::new_expr(left.clone(), functor.clone(), arg_terms);
    curr_literals.push(literal);

    // Collect flattened literals from args
    for arg in args {
      curr_literals.extend(self.collect_flattened_literals(arg));
    }

    // Return all of them
    curr_literals
  }

  pub fn atom_to_back_literals(&self, atom: &Atom) -> Vec<back::Literal> {
    let mut literals = vec![];

    // First get the atom
    let back_atom_args = atom.iter_args().map(|a| self.get_expr_term(a)).collect();
    let back_atom = back::Atom {
      predicate: atom.formatted_predicate().clone(),
      args: back_atom_args,
    };

    // Depending on whether the atom is foreign, add the literal differently
    let back_literal = back::Literal::Atom(back_atom);
    literals.push(back_literal);

    // Then collect all the intermediate variables
    for arg in atom.iter_args() {
      literals.extend(self.collect_flattened_literals(arg.location()));
    }

    literals
  }

  pub fn neg_atom_to_back_literals(&self, neg_atom: &NegAtom) -> Vec<back::Literal> {
    let mut literals = vec![];

    // First get the atom
    let back_atom_args = neg_atom.atom().iter_args().map(|a| self.get_expr_term(a)).collect();
    let back_atom = back::NegAtom {
      atom: back::Atom {
        predicate: neg_atom.atom().formatted_predicate().clone(),
        args: back_atom_args,
      },
    };

    // Then generate a literal
    let back_literal = back::Literal::NegAtom(back_atom);
    literals.push(back_literal);

    // Then collect all the intermediate variables
    for arg in neg_atom.atom().iter_args() {
      literals.extend(self.collect_flattened_literals(arg.location()));
    }

    literals
  }

  pub fn binary_constraint_to_back_literal(&self, b: &BinaryExpr) -> Vec<back::Literal> {
    if let Some(op) = Option::<back::BinaryConstraintOp>::from(b.op().op()) {
      let mut curr_literals = vec![];

      // First generate the `op1 (<=>) op2` constraint
      let op1_term = self.get_expr_term(b.op1());
      let op2_term = self.get_expr_term(b.op2());
      let literal = back::Literal::binary_constraint(op, op1_term, op2_term);
      curr_literals.push(literal);

      // Collect flattened literals from op1 and op2
      curr_literals.extend(self.collect_flattened_literals(b.op1().location()));
      curr_literals.extend(self.collect_flattened_literals(b.op2().location()));

      curr_literals
    } else {
      panic!("[Internal Error] Cannot use `{}` for binary constraint", b.op().op());
    }
  }

  pub fn unary_constraint_to_back_literal(&self, u: &UnaryExpr) -> Vec<back::Literal> {
    if let Some(op) = Option::<back::UnaryConstraintOp>::from(u.op().internal()) {
      let mut curr_literals = vec![];

      // First generate the `(*) op1` constraint
      let op1_term = self.get_expr_term(u.op1());
      let literal = back::Literal::unary_constraint(op, op1_term);
      curr_literals.push(literal);

      // Collect flattened literals from op1
      curr_literals.extend(self.collect_flattened_literals(u.op1().location()));

      curr_literals
    } else {
      panic!(
        "[Internal Error] Cannot use `{}` for unary constraint",
        u.op().internal()
      );
    }
  }

  pub fn constraint_to_back_literal(&self, c: &Constraint) -> Vec<back::Literal> {
    match c.expr() {
      Expr::Binary(b) => self.binary_constraint_to_back_literal(b),
      Expr::Unary(u) => self.unary_constraint_to_back_literal(u),
      _ => {
        panic!("[Internal Error] Cannot have non-binary/non-unary constraint expression");
      }
    }
  }

  pub fn front_literal_to_back_literal(&self, f: &Formula) -> Vec<back::Literal> {
    match f {
      Formula::Atom(atom) => self.atom_to_back_literals(atom),
      Formula::NegAtom(neg_atom) => self.neg_atom_to_back_literals(neg_atom),
      Formula::Constraint(c) => self.constraint_to_back_literal(c),
      Formula::Case(_)
      | Formula::Conjunction(_)
      | Formula::Disjunction(_)
      | Formula::Implies(_)
      | Formula::Reduce(_)
      | Formula::ForallExistsReduce(_)
      | Formula::Range(_) => {
        panic!("[Internal Error] Should not contain conjunction, disjunction, implies, reduce, or range");
      }
    }
  }

  pub fn to_back_literals(&self, conj: &Vec<Formula>) -> Vec<back::Literal> {
    conj
      .iter()
      .map(|f| self.front_literal_to_back_literal(f))
      .flatten()
      .collect()
  }
}

impl<'a> NodeVisitor<Constraint> for FlattenExprContext<'a> {
  fn visit(&mut self, constraint: &Constraint) {
    self.ignore_exprs.insert(constraint.expr().location().clone());
  }
}

impl<'a> NodeVisitor<BinaryExpr> for FlattenExprContext<'a> {
  fn visit(&mut self, b: &BinaryExpr) {
    if !self.ignore_exprs.contains(b.location()) {
      let tmp_var_name = self.allocate_tmp_var();
      self.internal.insert(
        b.location().clone(),
        FlattenedNode::Binary {
          left: back::Variable {
            name: tmp_var_name,
            ty: self.type_inference.expr_value_type(b).unwrap(),
          },
          op: b.op().op().clone(),
          op1: b.op1().location().clone(),
          op2: b.op2().location().clone(),
        },
      );
    }
  }
}

impl<'a> NodeVisitor<UnaryExpr> for FlattenExprContext<'a> {
  fn visit(&mut self, u: &UnaryExpr) {
    if !self.ignore_exprs.contains(u.location()) {
      let tmp_var_name = self.allocate_tmp_var();
      let op = match u.op().internal() {
        _UnaryOp::Neg => back::UnaryExprOp::Neg,
        _UnaryOp::Pos => back::UnaryExprOp::Pos,
        _UnaryOp::Not => back::UnaryExprOp::Not,
        _UnaryOp::TypeCast(t) => back::UnaryExprOp::TypeCast(self.type_inference.find_value_type(t).unwrap()),
      };
      self.internal.insert(
        u.location().clone(),
        FlattenedNode::Unary {
          left: back::Variable {
            name: tmp_var_name,
            ty: self.type_inference.expr_value_type(u).unwrap(),
          },
          op,
          op1: u.op1().location().clone(),
        },
      );
    }
  }
}

impl<'a> NodeVisitor<IfThenElseExpr> for FlattenExprContext<'a> {
  fn visit(&mut self, i: &IfThenElseExpr) {
    let tmp_var_name = self.allocate_tmp_var();
    self.internal.insert(
      i.location().clone(),
      FlattenedNode::IfThenElse {
        left: back::Variable {
          name: tmp_var_name,
          ty: self.type_inference.expr_value_type(i).unwrap(),
        },
        cond: i.cond().location().clone(),
        then_br: i.then_br().location().clone(),
        else_br: i.else_br().location().clone(),
      },
    );
  }
}

impl<'a> NodeVisitor<CallExpr> for FlattenExprContext<'a> {
  fn visit(&mut self, c: &CallExpr) {
    let tmp_var_name = self.allocate_tmp_var();
    let function = c.function_identifier().name().to_string();
    self.internal.insert(
      c.location().clone(),
      FlattenedNode::Call {
        left: back::Variable {
          name: tmp_var_name,
          ty: self.type_inference.expr_value_type(c).unwrap(),
        },
        function,
        args: c.iter_args().map(|a| a.location().clone()).collect(),
      },
    );
  }
}

impl<'a> NodeVisitor<NewExpr> for FlattenExprContext<'a> {
  fn visit(&mut self, n: &NewExpr) {
    let tmp_var_name = self.allocate_tmp_var();
    let functor = format!("adt#{}", n.functor_name());
    self.internal.insert(
      n.location().clone(),
      FlattenedNode::New {
        left: back::Variable {
          name: tmp_var_name,
          ty: self.type_inference.expr_value_type(n).unwrap(),
        },
        functor,
        args: n.iter_args().map(|a| a.location().clone()).collect(),
      },
    );
  }
}

impl<'a> NodeVisitor<Variable> for FlattenExprContext<'a> {
  fn visit(&mut self, v: &Variable) {
    let back_var = back::Variable {
      name: v.name().to_string(),
      ty: self.type_inference.expr_value_type(v).unwrap(),
    };
    self
      .leaf
      .insert(v.location().clone(), FlattenedLeaf::Variable(back_var));
  }
}

impl<'a> NodeVisitor<Wildcard> for FlattenExprContext<'a> {
  fn visit(&mut self, w: &Wildcard) {
    let wc_name = self.allocate_wildcard_var();
    let back_var = back::Variable {
      name: wc_name,
      ty: self.type_inference.expr_value_type(w).unwrap(),
    };
    self
      .leaf
      .insert(w.location().clone(), FlattenedLeaf::Variable(back_var));
  }
}

impl<'a> NodeVisitor<Constant> for FlattenExprContext<'a> {
  fn visit(&mut self, c: &Constant) {
    let ty = self.type_inference.expr_types[c.location()].to_default_value_type();
    self
      .leaf
      .insert(c.location().clone(), FlattenedLeaf::Constant(c.to_value(&ty)));
  }
}
