use super::ast;
use super::ast::*;

macro_rules! node_visitor_mut_func_def {
  ($func:ident, $node:ident) => {
    fn $func(&mut self, _: &mut ast::$node) {}
  };
}

pub trait NodeVisitorMut {
  fn visit_location(&mut self, _: &mut AstNodeLocation) {}

  node_visitor_mut_func_def!(visit_item, Item);
  node_visitor_mut_func_def!(visit_import_decl, ImportDecl);
  node_visitor_mut_func_def!(visit_import_file, ImportFile);
  node_visitor_mut_func_def!(visit_arg_type_binding, ArgTypeBinding);
  node_visitor_mut_func_def!(visit_type, Type);
  node_visitor_mut_func_def!(visit_alias_type_decl, AliasTypeDecl);
  node_visitor_mut_func_def!(visit_subtype_decl, SubtypeDecl);
  node_visitor_mut_func_def!(visit_relation_type_decl, RelationTypeDecl);
  node_visitor_mut_func_def!(visit_relation_type, RelationType);
  node_visitor_mut_func_def!(visit_enum_type_decl, EnumTypeDecl);
  node_visitor_mut_func_def!(visit_enum_type_member, EnumTypeMember);
  node_visitor_mut_func_def!(visit_const_decl, ConstDecl);
  node_visitor_mut_func_def!(visit_const_assignment, ConstAssignment);
  node_visitor_mut_func_def!(visit_relation_decl, RelationDecl);
  node_visitor_mut_func_def!(visit_constant_set_decl, ConstantSetDecl);
  node_visitor_mut_func_def!(visit_constant_set, ConstantSet);
  node_visitor_mut_func_def!(visit_constant_set_tuple, ConstantSetTuple);
  node_visitor_mut_func_def!(visit_constant_tuple, ConstantTuple);
  node_visitor_mut_func_def!(visit_constant_or_variable, ConstantOrVariable);
  node_visitor_mut_func_def!(visit_fact_decl, FactDecl);
  node_visitor_mut_func_def!(visit_rule_decl, RuleDecl);
  node_visitor_mut_func_def!(visit_query_decl, QueryDecl);
  node_visitor_mut_func_def!(visit_query, Query);
  node_visitor_mut_func_def!(visit_tag, Tag);
  node_visitor_mut_func_def!(visit_rule, Rule);
  node_visitor_mut_func_def!(visit_atom, Atom);
  node_visitor_mut_func_def!(visit_neg_atom, NegAtom);
  node_visitor_mut_func_def!(visit_attribute, Attribute);
  node_visitor_mut_func_def!(visit_formula, Formula);
  node_visitor_mut_func_def!(visit_conjunction, Conjunction);
  node_visitor_mut_func_def!(visit_disjunction, Disjunction);
  node_visitor_mut_func_def!(visit_implies, Implies);
  node_visitor_mut_func_def!(visit_constraint, Constraint);
  node_visitor_mut_func_def!(visit_reduce, Reduce);
  node_visitor_mut_func_def!(visit_forall_exists_reduce, ForallExistsReduce);
  node_visitor_mut_func_def!(visit_variable_binding, VariableBinding);
  node_visitor_mut_func_def!(visit_expr, Expr);
  node_visitor_mut_func_def!(visit_binary_expr, BinaryExpr);
  node_visitor_mut_func_def!(visit_unary_expr, UnaryExpr);
  node_visitor_mut_func_def!(visit_if_then_else_expr, IfThenElseExpr);
  node_visitor_mut_func_def!(visit_call_expr, CallExpr);
  node_visitor_mut_func_def!(visit_constant, Constant);
  node_visitor_mut_func_def!(visit_variable, Variable);
  node_visitor_mut_func_def!(visit_wildcard, Wildcard);
  node_visitor_mut_func_def!(visit_identifier, Identifier);
  node_visitor_mut_func_def!(visit_function_identifier, FunctionIdentifier);

  fn walk_items(&mut self, items: &mut Vec<Item>) {
    for item in items {
      self.walk_item(item);
    }
  }

  fn walk_item(&mut self, item: &mut Item) {
    self.visit_item(item);
    match item {
      Item::ImportDecl(id) => self.walk_import_decl(id),
      Item::TypeDecl(td) => self.walk_type_decl(td),
      Item::ConstDecl(cd) => self.walk_const_decl(cd),
      Item::RelationDecl(rd) => self.walk_relation_decl(rd),
      Item::QueryDecl(qd) => self.walk_query_decl(qd),
    }
  }

  fn walk_import_decl(&mut self, import_decl: &mut ImportDecl) {
    self.visit_import_decl(import_decl);
    self.visit_location(&mut import_decl.loc);
    self.walk_attributes(&mut import_decl.node.attrs);
    self.walk_import_file(&mut import_decl.node.import_file);
  }

  fn walk_import_file(&mut self, import_file: &mut ImportFile) {
    self.visit_import_file(import_file);
    self.visit_location(&mut import_file.loc);
  }

  fn walk_arg_type_binding(&mut self, arg_type_binding: &mut ArgTypeBinding) {
    self.visit_arg_type_binding(arg_type_binding);
    self.visit_location(&mut arg_type_binding.loc);
    self.walk_option_identifier(&mut arg_type_binding.node.name);
    self.walk_type(&mut arg_type_binding.node.ty);
  }

  fn walk_option_identifier(&mut self, identifier: &mut Option<Identifier>) {
    if let Some(id) = identifier {
      self.walk_identifier(id);
    }
  }

  fn walk_type_decl(&mut self, type_decl: &mut TypeDecl) {
    self.visit_location(&mut type_decl.loc);
    match &mut type_decl.node {
      TypeDeclNode::Alias(a) => self.walk_alias_type_decl(a),
      TypeDeclNode::Subtype(s) => self.walk_subtype_decl(s),
      TypeDeclNode::Relation(r) => self.walk_relation_type_decl(r),
      TypeDeclNode::Enum(e) => self.walk_enum_type_decl(e),
    }
  }

  fn walk_alias_type_decl(&mut self, alias_type_decl: &mut AliasTypeDecl) {
    self.visit_alias_type_decl(alias_type_decl);
    self.visit_location(&mut alias_type_decl.loc);
    self.walk_attributes(&mut alias_type_decl.node.attrs);
    self.walk_identifier(&mut alias_type_decl.node.name);
    self.walk_type(&mut alias_type_decl.node.alias_of);
  }

  fn walk_subtype_decl(&mut self, subtype_decl: &mut SubtypeDecl) {
    self.visit_subtype_decl(subtype_decl);
    self.visit_location(&mut subtype_decl.loc);
    self.walk_attributes(&mut subtype_decl.node.attrs);
    self.walk_identifier(&mut subtype_decl.node.name);
    self.walk_type(&mut subtype_decl.node.subtype_of);
  }

  fn walk_relation_type_decl(&mut self, relation_type_decl: &mut RelationTypeDecl) {
    self.visit_relation_type_decl(relation_type_decl);
    self.visit_location(&mut relation_type_decl.loc);
    self.walk_attributes(&mut relation_type_decl.node.attrs);
    for rel_type in relation_type_decl.relation_types_mut() {
      self.walk_relation_type(rel_type);
    }
  }

  fn walk_relation_type(&mut self, relation_type: &mut RelationType) {
    self.visit_relation_type(relation_type);
    self.visit_location(&mut relation_type.loc);
    self.walk_identifier(&mut relation_type.node.name);
    for arg_type in &mut relation_type.node.arg_types {
      self.walk_arg_type_binding(arg_type);
    }
  }

  fn walk_enum_type_decl(&mut self, enum_type_decl: &mut EnumTypeDecl) {
    self.visit_enum_type_decl(enum_type_decl);
    self.visit_location(&mut enum_type_decl.loc);
    self.walk_identifier(&mut enum_type_decl.node.name);
    for member in enum_type_decl.iter_members_mut() {
      self.walk_enum_type_member(member);
    }
  }

  fn walk_enum_type_member(&mut self, enum_type_member: &mut EnumTypeMember) {
    self.visit_enum_type_member(enum_type_member);
    self.visit_location(&mut enum_type_member.loc);
    self.walk_identifier(&mut enum_type_member.node.name);
    if let Some(assigned_num) = enum_type_member.assigned_number_mut() {
      self.walk_constant(assigned_num);
    }
  }

  fn walk_const_decl(&mut self, const_decl: &mut ConstDecl) {
    self.visit_const_decl(const_decl);
    self.visit_location(const_decl.location_mut());
    self.walk_attributes(const_decl.attributes_mut());
    for const_assign in const_decl.iter_assignments_mut() {
      self.walk_const_assignment(const_assign);
    }
  }

  fn walk_const_assignment(&mut self, const_assign: &mut ConstAssignment) {
    self.visit_const_assignment(const_assign);
    self.visit_location(const_assign.location_mut());
    self.walk_identifier(const_assign.identifier_mut());
    if let Some(ty) = const_assign.ty_mut() {
      self.walk_type(ty);
    }
    self.walk_constant(const_assign.value_mut())
  }

  fn walk_relation_decl(&mut self, relation_decl: &mut RelationDecl) {
    self.visit_relation_decl(relation_decl);
    self.visit_location(&mut relation_decl.loc);
    match &mut relation_decl.node {
      RelationDeclNode::Set(f) => self.walk_constant_set_decl(f),
      RelationDeclNode::Fact(f) => self.walk_fact_decl(f),
      RelationDeclNode::Rule(f) => self.walk_rule_decl(f),
    }
  }

  fn walk_constant_set_decl(&mut self, set_decl: &mut ConstantSetDecl) {
    self.visit_constant_set_decl(set_decl);
    self.visit_location(&mut set_decl.loc);
    self.walk_attributes(&mut set_decl.node.attrs);
    self.walk_identifier(&mut set_decl.node.name);
    self.walk_constant_set(&mut set_decl.node.set);
  }

  fn walk_constant_set(&mut self, set: &mut ConstantSet) {
    self.visit_constant_set(set);
    self.visit_location(&mut set.loc);
    for tuple in &mut set.node.tuples {
      self.walk_constant_set_tuple(tuple);
    }
  }

  fn walk_constant_set_tuple(&mut self, tuple: &mut ConstantSetTuple) {
    self.visit_constant_set_tuple(tuple);
    self.visit_location(&mut tuple.loc);
    self.walk_tag(&mut tuple.node.tag);
    self.walk_constant_tuple(&mut tuple.node.tuple);
  }

  fn walk_constant_tuple(&mut self, tuple: &mut ConstantTuple) {
    self.visit_constant_tuple(tuple);
    self.visit_location(&mut tuple.loc);
    for elem in &mut tuple.node.elems {
      self.walk_constant_or_variable(elem);
    }
  }

  fn walk_constant_or_variable(&mut self, cov: &mut ConstantOrVariable) {
    self.visit_constant_or_variable(cov);
    match cov {
      ConstantOrVariable::Constant(c) => self.walk_constant(c),
      ConstantOrVariable::Variable(v) => self.walk_variable(v),
    }
  }

  fn walk_fact_decl(&mut self, fact_decl: &mut FactDecl) {
    self.visit_fact_decl(fact_decl);
    self.visit_location(&mut fact_decl.loc);
    self.walk_tag(&mut fact_decl.node.tag);
    self.walk_atom(&mut fact_decl.node.atom);
  }

  fn walk_rule_decl(&mut self, rule_decl: &mut RuleDecl) {
    self.visit_rule_decl(rule_decl);
    self.visit_location(&mut rule_decl.loc);
    self.walk_attributes(&mut rule_decl.node.attrs);
    self.walk_tag(&mut rule_decl.node.tag);
    self.walk_rule(&mut rule_decl.node.rule);
  }

  fn walk_query_decl(&mut self, query_decl: &mut QueryDecl) {
    self.visit_query_decl(query_decl);
    self.visit_location(&mut query_decl.loc);
    self.walk_attributes(&mut query_decl.node.attrs);
    self.walk_query(&mut query_decl.node.query);
  }

  fn walk_query(&mut self, query: &mut Query) {
    self.visit_query(query);
    self.visit_location(&mut query.loc);
    match &mut query.node {
      QueryNode::Predicate(p) => self.walk_identifier(p),
      QueryNode::Atom(a) => self.walk_atom(a),
    }
  }

  fn walk_rule(&mut self, rule: &mut Rule) {
    self.visit_rule(rule);
    self.visit_location(&mut rule.loc);
    self.walk_atom(&mut rule.node.head);
    self.walk_formula(&mut rule.node.body);
  }

  fn walk_formula(&mut self, formula: &mut Formula) {
    self.visit_formula(formula);
    match formula {
      Formula::Conjunction(c) => self.walk_conjunction(c),
      Formula::Disjunction(d) => self.walk_disjunction(d),
      Formula::Implies(i) => self.walk_implies(i),
      Formula::Constraint(c) => self.walk_constraint(c),
      Formula::Atom(a) => self.walk_atom(a),
      Formula::NegAtom(n) => self.walk_neg_atom(n),
      Formula::Reduce(r) => self.walk_reduce(r),
      Formula::ForallExistsReduce(r) => self.walk_forall_exists_reduce(r),
    }
  }

  fn walk_conjunction(&mut self, conj: &mut Conjunction) {
    self.visit_conjunction(conj);
    self.visit_location(&mut conj.loc);
    for arg in &mut conj.node.args {
      self.walk_formula(arg);
    }
  }

  fn walk_disjunction(&mut self, disj: &mut Disjunction) {
    self.visit_disjunction(disj);
    self.visit_location(&mut disj.loc);
    for arg in &mut disj.node.args {
      self.walk_formula(arg);
    }
  }

  fn walk_implies(&mut self, implies: &mut Implies) {
    self.visit_implies(implies);
    self.visit_location(&mut implies.loc);
    self.walk_formula(&mut implies.node.left);
    self.walk_formula(&mut implies.node.right);
  }

  fn walk_constraint(&mut self, cons: &mut Constraint) {
    self.visit_constraint(cons);
    self.visit_location(&mut cons.loc);
    self.walk_expr(&mut cons.node.expr);
  }

  fn walk_reduce_op(&mut self, reduce_op: &mut ReduceOperator) {
    self.visit_location(&mut reduce_op.loc);
  }

  fn walk_reduce(&mut self, reduce: &mut Reduce) {
    self.visit_reduce(reduce);
    self.visit_location(&mut reduce.loc);
    for l in &mut reduce.node.left {
      match l {
        VariableOrWildcard::Variable(v) => self.walk_variable(v),
        VariableOrWildcard::Wildcard(w) => self.walk_wildcard(w),
      }
    }
    self.walk_reduce_op(&mut reduce.node.operator);
    for binding in &mut reduce.node.bindings {
      self.walk_variable_binding(binding);
    }
    self.walk_formula(&mut reduce.node.body);
    if let Some((key_vars, key_body)) = &mut reduce.node.group_by {
      for binding in key_vars {
        self.walk_variable_binding(binding);
      }
      self.walk_formula(&mut *key_body);
    }
  }

  fn walk_forall_exists_reduce(&mut self, reduce: &mut ForallExistsReduce) {
    self.visit_forall_exists_reduce(reduce);
    self.visit_location(&mut reduce.loc);
    self.walk_reduce_op(&mut reduce.node.operator);
    for binding in &mut reduce.node.bindings {
      self.walk_variable_binding(binding);
    }
    self.walk_formula(&mut reduce.node.body);
    if let Some((key_vars, key_body)) = &mut reduce.node.group_by {
      for binding in key_vars {
        self.walk_variable_binding(binding);
      }
      self.walk_formula(&mut *key_body);
    }
  }

  fn walk_atom(&mut self, atom: &mut Atom) {
    self.visit_atom(atom);
    self.visit_location(&mut atom.loc);
    self.walk_identifier(&mut atom.node.predicate);
    for arg in &mut atom.node.args {
      self.walk_expr(arg);
    }
  }

  fn walk_neg_atom(&mut self, neg_atom: &mut NegAtom) {
    self.visit_neg_atom(neg_atom);
    self.visit_location(&mut neg_atom.loc);
    self.walk_atom(&mut neg_atom.node.atom);
  }

  fn walk_type(&mut self, ty: &mut Type) {
    self.visit_type(ty);
    self.visit_location(&mut ty.loc);
  }

  fn walk_option_type(&mut self, maybe_type: &mut Option<Type>) {
    if let Some(ty) = maybe_type {
      self.walk_type(ty);
    }
  }

  fn walk_variable_binding(&mut self, binding: &mut VariableBinding) {
    self.visit_variable_binding(binding);
    self.visit_location(&mut binding.loc);
    self.walk_identifier(&mut binding.node.name);
    self.walk_option_type(&mut binding.node.ty);
  }

  fn walk_expr(&mut self, expr: &mut Expr) {
    self.visit_expr(expr);
    match expr {
      Expr::Constant(c) => self.walk_constant(c),
      Expr::Variable(v) => self.walk_variable(v),
      Expr::Wildcard(v) => self.walk_wildcard(v),
      Expr::Binary(b) => self.walk_binary_expr(b),
      Expr::Unary(u) => self.walk_unary_expr(u),
      Expr::IfThenElse(i) => self.walk_if_then_else_expr(i),
      Expr::Call(c) => self.walk_call_expr(c),
    }
  }

  fn walk_binary_op(&mut self, o: &mut BinaryOp) {
    self.visit_location(&mut o.loc);
  }

  fn walk_binary_expr(&mut self, b: &mut BinaryExpr) {
    self.visit_binary_expr(b);
    self.visit_location(&mut b.loc);
    self.walk_binary_op(&mut b.node.op);
    self.walk_expr(&mut b.node.op1);
    self.walk_expr(&mut b.node.op2);
  }

  fn walk_unary_op(&mut self, o: &mut UnaryOp) {
    self.visit_location(&mut o.loc);
  }

  fn walk_unary_expr(&mut self, u: &mut UnaryExpr) {
    self.visit_unary_expr(u);
    self.visit_location(&mut u.loc);
    self.walk_unary_op(&mut u.node.op);
    self.walk_expr(&mut u.node.op1);
  }

  fn walk_if_then_else_expr(&mut self, if_then_else: &mut IfThenElseExpr) {
    self.visit_if_then_else_expr(if_then_else);
    self.visit_location(&mut if_then_else.loc);
    self.walk_expr(&mut if_then_else.node.cond);
    self.walk_expr(&mut if_then_else.node.then_br);
    self.walk_expr(&mut if_then_else.node.else_br);
  }

  fn walk_call_expr(&mut self, call: &mut CallExpr) {
    self.visit_call_expr(call);
    self.visit_location(&mut call.loc);
    self.walk_function_identifier(call.function_identifier_mut());
    for arg in call.iter_args_mut() {
      self.walk_expr(arg);
    }
  }

  fn walk_variable(&mut self, variable: &mut Variable) {
    self.visit_variable(variable);
    self.visit_location(&mut variable.loc);
    self.walk_identifier(&mut variable.node.name);
  }

  fn walk_wildcard(&mut self, wildcard: &mut Wildcard) {
    self.visit_wildcard(wildcard);
    self.visit_location(&mut wildcard.loc);
  }

  fn walk_constant(&mut self, constant: &mut Constant) {
    self.visit_constant(constant);
    self.visit_location(&mut constant.loc);
  }

  fn walk_tag(&mut self, tag: &mut Tag) {
    self.visit_tag(tag);
    self.visit_location(&mut tag.loc);
  }

  fn walk_identifier(&mut self, identifier: &mut Identifier) {
    self.visit_identifier(identifier);
    self.visit_location(&mut identifier.loc);
  }

  fn walk_function_identifier(&mut self, function_identifier: &mut FunctionIdentifier) {
    self.visit_function_identifier(function_identifier);
    self.visit_location(&mut function_identifier.loc);
  }

  fn walk_attributes(&mut self, attributes: &mut Attributes) {
    for attr in attributes {
      self.visit_attribute(attr);
      self.visit_location(&mut attr.loc);
      self.walk_identifier(&mut attr.node.name);
      for c in &mut attr.node.pos_args {
        self.walk_constant(c);
      }
      for (n, c) in &mut attr.node.kw_args {
        self.walk_identifier(n);
        self.walk_constant(c);
      }
    }
  }
}

macro_rules! node_visitor_mut_visit_node {
  ($func:ident, $node:ident, ($($elem:ident),*)) => {
    #[allow(unused_variables)]
    fn $func(&mut self, node: &mut ast::$node) {
      paste::item! { let ($( [<$elem:lower>],)*) = self; }
      $( paste::item! { [<$elem:lower>].$func(node); } )*
    }
  };
}

macro_rules! impl_node_visitor_mut_tuple {
  ( $($id:ident,)* ) => {
    impl<'a, $($id,)*> NodeVisitorMut for ($(&'a mut $id,)*)
    where
      $($id: NodeVisitorMut,)*
    {
      node_visitor_mut_visit_node!(visit_location, AstNodeLocation, ($($id),*));

      node_visitor_mut_visit_node!(visit_item, Item, ($($id),*));
      node_visitor_mut_visit_node!(visit_arg_type_binding, ArgTypeBinding, ($($id),*));
      node_visitor_mut_visit_node!(visit_type, Type, ($($id),*));
      node_visitor_mut_visit_node!(visit_alias_type_decl, AliasTypeDecl, ($($id),*));
      node_visitor_mut_visit_node!(visit_subtype_decl, SubtypeDecl, ($($id),*));
      node_visitor_mut_visit_node!(visit_relation_type_decl, RelationTypeDecl, ($($id),*));
      node_visitor_mut_visit_node!(visit_relation_type, RelationType, ($($id),*));
      node_visitor_mut_visit_node!(visit_enum_type_decl, EnumTypeDecl, ($($id),*));
      node_visitor_mut_visit_node!(visit_enum_type_member, EnumTypeMember, ($($id),*));
      node_visitor_mut_visit_node!(visit_const_decl, ConstDecl, ($($id),*));
      node_visitor_mut_visit_node!(visit_const_assignment, ConstAssignment, ($($id),*));
      node_visitor_mut_visit_node!(visit_relation_decl, RelationDecl, ($($id),*));
      node_visitor_mut_visit_node!(visit_constant_set_decl, ConstantSetDecl, ($($id),*));
      node_visitor_mut_visit_node!(visit_constant_set, ConstantSet, ($($id),*));
      node_visitor_mut_visit_node!(visit_constant_set_tuple, ConstantSetTuple, ($($id),*));
      node_visitor_mut_visit_node!(visit_constant_tuple, ConstantTuple, ($($id),*));
      node_visitor_mut_visit_node!(visit_constant_or_variable, ConstantOrVariable, ($($id),*));
      node_visitor_mut_visit_node!(visit_fact_decl, FactDecl, ($($id),*));
      node_visitor_mut_visit_node!(visit_rule_decl, RuleDecl, ($($id),*));
      node_visitor_mut_visit_node!(visit_query_decl, QueryDecl, ($($id),*));
      node_visitor_mut_visit_node!(visit_query, Query, ($($id),*));
      node_visitor_mut_visit_node!(visit_tag, Tag, ($($id),*));
      node_visitor_mut_visit_node!(visit_rule, Rule, ($($id),*));
      node_visitor_mut_visit_node!(visit_atom, Atom, ($($id),*));
      node_visitor_mut_visit_node!(visit_neg_atom, NegAtom, ($($id),*));
      node_visitor_mut_visit_node!(visit_attribute, Attribute, ($($id),*));
      node_visitor_mut_visit_node!(visit_formula, Formula, ($($id),*));
      node_visitor_mut_visit_node!(visit_conjunction, Conjunction, ($($id),*));
      node_visitor_mut_visit_node!(visit_disjunction, Disjunction, ($($id),*));
      node_visitor_mut_visit_node!(visit_implies, Implies, ($($id),*));
      node_visitor_mut_visit_node!(visit_constraint, Constraint, ($($id),*));
      node_visitor_mut_visit_node!(visit_reduce, Reduce, ($($id),*));
      node_visitor_mut_visit_node!(visit_forall_exists_reduce, ForallExistsReduce, ($($id),*));
      node_visitor_mut_visit_node!(visit_variable_binding, VariableBinding, ($($id),*));
      node_visitor_mut_visit_node!(visit_expr, Expr, ($($id),*));
      node_visitor_mut_visit_node!(visit_binary_expr, BinaryExpr, ($($id),*));
      node_visitor_mut_visit_node!(visit_unary_expr, UnaryExpr, ($($id),*));
      node_visitor_mut_visit_node!(visit_if_then_else_expr, IfThenElseExpr, ($($id),*));
      node_visitor_mut_visit_node!(visit_constant, Constant, ($($id),*));
      node_visitor_mut_visit_node!(visit_variable, Variable, ($($id),*));
      node_visitor_mut_visit_node!(visit_wildcard, Wildcard, ($($id),*));
      node_visitor_mut_visit_node!(visit_identifier, Identifier, ($($id),*));
      node_visitor_mut_visit_node!(visit_function_identifier, FunctionIdentifier, ($($id),*));
    }
  }
}

impl_node_visitor_mut_tuple!(A,);
impl_node_visitor_mut_tuple!(A, B,);
impl_node_visitor_mut_tuple!(A, B, C,);
impl_node_visitor_mut_tuple!(A, B, C, D,);
impl_node_visitor_mut_tuple!(A, B, C, D, E,);
impl_node_visitor_mut_tuple!(A, B, C, D, E, F,);
impl_node_visitor_mut_tuple!(A, B, C, D, E, F, G,);
impl_node_visitor_mut_tuple!(A, B, C, D, E, F, G, H,);
impl_node_visitor_mut_tuple!(A, B, C, D, E, F, G, H, I,);
impl_node_visitor_mut_tuple!(A, B, C, D, E, F, G, H, I, J,);
