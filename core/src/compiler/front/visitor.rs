use super::ast;
use super::ast::*;

macro_rules! node_visitor_func_def {
  ($func:ident, $node:ident) => {
    fn $func(&mut self, _: &ast::$node) {}
  };
}

pub trait NodeVisitor {
  fn visit_location(&mut self, _: &AstNodeLocation) {}

  node_visitor_func_def!(visit_item, Item);
  node_visitor_func_def!(visit_import_decl, ImportDecl);
  node_visitor_func_def!(visit_import_file, ImportFile);
  node_visitor_func_def!(visit_arg_type_binding, ArgTypeBinding);
  node_visitor_func_def!(visit_type, Type);
  node_visitor_func_def!(visit_type_decl, TypeDecl);
  node_visitor_func_def!(visit_alias_type_decl, AliasTypeDecl);
  node_visitor_func_def!(visit_subtype_decl, SubtypeDecl);
  node_visitor_func_def!(visit_relation_type_decl, RelationTypeDecl);
  node_visitor_func_def!(visit_relation_type, RelationType);
  node_visitor_func_def!(visit_enum_type_decl, EnumTypeDecl);
  node_visitor_func_def!(visit_enum_type_member, EnumTypeMember);
  node_visitor_func_def!(visit_algebraic_data_type_decl, AlgebraicDataTypeDecl);
  node_visitor_func_def!(visit_algebraic_data_type_variant, AlgebraicDataTypeVariant);
  node_visitor_func_def!(visit_const_decl, ConstDecl);
  node_visitor_func_def!(visit_const_assignment, ConstAssignment);
  node_visitor_func_def!(visit_entity, Entity);
  node_visitor_func_def!(visit_object, Object);
  node_visitor_func_def!(visit_relation_decl, RelationDecl);
  node_visitor_func_def!(visit_constant_set_decl, ConstantSetDecl);
  node_visitor_func_def!(visit_constant_set, ConstantSet);
  node_visitor_func_def!(visit_constant_set_tuple, ConstantSetTuple);
  node_visitor_func_def!(visit_constant_tuple, ConstantTuple);
  node_visitor_func_def!(visit_constant_or_variable, ConstantOrVariable);
  node_visitor_func_def!(visit_fact_decl, FactDecl);
  node_visitor_func_def!(visit_rule_decl, RuleDecl);
  node_visitor_func_def!(visit_query_decl, QueryDecl);
  node_visitor_func_def!(visit_query, Query);
  node_visitor_func_def!(visit_tag, Tag);
  node_visitor_func_def!(visit_rule, Rule);
  node_visitor_func_def!(visit_rule_head, RuleHead);
  node_visitor_func_def!(visit_atom, Atom);
  node_visitor_func_def!(visit_neg_atom, NegAtom);
  node_visitor_func_def!(visit_formula, Formula);
  node_visitor_func_def!(visit_conjunction, Conjunction);
  node_visitor_func_def!(visit_disjunction, Disjunction);
  node_visitor_func_def!(visit_implies, Implies);
  node_visitor_func_def!(visit_constraint, Constraint);
  node_visitor_func_def!(visit_case, Case);
  node_visitor_func_def!(visit_reduce, Reduce);
  node_visitor_func_def!(visit_forall_exists_reduce, ForallExistsReduce);
  node_visitor_func_def!(visit_variable_binding, VariableBinding);
  node_visitor_func_def!(visit_expr, Expr);
  node_visitor_func_def!(visit_binary_expr, BinaryExpr);
  node_visitor_func_def!(visit_unary_expr, UnaryExpr);
  node_visitor_func_def!(visit_if_then_else_expr, IfThenElseExpr);
  node_visitor_func_def!(visit_call_expr, CallExpr);
  node_visitor_func_def!(visit_new_expr, NewExpr);
  node_visitor_func_def!(visit_constant, Constant);
  node_visitor_func_def!(visit_constant_char, ConstantChar);
  node_visitor_func_def!(visit_constant_string, ConstantString);
  node_visitor_func_def!(visit_constant_symbol, ConstantSymbol);
  node_visitor_func_def!(visit_constant_duration, ConstantDuration);
  node_visitor_func_def!(visit_constant_datetime, ConstantDateTime);
  node_visitor_func_def!(visit_variable, Variable);
  node_visitor_func_def!(visit_wildcard, Wildcard);
  node_visitor_func_def!(visit_identifier, Identifier);
  node_visitor_func_def!(visit_function_identifier, FunctionIdentifier);
  node_visitor_func_def!(visit_attribute, Attribute);
  node_visitor_func_def!(visit_attribute_value, AttributeValue);

  fn walk_items(&mut self, items: &Vec<Item>) {
    for item in items {
      self.walk_item(item);
    }
  }

  fn walk_item(&mut self, item: &Item) {
    self.visit_item(item);
    match item {
      Item::ImportDecl(id) => self.walk_import_decl(id),
      Item::TypeDecl(td) => self.walk_type_decl(td),
      Item::ConstDecl(cd) => self.walk_const_decl(cd),
      Item::RelationDecl(rd) => self.walk_relation_decl(rd),
      Item::QueryDecl(qd) => self.walk_query_decl(qd),
    }
  }

  fn walk_import_decl(&mut self, import_decl: &ImportDecl) {
    self.visit_import_decl(import_decl);
    self.visit_location(import_decl.location());
    self.walk_attributes(import_decl.attributes());
    self.walk_import_file(&import_decl.node.import_file);
  }

  fn walk_import_file(&mut self, import_file: &ImportFile) {
    self.visit_import_file(import_file);
    self.visit_location(import_file.location());
  }

  fn walk_arg_type_binding(&mut self, arg_type_binding: &ArgTypeBinding) {
    self.visit_arg_type_binding(arg_type_binding);
    self.visit_location(&arg_type_binding.loc);
    self.walk_option_identifier(&arg_type_binding.node.name);
    self.walk_type(&arg_type_binding.node.ty);
  }

  fn walk_option_identifier(&mut self, identifier: &Option<Identifier>) {
    if let Some(id) = identifier {
      self.walk_identifier(id);
    }
  }

  fn walk_type_decl(&mut self, type_decl: &TypeDecl) {
    self.visit_type_decl(type_decl);
    self.visit_location(&type_decl.loc);
    match &type_decl.node {
      TypeDeclNode::Alias(a) => self.walk_alias_type_decl(a),
      TypeDeclNode::Subtype(s) => self.walk_subtype_decl(s),
      TypeDeclNode::Relation(r) => self.walk_relation_type_decl(r),
      TypeDeclNode::Enum(e) => self.walk_enum_type_decl(e),
      TypeDeclNode::Algebraic(a) => self.walk_algebraic_data_type_decl(a),
    }
  }

  fn walk_alias_type_decl(&mut self, alias_type_decl: &AliasTypeDecl) {
    self.visit_alias_type_decl(alias_type_decl);
    self.visit_location(&alias_type_decl.loc);
    self.walk_attributes(&alias_type_decl.node.attrs);
    self.walk_identifier(&alias_type_decl.node.name);
    self.walk_type(&alias_type_decl.node.alias_of);
  }

  fn walk_subtype_decl(&mut self, subtype_decl: &SubtypeDecl) {
    self.visit_subtype_decl(subtype_decl);
    self.visit_location(&subtype_decl.loc);
    self.walk_attributes(&subtype_decl.node.attrs);
    self.walk_identifier(&subtype_decl.node.name);
    self.walk_type(&subtype_decl.node.subtype_of);
  }

  fn walk_relation_type_decl(&mut self, relation_type_decl: &RelationTypeDecl) {
    self.visit_relation_type_decl(relation_type_decl);
    self.visit_location(&relation_type_decl.loc);
    self.walk_attributes(&relation_type_decl.node.attrs);
    for rel_type in relation_type_decl.relation_types() {
      self.walk_relation_type(rel_type);
    }
  }

  fn walk_relation_type(&mut self, relation_type: &RelationType) {
    self.visit_relation_type(relation_type);
    self.visit_location(&relation_type.loc);
    self.walk_identifier(&relation_type.node.name);
    for arg_type in &relation_type.node.arg_types {
      self.walk_arg_type_binding(arg_type);
    }
  }

  fn walk_enum_type_decl(&mut self, enum_type_decl: &EnumTypeDecl) {
    self.visit_enum_type_decl(enum_type_decl);
    self.visit_location(&enum_type_decl.loc);
    self.walk_identifier(&enum_type_decl.node.name);
    for member in enum_type_decl.members() {
      self.walk_enum_type_member(member);
    }
  }

  fn walk_enum_type_member(&mut self, enum_type_member: &EnumTypeMember) {
    self.visit_enum_type_member(enum_type_member);
    self.visit_location(&enum_type_member.loc);
    self.walk_identifier(&enum_type_member.node.name);
    if let Some(assigned_num) = enum_type_member.assigned_number() {
      self.walk_constant(assigned_num);
    }
  }

  fn walk_algebraic_data_type_decl(&mut self, algebraic_data_type_decl: &AlgebraicDataTypeDecl) {
    self.visit_algebraic_data_type_decl(algebraic_data_type_decl);
    self.visit_location(&algebraic_data_type_decl.loc);
    self.walk_identifier(&algebraic_data_type_decl.node.name);
    for variant in algebraic_data_type_decl.iter_variants() {
      self.walk_algebraic_data_type_variant(variant);
    }
  }

  fn walk_algebraic_data_type_variant(&mut self, variant: &AlgebraicDataTypeVariant) {
    self.visit_algebraic_data_type_variant(variant);
    self.visit_location(&variant.loc);
    self.walk_identifier(&variant.node.constructor);
    for arg_type in variant.iter_arg_types() {
      self.walk_type(arg_type)
    }
  }

  fn walk_const_decl(&mut self, const_decl: &ConstDecl) {
    self.visit_const_decl(const_decl);
    self.visit_location(const_decl.location());
    self.walk_attributes(const_decl.attributes());
    for const_assign in const_decl.iter_assignments() {
      self.walk_const_assignment(const_assign);
    }
  }

  fn walk_const_assignment(&mut self, const_assign: &ConstAssignment) {
    self.visit_const_assignment(const_assign);
    self.visit_location(const_assign.location());
    self.walk_identifier(const_assign.identifier());
    if let Some(ty) = const_assign.ty() {
      self.walk_type(ty);
    }
    self.walk_entity(const_assign.value())
  }

  fn walk_entity(&mut self, entity: &Entity) {
    self.visit_entity(entity);
    self.visit_location(entity.location());
    match &entity.node {
      EntityNode::Expr(e) => self.walk_expr(e),
      EntityNode::Object(o) => self.walk_object(o),
    }
  }

  fn walk_object(&mut self, object: &Object) {
    self.visit_object(object);
    self.visit_location(object.location());
    self.walk_identifier(object.functor());
    for arg in object.iter_args() {
      self.walk_entity(arg);
    }
  }

  fn walk_relation_decl(&mut self, relation_decl: &RelationDecl) {
    self.visit_relation_decl(relation_decl);
    self.visit_location(&relation_decl.loc);
    match &relation_decl.node {
      RelationDeclNode::Set(f) => self.walk_constant_set_decl(f),
      RelationDeclNode::Fact(f) => self.walk_fact_decl(f),
      RelationDeclNode::Rule(f) => self.walk_rule_decl(f),
    }
  }

  fn walk_constant_set_decl(&mut self, set_decl: &ConstantSetDecl) {
    self.visit_constant_set_decl(set_decl);
    self.visit_location(&set_decl.loc);
    self.walk_attributes(&set_decl.node.attrs);
    self.walk_identifier(&set_decl.node.name);
    self.walk_constant_set(&set_decl.node.set);
  }

  fn walk_constant_set(&mut self, set: &ConstantSet) {
    self.visit_constant_set(set);
    self.visit_location(&set.loc);
    for tuple in &set.node.tuples {
      self.walk_constant_set_tuple(tuple);
    }
  }

  fn walk_constant_set_tuple(&mut self, tuple: &ConstantSetTuple) {
    self.visit_constant_set_tuple(tuple);
    self.visit_location(&tuple.loc);
    self.walk_tag(&tuple.node.tag);
    self.walk_constant_tuple(&tuple.node.tuple);
  }

  fn walk_constant_tuple(&mut self, tuple: &ConstantTuple) {
    self.visit_constant_tuple(tuple);
    self.visit_location(&tuple.loc);
    for elem in &tuple.node.elems {
      self.walk_constant_or_variable(elem);
    }
  }

  fn walk_constant_or_variable(&mut self, cov: &ConstantOrVariable) {
    self.visit_constant_or_variable(cov);
    match cov {
      ConstantOrVariable::Constant(c) => self.walk_constant(c),
      ConstantOrVariable::Variable(v) => self.walk_variable(v),
    }
  }

  fn walk_fact_decl(&mut self, fact_decl: &FactDecl) {
    self.visit_fact_decl(fact_decl);
    self.visit_location(fact_decl.location());
    self.walk_tag(&fact_decl.node.tag);
    self.walk_atom(&fact_decl.node.atom);
  }

  fn walk_rule_decl(&mut self, rule_decl: &RuleDecl) {
    self.visit_rule_decl(rule_decl);
    self.visit_location(&rule_decl.loc);
    self.walk_attributes(&rule_decl.node.attrs);
    self.walk_tag(&rule_decl.node.tag);
    self.walk_rule(&rule_decl.node.rule);
  }

  fn walk_query_decl(&mut self, query_decl: &QueryDecl) {
    self.visit_query_decl(query_decl);
    self.visit_location(&query_decl.loc);
    self.walk_attributes(&query_decl.node.attrs);
    self.walk_query(&query_decl.node.query);
  }

  fn walk_query(&mut self, query: &Query) {
    self.visit_query(query);
    self.visit_location(&query.loc);
    match &query.node {
      QueryNode::Predicate(p) => self.walk_identifier(p),
      QueryNode::Atom(a) => self.walk_atom(a),
    }
  }

  fn walk_rule(&mut self, rule: &Rule) {
    self.visit_rule(rule);
    self.visit_location(&rule.loc);
    self.walk_rule_head(&rule.node.head);
    self.walk_formula(&rule.node.body);
  }

  fn walk_rule_head(&mut self, rule_head: &RuleHead) {
    self.visit_rule_head(rule_head);
    self.visit_location(rule_head.location());
    match &rule_head.node {
      RuleHeadNode::Atom(a) => self.walk_atom(a),
      RuleHeadNode::Conjunction(c) => c.iter().for_each(|a| self.walk_atom(a)),
      RuleHeadNode::Disjunction(d) => d.iter().for_each(|a| self.walk_atom(a)),
    }
  }

  fn walk_formula(&mut self, formula: &Formula) {
    self.visit_formula(formula);
    match formula {
      Formula::Conjunction(c) => self.walk_conjunction(c),
      Formula::Disjunction(d) => self.walk_disjunction(d),
      Formula::Implies(i) => self.walk_implies(i),
      Formula::Constraint(c) => self.walk_constraint(c),
      Formula::Atom(a) => self.walk_atom(a),
      Formula::NegAtom(n) => self.walk_neg_atom(n),
      Formula::Case(c) => self.walk_case(c),
      Formula::Reduce(r) => self.walk_reduce(r),
      Formula::ForallExistsReduce(r) => self.walk_forall_exists_reduce(r),
    }
  }

  fn walk_conjunction(&mut self, conj: &Conjunction) {
    self.visit_conjunction(conj);
    self.visit_location(&conj.loc);
    for arg in &conj.node.args {
      self.walk_formula(arg);
    }
  }

  fn walk_disjunction(&mut self, disj: &Disjunction) {
    self.visit_disjunction(disj);
    self.visit_location(&disj.loc);
    for arg in &disj.node.args {
      self.walk_formula(arg);
    }
  }

  fn walk_implies(&mut self, implies: &Implies) {
    self.visit_implies(implies);
    self.visit_location(&implies.loc);
    self.walk_formula(implies.left());
    self.walk_formula(implies.right());
  }

  fn walk_constraint(&mut self, cons: &Constraint) {
    self.visit_constraint(cons);
    self.visit_location(&cons.loc);
    self.walk_expr(&cons.node.expr);
  }

  fn walk_case(&mut self, case: &Case) {
    self.visit_case(case);
    self.visit_location(&case.loc);
    self.walk_variable(&case.node.variable);
    self.walk_entity(&case.node.entity);
  }

  fn walk_reduce_op(&mut self, reduce_op: &ReduceOperator) {
    self.visit_location(&reduce_op.loc);
  }

  fn walk_reduce(&mut self, reduce: &Reduce) {
    self.visit_reduce(reduce);
    self.visit_location(&reduce.loc);
    for l in &reduce.node.left {
      match l {
        VariableOrWildcard::Variable(v) => self.walk_variable(v),
        VariableOrWildcard::Wildcard(w) => self.walk_wildcard(w),
      }
    }
    self.walk_reduce_op(&reduce.node.operator);
    for binding in &reduce.node.bindings {
      self.walk_variable_binding(binding);
    }
    self.walk_formula(&reduce.node.body);
    if let Some((key_vars, key_body)) = &reduce.node.group_by {
      for binding in key_vars {
        self.walk_variable_binding(binding);
      }
      self.walk_formula(&*key_body);
    }
  }

  fn walk_forall_exists_reduce(&mut self, reduce: &ForallExistsReduce) {
    self.visit_forall_exists_reduce(reduce);
    self.visit_location(&reduce.loc);
    self.walk_reduce_op(&reduce.node.operator);
    for binding in &reduce.node.bindings {
      self.walk_variable_binding(binding);
    }
    self.walk_formula(&reduce.node.body);
    if let Some((key_vars, key_body)) = &reduce.node.group_by {
      for binding in key_vars {
        self.walk_variable_binding(binding);
      }
      self.walk_formula(&*key_body);
    }
  }

  fn walk_atom(&mut self, atom: &Atom) {
    self.visit_atom(atom);
    self.visit_location(atom.location());
    self.walk_identifier(atom.predicate_identifier());
    for type_arg in atom.iter_type_arguments() {
      self.walk_type(type_arg)
    }
    for arg in atom.iter_arguments() {
      self.walk_expr(arg);
    }
  }

  fn walk_neg_atom(&mut self, neg_atom: &NegAtom) {
    self.visit_neg_atom(neg_atom);
    self.visit_location(&neg_atom.loc);
    self.walk_atom(&neg_atom.node.atom);
  }

  fn walk_type(&mut self, ty: &Type) {
    self.visit_type(ty);
    self.visit_location(&ty.loc);
  }

  fn walk_option_type(&mut self, maybe_type: &Option<Type>) {
    if let Some(ty) = maybe_type {
      self.walk_type(ty);
    }
  }

  fn walk_variable_binding(&mut self, binding: &VariableBinding) {
    self.visit_variable_binding(binding);
    self.visit_location(&binding.loc);
    self.walk_identifier(&binding.node.name);
    self.walk_option_type(&binding.node.ty);
  }

  fn walk_expr(&mut self, expr: &Expr) {
    self.visit_expr(expr);
    match expr {
      Expr::Constant(c) => self.walk_constant(c),
      Expr::Variable(v) => self.walk_variable(v),
      Expr::Wildcard(v) => self.walk_wildcard(v),
      Expr::Binary(b) => self.walk_binary_expr(b),
      Expr::Unary(u) => self.walk_unary_expr(u),
      Expr::IfThenElse(i) => self.walk_if_then_else_expr(i),
      Expr::Call(c) => self.walk_call_expr(c),
      Expr::New(n) => self.walk_new_expr(n),
    }
  }

  fn walk_binary_op(&mut self, o: &BinaryOp) {
    self.visit_location(&o.loc);
  }

  fn walk_binary_expr(&mut self, b: &BinaryExpr) {
    self.visit_binary_expr(b);
    self.visit_location(&b.loc);
    self.walk_binary_op(&b.node.op);
    self.walk_expr(&b.node.op1);
    self.walk_expr(&b.node.op2);
  }

  fn walk_unary_op(&mut self, o: &UnaryOp) {
    self.visit_location(&o.loc);
  }

  fn walk_unary_expr(&mut self, u: &UnaryExpr) {
    self.visit_unary_expr(u);
    self.visit_location(&u.loc);
    self.walk_unary_op(&u.node.op);
    self.walk_expr(&u.node.op1);
  }

  fn walk_if_then_else_expr(&mut self, i: &IfThenElseExpr) {
    self.visit_if_then_else_expr(i);
    self.visit_location(&i.loc);
    self.walk_expr(i.cond());
    self.walk_expr(i.then_br());
    self.walk_expr(i.else_br());
  }

  fn walk_call_expr(&mut self, c: &CallExpr) {
    self.visit_call_expr(c);
    self.visit_location(&c.loc);
    self.walk_function_identifier(c.function_identifier());
    for arg in c.iter_args() {
      self.walk_expr(arg);
    }
  }

  fn walk_new_expr(&mut self, n: &NewExpr) {
    self.visit_new_expr(n);
    self.visit_location(&n.loc);
    self.walk_identifier(n.functor_identifier());
    for arg in n.iter_args() {
      self.walk_expr(arg);
    }
  }

  fn walk_variable(&mut self, variable: &Variable) {
    self.visit_variable(variable);
    self.visit_location(&variable.loc);
    self.walk_identifier(&variable.node.name);
  }

  fn walk_wildcard(&mut self, wildcard: &Wildcard) {
    self.visit_wildcard(wildcard);
    self.visit_location(&wildcard.loc);
  }

  fn walk_constant(&mut self, constant: &Constant) {
    self.visit_constant(constant);
    self.visit_location(&constant.loc);
    match &constant.node {
      ConstantNode::Char(c) => self.walk_constant_char(c),
      ConstantNode::String(s) => self.walk_constant_string(s),
      ConstantNode::Symbol(s) => self.walk_constant_symbol(s),
      ConstantNode::DateTime(d) => self.walk_constant_datetime(d),
      ConstantNode::Duration(d) => self.walk_constant_duration(d),
      _ => {}
    }
  }

  fn walk_constant_char(&mut self, constant_char: &ConstantChar) {
    self.visit_constant_char(constant_char);
    self.visit_location(constant_char.location());
  }

  fn walk_constant_string(&mut self, constant_string: &ConstantString) {
    self.visit_constant_string(constant_string);
    self.visit_location(constant_string.location());
  }

  fn walk_constant_symbol(&mut self, constant_symbol: &ConstantSymbol) {
    self.visit_constant_symbol(constant_symbol);
    self.visit_location(constant_symbol.location());
  }

  fn walk_constant_datetime(&mut self, constant_datetime: &ConstantDateTime) {
    self.visit_constant_datetime(constant_datetime);
    self.visit_location(constant_datetime.location());
  }

  fn walk_constant_duration(&mut self, constant_duration: &ConstantDuration) {
    self.visit_constant_duration(constant_duration);
    self.visit_location(constant_duration.location());
  }

  fn walk_tag(&mut self, tag: &Tag) {
    self.visit_tag(tag);
    self.visit_location(&tag.loc);
  }

  fn walk_identifier(&mut self, identifier: &Identifier) {
    self.visit_identifier(identifier);
    self.visit_location(&identifier.loc);
  }

  fn walk_function_identifier(&mut self, function_identifier: &FunctionIdentifier) {
    self.visit_function_identifier(function_identifier);
    self.visit_location(&function_identifier.loc);
  }

  fn walk_attributes(&mut self, attributes: &Attributes) {
    for attr in attributes {
      self.visit_attribute(attr);
      self.visit_location(&attr.loc);
      self.walk_identifier(&attr.node.name);
      for v in &attr.node.pos_args {
        self.walk_attribute_value(v);
      }
      for (n, v) in &attr.node.kw_args {
        self.walk_identifier(n);
        self.walk_attribute_value(v);
      }
    }
  }

  fn walk_attribute_value(&mut self, attribute_value: &AttributeValue) {
    self.visit_attribute_value(attribute_value);
    self.visit_location(&attribute_value.loc);
    match &attribute_value.node {
      AttributeValueNode::Constant(c) => self.walk_constant(c),
      AttributeValueNode::List(l) => {
        for v in l {
          self.walk_attribute_value(v);
        }
      }
      AttributeValueNode::Tuple(t) => {
        for v in t {
          self.walk_attribute_value(v);
        }
      }
    }
  }
}

macro_rules! node_visitor_visit_node {
  ($func:ident, $node:ident, ($($elem:ident),*)) => {
    #[allow(unused_variables)]
    fn $func(&mut self, node: &ast::$node) {
      paste::item! { let ($( [<$elem:lower>],)*) = self; }
      $( paste::item! { [<$elem:lower>].$func(node); } )*
    }
  };
}

macro_rules! impl_node_visitor_tuple {
  ( $($id:ident,)* ) => {
    impl<'a, $($id,)*> NodeVisitor for ($(&'a mut $id,)*)
    where
      $($id: NodeVisitor,)*
    {
      node_visitor_visit_node!(visit_location, AstNodeLocation, ($($id),*));

      node_visitor_visit_node!(visit_item, Item, ($($id),*));
      node_visitor_visit_node!(visit_import_decl, ImportDecl, ($($id),*));
      node_visitor_visit_node!(visit_import_file, ImportFile, ($($id),*));
      node_visitor_visit_node!(visit_arg_type_binding, ArgTypeBinding, ($($id),*));
      node_visitor_visit_node!(visit_type, Type, ($($id),*));
      node_visitor_visit_node!(visit_type_decl, TypeDecl, ($($id),*));
      node_visitor_visit_node!(visit_alias_type_decl, AliasTypeDecl, ($($id),*));
      node_visitor_visit_node!(visit_subtype_decl, SubtypeDecl, ($($id),*));
      node_visitor_visit_node!(visit_relation_type_decl, RelationTypeDecl, ($($id),*));
      node_visitor_visit_node!(visit_relation_type, RelationType, ($($id),*));
      node_visitor_visit_node!(visit_enum_type_decl, EnumTypeDecl, ($($id),*));
      node_visitor_visit_node!(visit_enum_type_member, EnumTypeMember, ($($id),*));
      node_visitor_visit_node!(visit_algebraic_data_type_decl, AlgebraicDataTypeDecl, ($($id),*));
      node_visitor_visit_node!(visit_algebraic_data_type_variant, AlgebraicDataTypeVariant, ($($id),*));
      node_visitor_visit_node!(visit_const_decl, ConstDecl, ($($id),*));
      node_visitor_visit_node!(visit_const_assignment, ConstAssignment, ($($id),*));
      node_visitor_visit_node!(visit_entity, Entity, ($($id),*));
      node_visitor_visit_node!(visit_object, Object, ($($id),*));
      node_visitor_visit_node!(visit_relation_decl, RelationDecl, ($($id),*));
      node_visitor_visit_node!(visit_constant_set_decl, ConstantSetDecl, ($($id),*));
      node_visitor_visit_node!(visit_constant_set, ConstantSet, ($($id),*));
      node_visitor_visit_node!(visit_constant_set_tuple, ConstantSetTuple, ($($id),*));
      node_visitor_visit_node!(visit_constant_tuple, ConstantTuple, ($($id),*));
      node_visitor_visit_node!(visit_constant_or_variable, ConstantOrVariable, ($($id),*));
      node_visitor_visit_node!(visit_fact_decl, FactDecl, ($($id),*));
      node_visitor_visit_node!(visit_rule_decl, RuleDecl, ($($id),*));
      node_visitor_visit_node!(visit_query_decl, QueryDecl, ($($id),*));
      node_visitor_visit_node!(visit_query, Query, ($($id),*));
      node_visitor_visit_node!(visit_tag, Tag, ($($id),*));
      node_visitor_visit_node!(visit_rule, Rule, ($($id),*));
      node_visitor_visit_node!(visit_rule_head, RuleHead, ($($id),*));
      node_visitor_visit_node!(visit_atom, Atom, ($($id),*));
      node_visitor_visit_node!(visit_neg_atom, NegAtom, ($($id),*));
      node_visitor_visit_node!(visit_formula, Formula, ($($id),*));
      node_visitor_visit_node!(visit_conjunction, Conjunction, ($($id),*));
      node_visitor_visit_node!(visit_disjunction, Disjunction, ($($id),*));
      node_visitor_visit_node!(visit_implies, Implies, ($($id),*));
      node_visitor_visit_node!(visit_constraint, Constraint, ($($id),*));
      node_visitor_visit_node!(visit_case, Case, ($($id),*));
      node_visitor_visit_node!(visit_reduce, Reduce, ($($id),*));
      node_visitor_visit_node!(visit_forall_exists_reduce, ForallExistsReduce, ($($id),*));
      node_visitor_visit_node!(visit_variable_binding, VariableBinding, ($($id),*));
      node_visitor_visit_node!(visit_expr, Expr, ($($id),*));
      node_visitor_visit_node!(visit_binary_expr, BinaryExpr, ($($id),*));
      node_visitor_visit_node!(visit_unary_expr, UnaryExpr, ($($id),*));
      node_visitor_visit_node!(visit_if_then_else_expr, IfThenElseExpr, ($($id),*));
      node_visitor_visit_node!(visit_call_expr, CallExpr, ($($id),*));
      node_visitor_visit_node!(visit_new_expr, NewExpr, ($($id),*));
      node_visitor_visit_node!(visit_constant, Constant, ($($id),*));
      node_visitor_visit_node!(visit_constant_char, ConstantChar, ($($id),*));
      node_visitor_visit_node!(visit_constant_string, ConstantString, ($($id),*));
      node_visitor_visit_node!(visit_constant_symbol, ConstantSymbol, ($($id),*));
      node_visitor_visit_node!(visit_constant_datetime, ConstantDateTime, ($($id),*));
      node_visitor_visit_node!(visit_constant_duration, ConstantDuration, ($($id),*));
      node_visitor_visit_node!(visit_variable, Variable, ($($id),*));
      node_visitor_visit_node!(visit_wildcard, Wildcard, ($($id),*));
      node_visitor_visit_node!(visit_identifier, Identifier, ($($id),*));
      node_visitor_visit_node!(visit_function_identifier, FunctionIdentifier, ($($id),*));
      node_visitor_visit_node!(visit_attribute, Attribute, ($($id),*));
      node_visitor_visit_node!(visit_attribute_value, AttributeValue, ($($id),*));
    }
  }
}

impl_node_visitor_tuple!(A,);
impl_node_visitor_tuple!(A, B,);
impl_node_visitor_tuple!(A, B, C,);
impl_node_visitor_tuple!(A, B, C, D,);
impl_node_visitor_tuple!(A, B, C, D, E,);
impl_node_visitor_tuple!(A, B, C, D, E, F,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J,);
