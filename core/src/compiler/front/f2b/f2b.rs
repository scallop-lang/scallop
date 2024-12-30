use std::collections::*;

use crate::common::input_tag::DynamicInputTag;
use crate::common::output_option::OutputOption;
use crate::common::value_type::ValueType;
use crate::compiler::back;
use crate::compiler::front::ast::ReduceParam;

use super::super::analyzers::boundness::{AggregationContext, ForeignPredicateBindings, RuleContext};
use super::super::ast as front;
use super::super::compile::*;
use super::FlattenExprContext;

use front::{AstNode, AstWalker, NodeLocation};

impl FrontContext {
  pub fn to_back_program(&self) -> back::Program {
    // Generate relations and facts
    let base_relations = self.to_back_relations();
    let facts = self.to_back_facts();
    let disjunctive_facts = self.to_back_disjunctive_facts();

    // Generate relations and rules
    let mut temp_relations = vec![];
    let mut output_relations = self.collect_back_outputs();
    let rules = self.to_back_rules(&mut temp_relations);

    // Output every base relation if there is no specified output relations
    if output_relations.is_empty() {
      output_relations.extend(base_relations.iter().filter_map(|r| {
        // Make sure that the predicate is not a temporary relation we've created
        if r.predicate.contains('#') {
          None
        } else if self.analysis.borrow().hidden_analysis.contains(&r.predicate) {
          Some((r.predicate.clone(), OutputOption::Hidden))
        } else {
          Some((r.predicate.clone(), OutputOption::default()))
        }
      }));
    }

    // Generate the program
    back::Program {
      relations: vec![base_relations, temp_relations].concat(),
      outputs: output_relations,
      facts,
      disjunctive_facts,
      rules,
      function_registry: self.foreign_function_registry.clone(),
      predicate_registry: self.foreign_predicate_registry.clone(),
      aggregate_registry: self.foreign_aggregate_registry.clone(),
      adt_variant_registry: self.type_inference().create_adt_variant_registry(),
    }
  }

  fn collect_back_outputs(&self) -> HashMap<String, OutputOption> {
    self
      .items
      .iter()
      .filter_map(|item| match item {
        front::Item::QueryDecl(q) => {
          let name = q.query().create_relation_name().clone();
          if let Some(file) = self.analysis.borrow().output_files_analysis.output_file(&name) {
            Some((name, OutputOption::File(file.clone())))
          } else {
            Some((name, OutputOption::default()))
          }
        }
        _ => None,
      })
      .collect()
  }

  fn to_back_relations(&self) -> Vec<back::Relation> {
    self
      .analysis
      .borrow()
      .type_inference
      .inferred_relation_types
      .iter()
      .filter_map(|(pred, (tys, _))| {
        if self.foreign_predicate_registry.contains(pred) {
          None
        } else {
          let arg_types = tys.iter().map(|type_set| type_set.to_default_value_type()).collect();
          Some(back::Relation {
            attributes: self.back_relation_attributes(pred),
            predicate: pred.clone(),
            arg_types,
          })
        }
      })
      .collect::<Vec<_>>()
  }

  fn to_back_facts(&self) -> Vec<back::Fact> {
    self
      .iter_relation_decls()
      .filter_map(|rd| match rd {
        front::RelationDecl::Set(cs) if !cs.is_disjunction() => {
          let pred = cs.name().name();

          // If there is no relation arg types being inferred, we return None
          let tys = match self.relation_arg_types(pred) {
            Some(tys) => tys,
            None => {
              return None;
            }
          };

          // Otherwise, we turn all the tuples into the corresponding types
          let fs = cs
            .set()
            .iter_tuples()
            .map(|tuple| {
              let args = tuple
                .iter_constants()
                .zip(tys.iter())
                .map(|(c, t)| {
                  // Unwrap is okay here since we have checked for constant in pre-transformation analysis
                  c.as_constant().unwrap().to_value(t)
                })
                .collect();
              let tag = to_constant_dyn_input_tag(tuple.tag());
              back::Fact {
                tag,
                predicate: pred.clone(),
                args,
              }
            })
            .collect();
          Some(fs)
        }
        front::RelationDecl::Fact(f) => {
          let pred = f.predicate_name();
          let tys = self.relation_arg_types(&pred).unwrap();
          let args = f.iter_constants().zip(tys.iter()).map(|(c, t)| c.to_value(t)).collect();
          let tag = to_constant_dyn_input_tag(&f.tag().clone().and_then(|e| e.as_constant().cloned()));
          let back_fact = back::Fact {
            tag: tag,
            predicate: pred.clone(),
            args,
          };
          Some(vec![back_fact])
        }
        _ => None,
      })
      .collect::<Vec<_>>()
      .concat()
  }

  fn to_back_disjunctive_facts(&self) -> Vec<Vec<back::Fact>> {
    self
      .iter_relation_decls()
      .filter_map(|rd| match rd {
        front::RelationDecl::Set(cs) if cs.is_disjunction() => {
          let pred = cs.name().name();
          let tys = self.relation_arg_types(pred).unwrap();
          let fs = cs
            .set()
            .iter_tuples()
            .map(|tuple| {
              let args = tuple
                .iter_constants()
                .zip(tys.iter())
                .map(|(c, t)| {
                  // Unwrap is okay here since we have checked for constant in pre-transformation analysis
                  c.as_constant().unwrap().to_value(t)
                })
                .collect();
              let tag = to_constant_dyn_input_tag(tuple.tag());
              back::Fact {
                tag,
                predicate: pred.clone(),
                args,
              }
            })
            .collect();
          Some(fs)
        }
        _ => None,
      })
      .collect::<Vec<_>>()
  }

  fn to_back_rules(&self, temp_relations: &mut Vec<back::Relation>) -> Vec<back::Rule> {
    self
      .iter_relation_decls()
      .filter_map(|rd| match rd {
        front::RelationDecl::Rule(rd) => Some(self.rule_decl_to_back_rules(rd, temp_relations)),
        _ => None,
      })
      .collect::<Vec<_>>()
      .concat()
  }

  fn rule_decl_to_back_rules(&self, rd: &front::RuleDecl, temp_relations: &mut Vec<back::Relation>) -> Vec<back::Rule> {
    let rule_loc = rd.rule().location();
    let rule_attrs = rd.attrs().iter().filter_map(|attr| self.attr_to_back_attr(attr)).collect();
    match rd.rule().head() {
      front::RuleHead::Atom(head) => {
        self.atomic_rule_decl_to_back_rules(rule_loc, head, rule_attrs, temp_relations)
      },
      front::RuleHead::Conjunction(_) => {
        panic!("[Internal Error] Conjunction should be flattened and de-sugared. This is probably a bug.")
      }
      front::RuleHead::Disjunction(head_atoms) => {
        self.disjunctive_rule_decl_to_back_rules(rule_loc, head_atoms.atoms(), rule_attrs, temp_relations)
      }
    }
  }

  fn attr_to_back_attr(&self, attr: &front::Attribute) -> Option<back::Attribute> {
    if attr.name().name() == "no_temp_relation" {
      Some(back::attributes::NoTemporaryRelationAttribute.into())
    } else {
      None
    }
  }

  fn atomic_rule_decl_to_back_rules(
    &self,
    rule_loc: &NodeLocation,
    head: &front::Atom,
    rule_attrs: Vec<back::Attribute>,
    temp_relations: &mut Vec<back::Relation>,
  ) -> Vec<back::Rule> {
    let analysis = self.analysis.borrow();

    // Basic information
    let pred = head.predicate();
    let attributes = back::Attributes::from(rule_attrs);

    // Collect information for flattening
    let mut flatten_expr = FlattenExprContext::new(&analysis.type_inference, &self.foreign_predicate_registry);
    head.walk(&mut flatten_expr);

    // Create the flattened expression that the head needs
    let head_exprs = head
      .iter_args()
      .map(|a| flatten_expr.collect_flattened_literals(a.location()))
      .flatten()
      .collect::<Vec<_>>();

    // Create the head that will be shared across all back rules
    let args = head.iter_args().map(|a| flatten_expr.get_expr_term(a)).collect();
    let head = back::Head::atom(pred.name().clone(), args);

    // Get the back rules
    let boundness_analysis = &self.analysis.borrow().boundness_analysis;
    let rule_ctx = boundness_analysis.get_rule_context(rule_loc).unwrap();
    self.formula_to_back_rules(
      &mut flatten_expr,
      rule_loc,
      attributes,
      pred.name().clone(),
      rule_ctx,
      head,
      head_exprs,
      temp_relations,
    )
  }

  fn disjunctive_rule_decl_to_back_rules(
    &self,
    rule_loc: &NodeLocation,
    head_atoms: &[front::Atom],
    rule_attrs: Vec<back::Attribute>,
    temp_relations: &mut Vec<back::Relation>,
  ) -> Vec<back::Rule> {
    let analysis = self.analysis.borrow();

    // Basic information
    let pred = head_atoms[0].predicate();
    let attributes = back::Attributes::from(rule_attrs);

    // Collect information for flattening
    let mut flatten_expr = FlattenExprContext::new(&analysis.type_inference, &self.foreign_predicate_registry);
    for head in head_atoms {
      head.walk(&mut flatten_expr)
    }

    // Create the flattened expression that the head needs
    let head_exprs = head_atoms
      .iter()
      .flat_map(|a| a.iter_args())
      .flat_map(|a| flatten_expr.collect_flattened_literals(a.location()))
      .collect::<Vec<_>>();

    // Create the head that will be shared across all back rules
    let back_head_atoms = head_atoms
      .iter()
      .map(|a| {
        let args = a.iter_args().map(|a| flatten_expr.get_expr_term(a)).collect();
        back::Atom::new(a.predicate().name().clone(), args)
      })
      .collect();
    let head = back::Head::Disjunction(back_head_atoms);

    // Get the back rules
    let boundness_analysis = &self.analysis.borrow().boundness_analysis;
    let rule_ctx = boundness_analysis.get_rule_context(rule_loc).unwrap();
    self.formula_to_back_rules(
      &mut flatten_expr,
      rule_loc,
      attributes,
      pred.name().clone(),
      rule_ctx,
      head,
      head_exprs,
      temp_relations,
    )
  }

  fn formula_to_back_rules(
    &self,
    flatten_expr: &mut FlattenExprContext,
    src_rule_loc: &NodeLocation,
    attributes: back::Attributes,
    parent_predicate: String,
    rule_ctx: &RuleContext,
    head: back::Head,
    additional: Vec<back::Literal>,
    temp_relations: &mut Vec<back::Relation>,
  ) -> Vec<back::Rule> {
    let mut rules = vec![];
    let mut temp_rules = vec![];

    // First pull out the boundness analysis
    for (conj_idx, conj_ctx) in rule_ctx.body.conjuncts.iter().enumerate() {
      // Generate aggregations
      let reduce_formulas = conj_ctx
        .agg_contexts
        .iter()
        .enumerate()
        .map(|(agg_idx, agg_ctx)| {
          let predicate = format!(
            "{}#{}#agg#{}#{}",
            parent_predicate,
            src_rule_loc.id.unwrap(),
            conj_idx,
            agg_idx
          );
          self.reduce_to_back_literal(
            flatten_expr,
            src_rule_loc,
            rule_ctx,
            agg_ctx,
            predicate,
            temp_relations,
            &mut temp_rules,
          )
        })
        .collect::<Vec<_>>();

      // Create a context and visit all atoms
      conj_ctx.pos_atoms.walk(flatten_expr);
      conj_ctx.neg_atoms.walk(flatten_expr);

      // Add positive/negative atoms
      let pos_formulas = flatten_expr.to_back_literals(&conj_ctx.pos_atoms);
      let neg_formulas = flatten_expr.to_back_literals(&conj_ctx.neg_atoms);

      // Merge all formulas
      let all_body_formulas = vec![pos_formulas, neg_formulas, reduce_formulas, additional.clone()].concat();
      let body_conj = back::Conjunction {
        args: all_body_formulas,
      };
      let rule = back::Rule {
        attributes: attributes.clone(),
        head: head.clone(),
        body: body_conj,
      };
      rules.push(rule);
    }

    vec![rules, temp_rules].concat()
  }

  fn reduce_to_back_literal(
    &self,
    flatten_expr: &mut FlattenExprContext,
    src_rule_loc: &NodeLocation,
    rule_ctx: &RuleContext,
    agg_ctx: &AggregationContext,
    predicate: String,
    temp_relations: &mut Vec<back::Relation>,
    temp_rules: &mut Vec<back::Rule>,
  ) -> back::Literal {
    // unwrap is ok because the success of compute boundness is checked already
    let pred_bindings = ForeignPredicateBindings::from(&self.foreign_predicate_registry);
    let body_bounded_vars = agg_ctx.body.compute_boundness(&pred_bindings, &vec![]).unwrap();
    let group_by_bounded_vars = agg_ctx.group_by.as_ref().map_or(BTreeSet::new(), |(ctx, _, _)| {
      ctx
        .compute_boundness(&pred_bindings, &vec![])
        .unwrap()
        .into_iter()
        .collect()
    });
    let all_bounded_vars = body_bounded_vars
      .union(&group_by_bounded_vars)
      .cloned()
      .collect::<HashSet<_>>();

    // get the core variables
    let to_agg_var_names = agg_ctx.binding_variable_names();
    let arg_var_names = agg_ctx.argument_variable_names();

    // check if there is group-by formula
    let (group_by_vars, other_group_by_vars, group_by_atom) = if let Some((group_by_ctx, _, _)) = &agg_ctx.group_by {
      // We need a set of group-by variables for the formula head
      let joined_vars = group_by_bounded_vars
        .intersection(&body_bounded_vars)
        .cloned()
        .collect::<BTreeSet<_>>();
      let other_group_by_vars = group_by_bounded_vars
        .difference(&joined_vars)
        .cloned()
        .collect::<BTreeSet<_>>();

      // Get basic information for group_by
      let group_by_predicate = format!("{}#groupby", predicate);
      let group_by_vars = joined_vars
        .iter()
        .chain(other_group_by_vars.iter())
        .cloned()
        .collect::<Vec<_>>();
      let group_by_types = self.type_inference().variable_types(src_rule_loc, group_by_vars.iter());
      let group_by_terms = self.back_terms_with_types(group_by_vars, group_by_types.clone());

      // Create a temporary relation for group_by
      let group_by_relation_attr = back::attributes::AggregateGroupByAttribute::new(joined_vars.len(), other_group_by_vars.len());
      let group_by_relation = back::Relation::new_with_attrs(
        group_by_relation_attr.into(),
        group_by_predicate.clone(),
        group_by_types.clone(),
      );
      temp_relations.push(group_by_relation);

      // Create temporary rule(s) for group_by
      let group_by_rule_head = back::Head::atom(group_by_predicate.clone(), group_by_terms.clone());
      let group_by_rules = self.formula_to_back_rules(
        flatten_expr,
        src_rule_loc,
        back::Attributes::new(),
        group_by_predicate.clone(),
        group_by_ctx,
        group_by_rule_head.clone(),
        vec![],
        temp_relations,
      );
      temp_rules.extend(group_by_rules);

      // Create group_by atom to be placed inside reduce literal
      let group_by_atom = back::Atom::new(group_by_predicate.clone(), group_by_terms.clone());

      // Return the joined vars that is going to be joining the to_aggregate part
      (
        joined_vars,
        other_group_by_vars.into_iter().collect(),
        Some(group_by_atom),
      )
    } else {
      // If there is no group-by formula, we could still have group-by variables by looking at the variables in the head
      // that are not captured by binding variables
      let group_by_vars = all_bounded_vars
        .iter()
        .filter(|v| {
          let is_head_var = rule_ctx.head_vars.iter().any(|(hv, _)| &hv == v);
          let is_to_agg_var = to_agg_var_names.contains(*v);
          let is_arg_var = arg_var_names.contains(*v);
          is_head_var && !is_to_agg_var && !is_arg_var
        })
        .cloned()
        .collect::<BTreeSet<_>>();

      // Return the group by variable
      (group_by_vars, vec![], None)
    };

    // Get the body arguments and types
    let body_predicate = format!("{}#body", predicate);
    let body_args = group_by_vars
      .iter()
      .chain(arg_var_names.iter())
      .chain(to_agg_var_names.iter())
      .cloned()
      .collect::<Vec<_>>();
    let body_arg_tys = self.type_inference().variable_types(src_rule_loc, body_args.iter());
    let body_terms = self.back_terms_with_types(body_args.clone(), body_arg_tys.clone());

    // Get the variables for reduce literal
    let group_by_vars = self.back_vars(src_rule_loc, group_by_vars.into_iter().collect());
    let other_group_by_vars = self.back_vars(src_rule_loc, other_group_by_vars);
    let to_agg_vars = self.back_vars(src_rule_loc, to_agg_var_names.clone());
    let left_vars = agg_ctx
      .result_var_or_wildcards
      .iter()
      .map(|(loc, maybe_name)| {
        if let Some(name) = maybe_name {
          back::Variable::new(name.clone(), self.type_inference().variable_type(src_rule_loc, name))
        } else {
          let name = format!("agg#wc#{}", loc.id.expect("[internal error] should have id"));
          let ty = self
            .type_inference()
            .loc_value_type(loc)
            .expect("[internal error] should have inferred type");
          back::Variable::new(name, ty)
        }
      })
      .collect();
    let arg_vars = self.back_vars(src_rule_loc, arg_var_names.iter().cloned().collect());

    // Get the types of the variables for reduce literal
    let left_var_types = self
      .type_inference()
      .loc_value_types(agg_ctx.result_var_or_wildcards.iter().map(|(l, _)| l))
      .expect("[internal error] detected result of reduce without inferred type");
    let arg_var_types = self.type_inference().variable_types(src_rule_loc, arg_var_names.iter());
    let to_agg_var_types = self
      .type_inference()
      .variable_types(src_rule_loc, to_agg_var_names.iter());

    // Generate the internal aggregate operator
    let op = agg_ctx.aggregate_op.name.name().to_string();
    let params = agg_ctx
      .aggregate_op
      .parameters
      .iter()
      .filter_map(|p| {
        if let ReduceParam::Positional(c) = p {
          let value_type = self
            .type_inference()
            .expr_value_type(p)
            .expect("[internal error] aggregate param not grounded with value type");
          Some(c.to_value(&value_type))
        } else {
          None
        }
      })
      .collect::<Vec<_>>();
    let named_params = agg_ctx
      .aggregate_op
      .parameters
      .iter()
      .filter_map(|p| {
        if let ReduceParam::Named(n) = p {
          let c = n.value();
          let value_type = self
            .type_inference()
            .expr_value_type(c)
            .expect("[internal error] aggregate param not grounded with value type");
          Some((n.name().name().clone(), c.to_value(&value_type)))
        } else {
          None
        }
      })
      .collect::<BTreeMap<_, _>>();
    let has_exclamation_mark = agg_ctx.aggregate_op.has_exclaimation_mark;

    // Get the body to-aggregate relation
    let body_attr =
      back::attributes::AggregateBodyAttribute::new(op.clone(), group_by_vars.len(), arg_vars.len(), to_agg_vars.len());
    let body_relation = back::Relation::new_with_attrs(body_attr.into(), body_predicate.clone(), body_arg_tys.clone());
    temp_relations.push(body_relation);

    // Get the rules for body
    let body_head = back::Head::atom(body_predicate.clone(), body_terms.clone());
    let body_rules = self.formula_to_back_rules(
      flatten_expr,
      src_rule_loc,
      back::Attributes::new(),
      body_predicate.clone(),
      &agg_ctx.body,
      body_head,
      vec![],
      temp_relations,
    );
    temp_rules.extend(body_rules);

    // Get the literal
    let body_atom = back::Atom::new(body_predicate.clone(), body_terms);
    let reduce_literal = back::Reduce::new(
      // Aggregator
      op,
      params,
      named_params,
      has_exclamation_mark,
      // types
      left_var_types,
      arg_var_types,
      to_agg_var_types,
      // Variables
      left_vars,
      group_by_vars,
      other_group_by_vars,
      arg_vars,
      to_agg_vars,
      // Bodies
      body_atom,
      group_by_atom,
    );

    // Return
    back::Literal::Reduce(reduce_literal)
  }

  fn back_terms_with_types(&self, var_names: Vec<String>, var_tys: Vec<ValueType>) -> Vec<back::Term> {
    var_names
      .into_iter()
      .zip(var_tys.into_iter())
      .map(|(v, t)| back::Term::variable(v, t))
      .collect()
  }

  fn back_vars_with_types(&self, var_names: Vec<String>, var_tys: Vec<ValueType>) -> Vec<back::Variable> {
    var_names
      .into_iter()
      .zip(var_tys.into_iter())
      .map(|(v, t)| back::Variable::new(v, t))
      .collect()
  }

  fn back_vars(&self, src_rule_loc: &NodeLocation, var_names: Vec<String>) -> Vec<back::Variable> {
    let var_tys = self.type_inference().variable_types(src_rule_loc, var_names.iter());
    self.back_vars_with_types(var_names, var_tys)
  }
}

fn to_constant_dyn_input_tag(constant: &Option<front::Constant>) -> DynamicInputTag {
  if let Some(constant) = constant {
    match constant {
      front::Constant::Boolean(b) => DynamicInputTag::Bool(*b.value()),
      front::Constant::Float(f) => DynamicInputTag::Float(*f.float()),
      front::Constant::Integer(i) => DynamicInputTag::Float(*i.int() as f64),
      _ => DynamicInputTag::None,
    }
  } else {
    DynamicInputTag::None
  }
}
