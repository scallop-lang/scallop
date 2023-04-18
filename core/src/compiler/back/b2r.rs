use std::collections::*;

use itertools::Itertools;

use super::*;
use crate::common::binary_op::BinaryOp;
use crate::common::expr::Expr;
use crate::common::foreign_predicate::ForeignPredicate;
use crate::common::output_option::OutputOption;
use crate::common::tuple::Tuple;
use crate::common::tuple_access::TupleAccessor;
use crate::common::tuple_type::TupleType;
use crate::common::unary_op::UnaryOp;
use crate::compiler::ram::{self, ReduceGroupByType};
use crate::utils::IdAllocator;

struct B2RContext<'a> {
  id_alloc: &'a mut IdAllocator,
  relations: &'a mut BTreeMap<String, ram::Relation>,
  temp_updates: &'a mut Vec<ram::Update>,
  pred_permutations: &'a mut HashMap<String, HashSet<Permutation>>,
  negative_dataflows: &'a mut Vec<NegativeDataflow>,
}

impl<'a> B2RContext<'a> {
  pub fn add_permutation(&mut self, pred: String, perm: Permutation) {
    self.pred_permutations.entry(pred).or_default().insert(perm);
  }
}

struct NegativeDataflow {
  sources: HashSet<String>,
  relation: ram::Relation,
  dataflow: ram::Dataflow,
}

/// Property of a dataflow
#[derive(Default, Clone)]
struct DataflowProp {
  /// Whether the dataflow needs to be sorted
  need_sorted: bool,

  /// Whether the dataflow is negative
  is_negative: bool,
}

impl From<(bool, bool)> for DataflowProp {
  fn from((need_sorted, is_negative): (bool, bool)) -> Self {
    Self {
      need_sorted,
      is_negative,
    }
  }
}

impl From<bool> for DataflowProp {
  fn from(need_sorted: bool) -> Self {
    Self {
      need_sorted,
      is_negative: false,
    }
  }
}

impl DataflowProp {
  fn with_need_sorted(&self, need_sorted: bool) -> Self {
    Self {
      need_sorted,
      is_negative: self.is_negative,
    }
  }
}

impl Program {
  pub fn strata_to_ram_strata(&self, strata: Vec<Stratum>) -> Vec<ram::Stratum> {
    let mut id_alloc = IdAllocator::default();
    let mut pred_permutations = HashMap::<String, HashSet<Permutation>>::new();
    let mut negative_dataflows = Vec::<NegativeDataflow>::new();

    // For each stratum, generate a ram stratum
    // populate the EDB permutations
    let mut ram_strata = strata
      .iter()
      .enumerate()
      .map(|(_, s)| {
        // Compute the ram stratum
        let ram_stratum =
          self.stratum_to_ram_stratum(s, &mut id_alloc, &mut pred_permutations, &mut negative_dataflows);

        // Return the stratum
        ram_stratum
      })
      .collect::<Vec<_>>();

    // Turn permutations into ram updates in their respective stratums
    ram_strata.iter_mut().for_each(|stratum| {
      let mut perm_relations = HashMap::new();

      // Go through all the predicates, add permutation updates
      for (pred, relation) in &stratum.relations {
        if let Some(permutations) = pred_permutations.get(pred) {
          for perm in permutations {
            // Populate the permutated relation
            let perm_pred = Self::permutated_predicate_name(pred, perm);
            let perm_type = perm.permute(&relation.tuple_type);
            let perm_relation = ram::Relation::hidden_relation(perm_pred.clone(), perm_type);
            perm_relations.insert(perm_pred.clone(), perm_relation);

            // Add permutation update
            stratum.updates.push(self.perm_to_ram_update(perm_pred, pred, perm));
          }
        }
      }

      // Add all permutated relations into the stratum
      stratum.relations.extend(perm_relations);
    });

    // Add negative dataflow into the earliest stratum possible
    let mut accumulated_sources = HashSet::new();
    for ram_stratum in &mut ram_strata {
      accumulated_sources.extend(ram_stratum.relations.iter().map(|(n, _)| n.clone()));

      // Get the negative dataflows that can be computed at this stratum
      let curr_neg_dfs = negative_dataflows
        .drain_filter(|ndf| ndf.sources.is_subset(&accumulated_sources))
        .collect::<Vec<_>>();

      // Add the negative dataflow into the stratum
      for neg_df in curr_neg_dfs {
        let name = neg_df.relation.predicate.clone();
        ram_stratum.relations.insert(name.clone(), neg_df.relation);
        ram_stratum.updates.push(ram::Update {
          target: name,
          dataflow: neg_df.dataflow,
        });
      }
    }

    // Return all the strata
    ram_strata
  }

  fn stratum_to_ram_stratum(
    &self,
    stratum: &Stratum,
    id_alloc: &mut IdAllocator,
    pred_permutations: &mut HashMap<String, HashSet<Permutation>>,
    negative_dataflows: &mut Vec<NegativeDataflow>,
  ) -> ram::Stratum {
    // Create the list of predicates
    let mut relations = self.stratum_relations(stratum);

    // Vectors holding updates
    let mut temp_updates = vec![];
    let mut updates = vec![];

    // Compile context
    let mut b2r_context = B2RContext {
      id_alloc,
      relations: &mut relations,
      temp_updates: &mut temp_updates,
      pred_permutations,
      negative_dataflows,
    };

    // All the updates
    for predicate in &stratum.predicates {
      for rule in self.rules_of_predicate(predicate.clone()) {
        let ctx = QueryPlanContext::from_rule(stratum, &self.predicate_registry, rule);
        let plan = ctx.query_plan();
        updates.push(self.plan_to_ram_update(&mut b2r_context, &rule.head, &plan));
      }
    }

    // Create the stratum
    let all_updates = vec![temp_updates, updates].concat();
    ram::Stratum {
      is_recursive: stratum.is_recursive,
      relations,
      updates: all_updates,
    }
  }

  fn stratum_relations(&self, stratum: &Stratum) -> BTreeMap<String, ram::Relation> {
    stratum
      .predicates
      .iter()
      .map(|pred| (pred.clone(), self.predicate_to_ram_relation(pred)))
      .collect::<BTreeMap<_, _>>()
  }

  fn predicate_to_ram_relation(&self, pred: &String) -> ram::Relation {
    let rel = self.relation_of_predicate(pred).unwrap();

    // Get tuple type
    let tuple_type = if let Some(agg_body_attr) = rel.attributes.aggregate_body_attr() {
      let num_group_by = agg_body_attr.num_group_by_vars;
      let num_args = agg_body_attr.num_arg_vars;

      // Compute the items for aggregation
      let mut elems = vec![];
      if num_group_by > 0 {
        let ty = TupleType::from_types(&rel.arg_types[..num_group_by], true);
        elems.push(ty);
      }
      let to_agg_elems = TupleType::from_types(&rel.arg_types[num_group_by + num_args..], true);
      if num_args > 0 {
        let start = num_group_by;
        let end = num_group_by + num_args;
        let ty = TupleType::from_types(&rel.arg_types[start..end], true);
        elems.push((ty, to_agg_elems).into());
      } else {
        elems.push(to_agg_elems);
      }
      if elems.len() == 1 {
        elems[0].clone()
      } else {
        TupleType::Tuple(elems.into())
      }
    } else if let Some(agg_group_by_attr) = rel.attributes.aggregate_group_by_attr() {
      let num_group_by = agg_group_by_attr.num_join_group_by_vars;

      // Compute the items for aggregation group by
      let joined = TupleType::from_types(&rel.arg_types[..num_group_by], true);
      let others = TupleType::from_types(&rel.arg_types[num_group_by..], true);
      TupleType::from((joined, others))
    } else {
      TupleType::from_types(&rel.arg_types, false)
    };

    // Get facts
    let facts = self
      .facts
      .iter()
      .filter_map(|fact| {
        if &fact.predicate == pred {
          Some(self.fact_to_ram_fact(&fact))
        } else {
          None
        }
      })
      .collect::<Vec<_>>();

    // Get disjunctive facts
    let disjunctive_facts = self
      .disjunctive_facts
      .iter()
      .enumerate()
      .filter_map(|(disjunction_id, disjunction)| {
        assert!(disjunction.len() > 1);
        if &disjunction[0].predicate == pred {
          Some(
            disjunction
              .iter()
              .map(|f| self.exclusive_fact_to_ram_fact(disjunction_id.clone(), f))
              .collect::<Vec<_>>(),
          )
        } else {
          None
        }
      })
      .flatten()
      .collect::<Vec<_>>();

    // Check input file
    let input_file = if let Some(input_file_attr) = rel.attributes.input_file_attr() {
      Some(input_file_attr.input_file.clone())
    } else {
      None
    };

    // Check output file
    let output = self.outputs.get(pred).cloned().unwrap_or(OutputOption::Hidden);

    // Check immutability, i.e., the relation is not updated by rules
    let immutable = self.rules.iter().find_position(|r| r.head.predicate() == pred).is_none();

    // The Final Relation
    let ram_relation = ram::Relation {
      predicate: pred.clone(),
      tuple_type,
      facts: vec![facts, disjunctive_facts].concat(),
      input_file,
      output,
      immutable,
    };

    ram_relation
  }

  fn fact_to_ram_fact(&self, fact: &Fact) -> ram::Fact {
    ram::Fact {
      tag: fact.tag.clone(),
      tuple: Tuple::Tuple(fact.args.iter().map(|a| Tuple::Value(a.clone())).collect()),
    }
  }

  fn exclusive_fact_to_ram_fact(&self, disj_id: usize, fact: &Fact) -> ram::Fact {
    ram::Fact {
      tag: fact.tag.with_exclusivity(disj_id),
      tuple: Tuple::Tuple(fact.args.iter().map(|a| Tuple::Value(a.clone())).collect()),
    }
  }

  fn plan_to_ram_update(&self, ctx: &mut B2RContext, head: &Head, plan: &Plan) -> ram::Update {
    // Check if the dataflow needs projection and update the dataflow
    let dataflow = match head {
      Head::Atom(head_atom) => {
        let (head_goal, need_projection) = self.head_atom_variable_tuple(head_atom);
        let subgoal = head_goal.dedup();

        // Generate the dataflow
        let dataflow = self.plan_to_ram_dataflow(ctx, &subgoal, plan, false.into());

        // Project the dataflow if needed
        let dataflow = if need_projection {
          dataflow.project(self.projection_to_atom_head(&subgoal, head_atom))
        } else if head_goal != subgoal {
          dataflow.project(subgoal.projection(&head_goal))
        } else {
          dataflow
        };

        // Check if the head predicate is a magic-set; if so we wrap an overwrite_one dataflow around
        // NOTE: only head atom predicate can be magic-set predicate
        let dataflow = if self.is_magic_set_predicate(&head.predicate()) == Some(true) {
          dataflow.overwrite_one()
        } else {
          dataflow
        };

        dataflow
      }
      Head::Disjunction(head_atoms) => {
        if !head.has_multiple_patterns() {
          let head_var_goal = VariableTuple::from_vars(head_atoms[0].variable_args().cloned(), false);

          // Generate the sub-dataflow
          let sub_dataflow = self.plan_to_ram_dataflow(ctx, &head_var_goal, plan, false.into());

          // Get all the constants in the head atoms
          let constants: Vec<_> = head_atoms
            .iter()
            .map(|head_atom| {
              use std::iter::FromIterator;
              Tuple::from_iter(head_atom.constant_args().cloned())
            })
            .collect();

          // Disjunction dataflow
          let disj_dataflow = sub_dataflow.exclusion(constants);

          // Projection
          let (mut var_counter, mut const_counter) = (0, 0);
          let projection: Expr = head_atoms[0]
            .args
            .iter()
            .map(|arg| {
              match arg {
                Term::Variable(_) => {
                  let result = Expr::access((0, var_counter));
                  var_counter += 1;
                  result
                }
                Term::Constant(_) => {
                  let result = Expr::access((1, const_counter));
                  const_counter += 1;
                  result
                }
              }
            })
            .collect();

          // Wrap the disjunction dataflow with a projection
          disj_dataflow.project(projection)
        } else {
          unimplemented!("Disjunction with more than one pattern is not supported yet.")
        }
      }
    };

    // Return the update
    ram::Update {
      target: head.predicate().clone(),
      dataflow,
    }
  }

  fn plan_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    plan: &Plan,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    match &plan.ram_node {
      HighRamNode::Unit => ram::Dataflow::unit(goal.unit_type()),
      HighRamNode::Ground(atom) => self.ground_plan_to_ram_dataflow(ctx, goal, atom, prop),
      HighRamNode::Filter(d, f) => self.filter_plan_to_ram_dataflow(ctx, goal, &*d, f, prop),
      HighRamNode::Project(d, p) => self.project_plan_to_ram_dataflow(ctx, goal, &*d, p, prop),
      HighRamNode::Join(d1, d2) => self.join_plan_to_ram_dataflow(ctx, goal, &*d1, &*d2, prop),
      HighRamNode::Antijoin(d1, d2) => self.antijoin_plan_to_ram_dataflow(ctx, goal, &*d1, &*d2, prop),
      HighRamNode::Reduce(r) => self.reduce_plan_to_ram_dataflow(ctx, goal, r, prop),
      HighRamNode::ForeignPredicateGround(a) => self.fp_ground_plan_to_ram_dataflow(ctx, goal, a, prop),
      HighRamNode::ForeignPredicateConstraint(d, a) => self.fp_constraint_plan_to_ram_dataflow(ctx, goal, &*d, a, prop),
      HighRamNode::ForeignPredicateJoin(d, a) => self.fp_join_plan_to_ram_dataflow(ctx, goal, &*d, a, prop),
    }
  }

  fn ground_plan_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    atom: &Atom,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    if atom.args.is_empty() {
      let ground = ram::Dataflow::relation(atom.predicate.clone());
      if goal.matches(atom) {
        ground
      } else {
        let atom_var_tuple = VariableTuple::empty();
        ram::Dataflow::project(ground, atom_var_tuple.projection(goal))
      }
    } else if atom.has_constant_arg() {
      // Use find
      let (constants, variables) = atom.const_var_partition();
      let const_var = |i: usize, ty: Type| -> Variable {
        Variable {
          name: format!("const#{}", i),
          ty,
        }
      };

      // Temp atom
      let sub_atom = Atom {
        predicate: atom.predicate.clone(),
        args: atom
          .args
          .iter()
          .enumerate()
          .map(|(i, t)| match t {
            Term::Constant(c) => Term::Variable(const_var(i, c.value_type())),
            Term::Variable(v) => Term::Variable(v.clone()),
          })
          .collect(),
      };

      // Subgoal
      let constants_sub_goal = if constants.len() == 1 {
        VariableTuple::Value(const_var(constants[0].0, constants[0].1.value_type()))
      } else {
        VariableTuple::Tuple(
          constants
            .iter()
            .map(|(i, c)| VariableTuple::Value(const_var(i.clone(), c.value_type())))
            .collect(),
        )
      };
      let variables_sub_goal = if variables.len() == 1 {
        VariableTuple::Value(variables[0].1.clone())
      } else {
        VariableTuple::Tuple(
          variables
            .iter()
            .map(|(_, v)| VariableTuple::Value((*v).clone()))
            .collect(),
        )
      };
      let sub_goal = VariableTuple::from((constants_sub_goal, variables_sub_goal));

      // 1. Project it into (constants, variables) tuple
      let project_1_dataflow = self.ground_plan_to_ram_dataflow(ctx, &sub_goal, &sub_atom, prop.with_need_sorted(true));

      // 2. Find using the constants
      let find_tuple = if constants.len() == 1 {
        Tuple::Value(constants[0].1.clone())
      } else {
        Tuple::Tuple(constants.into_iter().map(|(_, t)| Tuple::Value(t.clone())).collect())
      };
      let find_dataflow = ram::Dataflow::find(project_1_dataflow, find_tuple);

      // 3. Project into goal
      let dataflow = ram::Dataflow::project(find_dataflow, sub_goal.projection(goal));

      // 4. Check if we need to create temporary variable
      Self::process_dataflow(ctx, goal, dataflow, prop)
    } else {
      if goal.matches(atom) {
        ram::Dataflow::Relation(atom.predicate.clone())
      } else {
        let perm = goal.permutation(atom);
        if let Some(filter) = Self::atom_filter(atom) {
          let dataflow = ram::Dataflow::project(
            ram::Dataflow::filter(ram::Dataflow::Relation(atom.predicate.clone()), filter),
            perm.expr(),
          );
          if prop.need_sorted && !perm.order_preserving() {
            ctx.add_permutation(atom.predicate.clone(), perm);
            if prop.is_negative {
              Self::create_negative_temp_relation(ctx, goal, dataflow)
            } else {
              Self::create_temp_relation(ctx, goal, dataflow)
            }
          } else {
            dataflow
          }
        } else {
          if prop.need_sorted && !perm.order_preserving() {
            let perm_name = Self::permutated_predicate_name(&atom.predicate, &perm);
            ctx.add_permutation(atom.predicate.clone(), perm);
            ram::Dataflow::Relation(perm_name)
          } else {
            ram::Dataflow::project(ram::Dataflow::Relation(atom.predicate.clone()), perm.expr())
          }
        }
      }
    }
  }

  fn filter_plan_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    subplan: &Plan,
    filters: &Vec<Constraint>,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    let sub_goal = goal.expand(
      filters
        .iter()
        .flat_map(|c| c.unique_variable_args().into_iter().cloned()),
    );
    let sub_dataflow = self.plan_to_ram_dataflow(ctx, &sub_goal, subplan, prop.clone());
    let filter = self.constraints_to_ram_filter(&sub_goal, filters);
    let project = ram::Dataflow::project(ram::Dataflow::filter(sub_dataflow, filter), sub_goal.projection(goal));
    // TODO: Optimize the case where sub_goal to goal preserves ordering
    Self::process_dataflow(ctx, goal, project, prop)
  }

  fn project_plan_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    subplan: &Plan,
    assigns: &Vec<Assign>,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    let goal_variables = goal.variables_set();
    let bounded = goal_variables
      .intersection(&subplan.bounded_vars)
      .collect::<HashSet<_>>();
    let to_bound = goal_variables.difference(&subplan.bounded_vars).collect::<HashSet<_>>();
    if to_bound.is_empty() {
      self.plan_to_ram_dataflow(ctx, goal, subplan, prop)
    } else if !assigns.is_empty() {
      let (current_assigns, rest): (Vec<_>, Vec<_>) = assigns.iter().cloned().partition(|a| to_bound.contains(&a.left));
      let sub_goal_vars = bounded.into_iter().cloned().chain(
        current_assigns
          .iter()
          .flat_map(|a| a.variable_args().into_iter().cloned()),
      );
      let sub_goal = VariableTuple::from_vars(sub_goal_vars, false.into());
      let current_assigns = current_assigns.into_iter().map(|a| (a.left, a.right)).collect();
      let sub_dataflow =
        self.project_plan_to_ram_dataflow(ctx, &sub_goal, subplan, &rest, prop.with_need_sorted(false));
      let dataflow = ram::Dataflow::project(sub_dataflow, sub_goal.projection_assigns(goal, &current_assigns));
      Self::process_dataflow(ctx, goal, dataflow, prop)
    } else {
      panic!("[Internal Error] Non-empty to_bound but with no more assign expressions");
    }
  }

  fn join_plan_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    d1: &Plan,
    d2: &Plan,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    let join_vars = d1
      .bounded_vars
      .intersection(&d2.bounded_vars)
      .cloned()
      .collect::<HashSet<_>>();
    if join_vars.is_empty() {
      // There is no variable to join, perform product
      self.product_to_ram_dataflow(ctx, goal, d1, d2, prop)
    } else {
      // Check if both sides have the same set of variables
      if d1.bounded_vars == d2.bounded_vars {
        // d1 and d2 share the same set of variables, perform intersect
        self.intersect_to_ram_dataflow(ctx, goal, d1, d2, prop)
      } else {
        // share only a subset of variables, perform join
        self.join_to_ram_dataflow(ctx, goal, join_vars, d1, d2, prop)
      }
    }
  }

  fn product_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    d1: &Plan,
    d2: &Plan,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    // 1. Select the variables from the goal that comes from the left
    let sub_goal_1 = self.sub_tuple(goal, d1);
    let sub_dataflow_1 = self.plan_to_ram_dataflow(ctx, &sub_goal_1, d1, true.into());

    // 2. Select the variables from the goal that comes from the right
    let sub_goal_2 = self.sub_tuple(goal, d2);
    let sub_dataflow_2 = self.plan_to_ram_dataflow(ctx, &sub_goal_2, d2, true.into());

    // 3. Join the two dataflows
    let joint_sub_goal = VariableTuple::from((sub_goal_1, sub_goal_2));
    let joint_sub_dataflow = ram::Dataflow::product(sub_dataflow_1, sub_dataflow_2);

    // 4. Project it
    let dataflow = ram::Dataflow::project(joint_sub_dataflow, joint_sub_goal.projection(goal));

    // 5. Check if we need to sort it
    Self::process_dataflow(ctx, goal, dataflow, prop)
  }

  fn intersect_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    d1: &Plan,
    d2: &Plan,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    // 1. Construct the subgoal for both
    let sub_goal = match (d1.ram_node.direct_atom(), d2.ram_node.direct_atom()) {
      (Some(a1), _) => VariableTuple::from_vars(a1.unique_variable_args(), false),
      (None, Some(a2)) => VariableTuple::from_vars(a2.unique_variable_args(), false),
      (None, None) => goal.clone(),
    };

    // 2. Create two dataflows
    let dataflow_1 = self.plan_to_ram_dataflow(ctx, &sub_goal, d1, true.into());
    let dataflow_2 = self.plan_to_ram_dataflow(ctx, &sub_goal, d2, true.into());

    // 3. Intersect them together
    let itsct_dataflow = ram::Dataflow::intersect(dataflow_1, dataflow_2);

    // 4. See if we need to project or even create a temporary relation
    if &sub_goal == goal {
      itsct_dataflow
    } else {
      let project_dataflow = ram::Dataflow::project(itsct_dataflow, sub_goal.projection(goal));
      Self::process_dataflow(ctx, goal, project_dataflow, prop)
    }
  }

  fn join_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    join_vars: HashSet<Variable>,
    d1: &Plan,
    d2: &Plan,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    // 1. Construct the joint subgoal
    let sorted_joint_vars = join_vars.iter().sorted().collect::<Vec<_>>();
    let joint_var_tuple = VariableTuple::from_vars(sorted_joint_vars.into_iter().cloned(), true);

    // 2. Construct the d1 subgoal
    let d1_vars_set = goal
      .variables_set()
      .intersection(&d1.bounded_vars)
      .cloned()
      .collect::<HashSet<_>>();
    let d1_vars = d1_vars_set.difference(&join_vars).sorted().collect::<Vec<_>>();
    let d1_var_tuple = VariableTuple::from_vars(d1_vars.into_iter().cloned(), true);
    let d1_sub_goal = VariableTuple::from((joint_var_tuple.clone(), d1_var_tuple.clone()));
    let d1_dataflow = self.plan_to_ram_dataflow(ctx, &d1_sub_goal, d1, true.into());

    // 3. Construct the d2 subgoal
    let d2_vars_set = goal
      .variables_set()
      .intersection(&d2.bounded_vars)
      .cloned()
      .collect::<HashSet<_>>();
    let d2_vars = d2_vars_set.difference(&join_vars).sorted().collect::<Vec<_>>();
    let d2_var_tuple = VariableTuple::from_vars(d2_vars.into_iter().cloned(), true);
    let d2_sub_goal = VariableTuple::from((joint_var_tuple.clone(), d2_var_tuple.clone()));
    let d2_dataflow = self.plan_to_ram_dataflow(ctx, &d2_sub_goal, d2, true.into());

    // 4. Join them
    let joint_sub_goal = VariableTuple::from((joint_var_tuple.clone(), d1_var_tuple.clone(), d2_var_tuple.clone()));
    let joint_dataflow = ram::Dataflow::join(d1_dataflow, d2_dataflow);
    let projected_joint_dataflow = ram::Dataflow::project(joint_dataflow, joint_sub_goal.projection(goal));

    // 5. Create temporary relation if needed to be sorted
    Self::process_dataflow(ctx, goal, projected_joint_dataflow, prop)
  }

  fn antijoin_plan_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    d1: &Plan,
    d2: &Plan,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    // Check if both sides have the same set of variables
    if d1.bounded_vars == d2.bounded_vars {
      // d1 and d2 share the same set of variables, perform difference
      self.difference_to_ram_dataflow(ctx, goal, d1, d2, prop)
    } else {
      // share only a subset of variables, perform antijoin
      self.antijoin_to_ram_dataflow(ctx, goal, d1, d2, prop)
    }
  }

  fn antijoin_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    d1: &Plan,
    d2: &Plan,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    let join_vars = d1
      .bounded_vars
      .intersection(&d2.bounded_vars)
      .cloned()
      .collect::<HashSet<_>>();

    // 1. Construct the joint subgoal
    let sorted_joint_vars = join_vars.iter().sorted().collect::<Vec<_>>();
    let joint_var_tuple = VariableTuple::from_vars(sorted_joint_vars.into_iter().cloned(), true);

    // 2. Positive part contains two parts, joint and others
    let d1_vars_set = goal
      .variables_set()
      .intersection(&d1.bounded_vars)
      .cloned()
      .collect::<HashSet<_>>();
    let d1_vars = d1_vars_set.difference(&join_vars).sorted().collect::<Vec<_>>();
    let d1_var_tuple = VariableTuple::from_vars(d1_vars.into_iter().cloned(), true);
    let d1_sub_goal = VariableTuple::from((joint_var_tuple.clone(), d1_var_tuple.clone()));
    let d1_dataflow = self.plan_to_ram_dataflow(ctx, &d1_sub_goal, d1, true.into());

    // 3. Negative part is just the joint part
    let d2_dataflow = self.plan_to_ram_dataflow(ctx, &joint_var_tuple, d2, (true, true).into());

    // 4. Construct the dataflow, which will be of form d1_sub_goal
    let dataflow = ram::Dataflow::project(
      ram::Dataflow::antijoin(d1_dataflow, d2_dataflow),
      d1_sub_goal.projection(goal),
    );

    // 5. Check if we need to sort
    Self::process_dataflow(ctx, goal, dataflow, prop)
  }

  fn difference_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    d1: &Plan,
    d2: &Plan,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    // 1. Construct the subgoal for both
    let sub_goal = match (d1.ram_node.direct_atom(), d2.ram_node.direct_atom()) {
      (Some(a1), _) => VariableTuple::from_vars(a1.unique_variable_args(), false),
      (None, Some(a2)) => VariableTuple::from_vars(a2.unique_variable_args(), false),
      (None, None) => goal.clone(),
    };

    // 2. Create two dataflows
    let dataflow_1 = self.plan_to_ram_dataflow(ctx, &sub_goal, d1, true.into());
    let dataflow_2 = self.plan_to_ram_dataflow(ctx, &sub_goal, d2, (true, true).into());

    // 3. Construct Difference
    let diff_dataflow = ram::Dataflow::difference(dataflow_1, dataflow_2);

    // 4. See if we need to project or even create a temporary relation
    if &sub_goal == goal {
      diff_dataflow
    } else {
      let project = sub_goal.projection(goal);
      let project_dataflow = ram::Dataflow::project(diff_dataflow, project);
      Self::process_dataflow(ctx, goal, project_dataflow, prop)
    }
  }

  fn reduce_plan_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    r: &Reduce,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    // Handle different versions of reduce...
    let lt = VariableTuple::from_vars(r.left_vars.iter().cloned(), true);
    let (var_tuple, has_group_by) = if !r.group_by_vars.is_empty() {
      let gbt = VariableTuple::from_vars(r.group_by_vars.iter().cloned(), true);
      let vt = if r.group_by_formula.is_some() {
        let ogbt = VariableTuple::from_vars(r.other_group_by_vars.iter().cloned(), true);
        if !r.arg_vars.is_empty() {
          let avt = VariableTuple::from_vars(r.arg_vars.iter().cloned(), true);
          VariableTuple::from((gbt, ogbt, (avt, lt)))
        } else {
          // With group by, no arg
          VariableTuple::from((gbt, ogbt, lt))
        }
      } else {
        if !r.arg_vars.is_empty() {
          let avt = VariableTuple::from_vars(r.arg_vars.iter().cloned(), true);
          VariableTuple::from((gbt, (avt, lt)))
        } else {
          // With group by, no arg
          VariableTuple::from((gbt, lt))
        }
      };
      (vt, true)
    } else {
      if !r.arg_vars.is_empty() {
        // No group by, with arg
        let avt = VariableTuple::from_vars(r.arg_vars.iter().cloned(), true);
        let var_tuple = VariableTuple::from((avt, lt));
        (var_tuple, false)
      } else {
        // No group_by, no arg
        (lt, false)
      }
    };

    // Get the type of group by...
    let group_by = if has_group_by {
      if let Some(group_by_pred) = &r.group_by_formula {
        ReduceGroupByType::Join(group_by_pred.predicate.clone())
      } else {
        ReduceGroupByType::Implicit
      }
    } else {
      ReduceGroupByType::None
    };

    // Construct the reduce and the dataflow
    let agg = ram::Dataflow::reduce(r.op.clone(), r.body_formula.predicate.clone(), group_by);
    let dataflow = ram::Dataflow::project(agg, var_tuple.projection(goal));

    // Check if we need to store into temporary variable
    Self::process_dataflow(ctx, goal, dataflow, prop)
  }

  fn fp_ground_plan_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    atom: &Atom,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    // Find the foreign predicate in the registry
    let fp = self.predicate_registry.get(&atom.predicate).unwrap();

    // Get information from the atom
    let pred: String = atom.predicate.clone();
    let inputs: Vec<Constant> = atom.args.iter().take(fp.num_bounded()).map(|arg| arg.as_constant().unwrap()).cloned().collect();
    let ground_dataflow = ram::Dataflow::ForeignPredicateGround(pred, inputs);

    // Get the projection onto the variable tuple
    let var_tuple = atom.args.iter().skip(fp.num_bounded()).map(|arg| arg.as_variable().unwrap()).cloned();
    let project = VariableTuple::from_vars(var_tuple, false).projection(goal);

    // Project the dataflow
    let dataflow = ram::Dataflow::project(ground_dataflow, project);
    Self::process_dataflow(ctx, goal, dataflow, prop)
  }

  fn fp_constraint_plan_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    d: &Plan,
    atom: &Atom,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    // Generate a sub-dataflow
    let sub_goal = VariableTuple::from_vars(d.bounded_vars.iter().cloned(), false);
    let sub_dataflow: ram::Dataflow = self.plan_to_ram_dataflow(ctx, &sub_goal, d, prop.with_need_sorted(false));

    // Generate information for foreign predicate constraint
    let pred: String = atom.predicate.clone();
    let exprs: Vec<Expr> = atom.args.iter().map(|arg| {
      match arg {
        Term::Constant(c) => Expr::Constant(c.clone()),
        Term::Variable(v) => Expr::Access(sub_goal.accessor_of(v).unwrap()),
      }
    }).collect();

    // Return a foreign predicate constraint dataflow
    let dataflow = sub_dataflow.foreign_predicate_constraint(pred, exprs);

    // Get the projection onto the variable tuple
    let projection = sub_goal.projection(goal);
    let dataflow = ram::Dataflow::project(dataflow, projection);
    Self::process_dataflow(ctx, goal, dataflow, prop)
  }

  fn fp_join_plan_to_ram_dataflow(
    &self,
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    d: &Plan,
    atom: &Atom,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    // Find the foreign predicate in the registry
    let fp = self.predicate_registry.get(&atom.predicate).unwrap();

    // Generate a sub-dataflow
    let left_goal = VariableTuple::from_vars(d.bounded_vars.iter().cloned(), false);
    let left_dataflow: ram::Dataflow = self.plan_to_ram_dataflow(ctx, &left_goal, d, prop.with_need_sorted(false));

    // Generate information for foreign predicate constraint
    let pred: String = atom.predicate.clone();
    let exprs: Vec<Expr> = atom.args.iter().take(fp.num_bounded()).map(|arg| {
      match arg {
        Term::Constant(c) => Expr::Constant(c.clone()),
        Term::Variable(v) => {
          Expr::Access(left_goal.accessor_of(v).unwrap())
        },
      }
    }).collect();

    // Generate the joint dataflow
    let join_dataflow = left_dataflow.foreign_predicate_join(pred, exprs);

    // Get the variable tuple of the joined output
    let free_vars: Vec<_> = atom.args.iter().skip(fp.num_bounded()).map(|arg| arg.as_variable().unwrap()).cloned().collect();
    let right_tuple = VariableTuple::from_vars(free_vars.iter().cloned(), false);
    let var_tuple = VariableTuple::from((left_goal, right_tuple));

    // Project the dataflow
    let projection = var_tuple.projection(goal);
    let project_dataflow = ram::Dataflow::project(join_dataflow, projection);
    Self::process_dataflow(ctx, goal, project_dataflow, prop)
  }

  fn process_dataflow(
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    dataflow: ram::Dataflow,
    prop: DataflowProp,
  ) -> ram::Dataflow {
    if prop.need_sorted {
      if prop.is_negative {
        Self::create_negative_temp_relation(ctx, goal, dataflow)
      } else {
        Self::create_temp_relation(ctx, goal, dataflow)
      }
    } else {
      dataflow
    }
  }

  fn head_atom_variable_tuple(&self, head: &Atom) -> (VariableTuple, bool) {
    let rel = self.relation_of_predicate(&head.predicate).unwrap();
    if let Some(agg_attr) = rel.attributes.aggregate_body_attr() {
      // For an aggregate sub-relation
      let head_args = head.variable_args().into_iter().cloned().collect::<Vec<_>>();
      let num_group_by = agg_attr.num_group_by_vars;
      let num_args = agg_attr.num_arg_vars;

      // Compute the items for aggregation
      let mut elems = vec![];
      if num_group_by > 0 {
        let tuple_vars = VariableTuple::from_vars(head_args[..num_group_by].iter().cloned(), true);
        elems.push(tuple_vars);
      }
      let start = num_group_by + num_args;
      let to_agg_tuple = VariableTuple::from_vars(head_args[start..].iter().cloned(), true);
      if num_args > 0 {
        let start = num_group_by;
        let end = num_group_by + num_args;
        let tuple_args = VariableTuple::from_vars(head_args[start..end].iter().cloned(), true);
        elems.push((tuple_args, to_agg_tuple).into());
      } else {
        elems.push(to_agg_tuple);
      }

      // Combine them into a var tuple
      let var_tuple = if elems.len() == 1 {
        elems[0].clone()
      } else {
        VariableTuple::Tuple(elems.into())
      };

      // Aggregate relation does not need projection, as there is no constant in the head
      (var_tuple, false)
    } else if let Some(agg_group_by_attr) = rel.attributes.aggregate_group_by_attr() {
      let num_group_by = agg_group_by_attr.num_join_group_by_vars;
      let var_args = head.variable_args().into_iter().cloned().collect::<Vec<_>>();
      let joined = VariableTuple::from_vars((&var_args[..num_group_by]).into_iter().cloned(), true);
      let others = VariableTuple::from_vars((&var_args[num_group_by..]).into_iter().cloned(), true);

      // Aggregate group by does not need projection too as there is no constant in head
      (VariableTuple::from((joined, others)), false)
    } else {
      // Normal relation may need additional projection if there is constant in the head
      let mut need_projection = false;
      let top = head
        .args
        .iter()
        .filter_map(|arg| match arg {
          Term::Variable(v) => {
            Some(VariableTuple::Value(v.clone()))
          },
          _ => {
            need_projection = true;
            None
          }
        })
        .collect();
      (VariableTuple::Tuple(top), need_projection)
    }
  }

  pub fn projection_to_atom_head(&self, var_tuple: &VariableTuple, head_atom: &Atom) -> Expr {
    Expr::Tuple(
      head_atom
        .args
        .iter()
        .map(|a| match a {
          Term::Variable(v) => {
            let acc = var_tuple
              .accessor_of(v)
              .expect("[Internal Error] Cannot find accessor of variable");
            Expr::Access(acc)
          }
          Term::Constant(c) => Expr::Constant(c.clone()),
        })
        .collect(),
    )
  }

  fn sub_tuple(&self, goal: &VariableTuple, plan: &Plan) -> VariableTuple {
    // Shortcut for pure atom where all args are distinct variables
    if let Some(a) = plan.ram_node.direct_atom() {
      if a.is_pure() {
        return goal.subtuple(a.variable_args().cloned().into_iter());
      }
    }

    // Otherwise, use arbitrary order of bounded args to create subtuple
    let mut sorted_vars = plan.bounded_vars.iter().cloned().collect::<Vec<_>>();
    sorted_vars.sort();
    goal.subtuple(sorted_vars.into_iter())
  }

  fn constraints_to_ram_filter(&self, var_tuple: &VariableTuple, constraints: &Vec<Constraint>) -> Expr {
    let term_to_expr = |t: &Term| match t {
      Term::Constant(c) => Expr::Constant(c.clone()),
      Term::Variable(v) => Expr::Access(var_tuple.accessor_of(v).unwrap()),
    };
    let constraint_to_expr = |c: &Constraint| match c {
      Constraint::Binary(b) => Expr::binary(BinaryOp::from(&b.op), term_to_expr(&b.op1), term_to_expr(&b.op2)),
      Constraint::Unary(u) => Expr::unary(UnaryOp::from(&u.op), term_to_expr(&u.op1)),
    };
    let mut expr = constraint_to_expr(&constraints[0]);
    for i in 1..constraints.len() {
      expr = Expr::binary(BinaryOp::And, expr, constraint_to_expr(&constraints[i]));
    }
    expr
  }

  fn create_negative_temp_relation(
    ctx: &mut B2RContext,
    goal: &VariableTuple,
    dataflow: ram::Dataflow,
  ) -> ram::Dataflow {
    // Create relation
    let relation_name = format!("#ntemp#{}", ctx.id_alloc.alloc());
    let relation_type = goal.tuple_type();
    let relation = ram::Relation::hidden_relation(relation_name.clone(), relation_type);

    // Get the sources
    let sources = dataflow.source_relations().into_iter().cloned().collect::<HashSet<_>>();

    // Insert negative dataflow
    ctx.negative_dataflows.push(NegativeDataflow {
      sources,
      relation,
      dataflow,
    });

    // Create outgoing dataflow from the temporary relation
    ram::Dataflow::Relation(relation_name)
  }

  fn create_temp_relation(ctx: &mut B2RContext, goal: &VariableTuple, dataflow: ram::Dataflow) -> ram::Dataflow {
    // Create relation
    let relation_name = format!("#temp#{}", ctx.id_alloc.alloc());
    let relation_type = goal.tuple_type();
    let relation = ram::Relation::hidden_relation(relation_name.clone(), relation_type);
    ctx.relations.insert(relation_name.clone(), relation);

    // Insert temporary update
    ctx.temp_updates.push(ram::Update {
      target: relation_name.clone(),
      dataflow,
    });

    // Create outgoing dataflow from the temporary relation
    ram::Dataflow::Relation(relation_name)
  }

  fn atom_eq_vars(atom: &Atom) -> Vec<(usize, usize)> {
    let mut var_id_map = HashMap::<Variable, usize>::new();
    let mut eq_vars = vec![];
    for (i, arg) in atom.args.iter().enumerate() {
      match arg {
        Term::Variable(v) => {
          if let Some(var_id) = var_id_map.get(v) {
            eq_vars.push((var_id.clone(), i));
          } else {
            var_id_map.insert(v.clone(), i);
          }
        }
        _ => {}
      }
    }
    eq_vars
  }

  fn atom_filter(atom: &Atom) -> Option<Expr> {
    let eq_vars = Self::atom_eq_vars(atom);
    if eq_vars.is_empty() {
      None
    } else {
      let eq_expr = |(a, b)| {
        Expr::binary(
          BinaryOp::Eq,
          Expr::Access(TupleAccessor::from(a)),
          Expr::Access(TupleAccessor::from(b)),
        )
      };
      let mut expr = eq_expr(eq_vars[0]);
      for i in 1..eq_vars.len() {
        expr = Expr::binary(BinaryOp::And, expr, eq_expr(eq_vars[i]));
      }
      Some(expr)
    }
  }

  fn perm_to_ram_update(&self, perm_pred_name: String, pred_name: &String, perm: &Permutation) -> ram::Update {
    ram::Update {
      target: perm_pred_name,
      dataflow: ram::Dataflow::project(ram::Dataflow::relation(pred_name.clone()), perm.expr()),
    }
  }

  fn permutated_predicate_name(pred: &String, perm: &Permutation) -> String {
    format!("{}#perm#{}", pred, perm)
  }
}
