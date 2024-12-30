use std::collections::*;

use crate::compiler::ram;
use crate::runtime::env::*;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;
use crate::runtime::database::*;

use super::dataflow::*;
use super::*;

pub struct DynamicIteration<'a, Prov: Provenance> {
  pub iter_num: usize,
  pub input_dynamic_collections: HashMap<String, DynamicCollectionRef<'a, Prov>>,
  pub dynamic_relations: HashMap<String, DynamicRelation<Prov>>,
  pub output_relations: Vec<(String, StorageMetadata)>,
  pub goal_relations: Vec<String>,
  pub relation_schedulers: HashMap<String, Scheduler>,
  pub updates: Vec<ram::Update>,
}

impl<'a, Prov: Provenance> DynamicIteration<'a, Prov> {
  pub fn new() -> Self {
    Self {
      iter_num: 0,
      input_dynamic_collections: HashMap::new(),
      dynamic_relations: HashMap::new(),
      output_relations: Vec::new(),
      goal_relations: Vec::new(),
      relation_schedulers: HashMap::new(),
      updates: Vec::new(),
    }
  }

  pub fn is_first_iteration(&self) -> bool {
    self.iter_num == 0
  }

  pub fn step(&mut self) {
    self.iter_num += 1;
  }

  pub fn add_input_dynamic_collection(&mut self, name: &str, col: DynamicCollectionRef<'a, Prov>) {
    self.input_dynamic_collections.insert(name.to_string(), col);
  }

  pub fn create_dynamic_relation(&mut self, name: &str) {
    self
      .dynamic_relations
      .insert(name.to_string(), DynamicRelation::<Prov>::new());
  }

  pub fn can_access_relation(&self, name: &str) -> bool {
    self.dynamic_relations.contains_key(name) || self.input_dynamic_collections.contains_key(name)
  }

  pub fn has_dynamic_relation(&self, name: &str) -> bool {
    self.dynamic_relations.contains_key(name)
  }

  pub fn get_dynamic_relation<'c>(&'c self, name: &str) -> Option<&'c DynamicRelation<Prov>> {
    self.dynamic_relations.get(name).map(|r| r)
  }

  pub fn get_dynamic_relation_unsafe<'c>(&'c self, name: &str) -> &'c DynamicRelation<Prov> {
    self.dynamic_relations.get(name).map(|r| r).unwrap()
  }

  pub fn add_update_dataflow(&mut self, target: &str, dataflow: ram::Dataflow) {
    self.add_update(ram::Update {
      target: target.to_string(),
      dataflow,
    });
  }

  pub fn add_update(&mut self, update: ram::Update) {
    self.updates.push(update)
  }

  pub fn add_output_relation_with_default_storage(&mut self, name: &str) {
    self.output_relations.push((name.to_string(), StorageMetadata::default()))
  }

  pub fn add_output_relation(&mut self, name: &str, metadata: &StorageMetadata) {
    self.output_relations.push((name.to_string(), metadata.clone()))
  }

  pub fn add_goal_relation(&mut self, name: &str) {
    self.goal_relations.push(name.to_string())
  }

  pub fn add_relation_scheduler(&mut self, relation_name: &str, scheduler: Scheduler) {
    self.relation_schedulers.insert(relation_name.to_string(), scheduler);
  }

  /// Run the main iteration, which iterates until a stopping criteria (provided in the runtime environment)
  /// is reached.
  pub fn run(&'a mut self, ctx: &Prov, runtime: &RuntimeEnvironment) -> HashMap<String, DynamicCollection<Prov>> {
    // Iterate until fixpoint
    while self.need_to_iterate(ctx, runtime) {
      // Perform updates
      for update in &self.updates {
        let dyn_update = self.build_dynamic_update(runtime, ctx, update);
        dyn_update
          .target
          .insert_dataflow_recent(ctx, &dyn_update.dataflow, runtime);
      }

      // Drain from dynamically generated entities in the runtime
      self.drain_dynamic_entities(ctx, runtime);

      // Update iteration number
      self.step();
    }

    // Generate result
    let mut result = HashMap::new();
    for (name, metadata) in &self.output_relations {
      let col = self.dynamic_relations.remove(name).unwrap().complete(ctx);
      result.insert(name.clone(), metadata.from_vec(col.elements, ctx));
    }
    result
  }

  /// Move the dynamic entities that are temporarily stored in the runtime into the actual relations.
  ///
  /// Note: dynamic entities are the outcome of side effects during the execution of the program.
  fn drain_dynamic_entities(&mut self, ctx: &Prov, runtime: &RuntimeEnvironment) {
    let drained_entity_facts = runtime.drain_new_entities(|r| self.has_dynamic_relation(r));
    for (relation, tuples) in drained_entity_facts {
      let update = ram::Update {
        target: relation,
        dataflow: ram::Dataflow::UntaggedVec(tuples),
      };
      let dyn_update = self.build_dynamic_update(runtime, ctx, &update);
      dyn_update
        .target
        .insert_dataflow_recent(ctx, &dyn_update.dataflow, runtime);
    }
  }

  /// Check if we need to continue the iteration
  fn need_to_iterate(&mut self, ctx: &Prov, runtime: &RuntimeEnvironment) -> bool {
    // Check if it has been changed
    if self.changed(ctx, runtime) || self.is_first_iteration() {
      // Check iter count; if reaching limit then we need to stop
      if let Some(iter_limit) = runtime.stopping_criteria.get_iter_limit() {
        if self.iter_num > iter_limit {
          self.changed(ctx, runtime);
          return false;
        }
      }

      // Check if we have reached a goal
      if runtime.stopping_criteria.stop_when_goal_relation_non_empty() {
        if self.derived_non_empty_goal_relation() {
          self.changed(ctx, runtime);
          return false;
        }
      }

      // If not reaching limit then we need to iterate
      return true;
    }

    // If it is no longer changing, but we are still less than expected iter limit, continue
    if let Some(iter_limit) = runtime.stopping_criteria.get_iter_limit() {
      if self.iter_num < iter_limit {
        return true;
      }
    }

    // Finally, stop
    return false;
  }

  pub fn run_with_monitor<M>(
    &'a mut self,
    ctx: &Prov,
    runtime: &RuntimeEnvironment,
    m: &M,
  ) -> HashMap<String, DynamicCollection<Prov>>
  where
    M: Monitor<Prov>,
  {
    // Iterate until fixpoint
    while self.need_to_iterate_with_monitor(ctx, runtime, m) {
      // !SPECIAL MONITORING!
      m.observe_stratum_iteration(self.iter_num);

      // Perform updates
      for update in &self.updates {
        println!("    [Iteration #{}: Updating {}...]", self.iter_num, update.target);
        let dyn_update = self.build_dynamic_update(runtime, ctx, update);
        dyn_update
          .target
          .insert_dataflow_recent(ctx, &dyn_update.dataflow, runtime);
      }

      // Update iteration number
      self.step();
    }

    // Generate result
    let mut result = HashMap::new();
    for (name, metadata) in &self.output_relations {
      let col = self.dynamic_relations.remove(name).unwrap().complete(ctx);
      result.insert(name.clone(), metadata.from_vec(col.elements, ctx));
    }
    result
  }

  fn need_to_iterate_with_monitor<M>(&mut self, ctx: &Prov, runtime: &RuntimeEnvironment, m: &M) -> bool
  where
    M: Monitor<Prov>,
  {
    // Check if it has been changed
    if self.changed(ctx, runtime) || self.is_first_iteration() {
      // Check iter count; if reaching limit then we need to stop
      if let Some(iter_limit) = runtime.stopping_criteria.get_iter_limit() {
        if self.iter_num > iter_limit {
          // !SPECIAL MONITORING!
          m.observe_hitting_iteration_limit();

          self.changed(ctx, runtime);
          return false;
        }
      }

      // Check if we have reached a goal
      if runtime.stopping_criteria.stop_when_goal_relation_non_empty() {
        if self.derived_non_empty_goal_relation() {
          // !SPECIAL MONITORING!
          m.observe_deriving_goal_relation();

          self.changed(ctx, runtime);
          return false;
        }
      }

      // If not reaching limit then we need to iterate
      return true;
    }

    // If it is no longer changing, but we are still less than expected iter limit, continue
    if let Some(iter_limit) = runtime.stopping_criteria.get_iter_limit() {
      if self.iter_num < iter_limit {
        return true;
      }
    }

    // !SPECIAL MONITORING!
    m.observe_converging();

    // Finally, stop
    return false;
  }

  fn derived_non_empty_goal_relation(&self) -> bool {
    if self.goal_relations.is_empty() {
      false
    } else {
      self
        .goal_relations
        .iter()
        .all(|r| self.get_dynamic_relation_unsafe(r).num_recent() > 0)
    }
  }

  fn changed(&mut self, ctx: &Prov, runtime: &RuntimeEnvironment) -> bool {
    let mut changed = false;
    for (relation_name, relation) in &mut self.dynamic_relations {
      // Get scheduler from current relation specific scheduling or from a scheduler manager
      let scheduler = if let Some(scheduler) = self.relation_schedulers.get(relation_name) {
        scheduler
      } else {
        runtime.scheduler_manager.get_default_scheduler()
      };

      // Check if the relation has changed
      if relation.changed(ctx, scheduler) {
        changed = true;
      }
    }
    changed
  }

  fn unsafe_get_dynamic_relation<'b>(&'b self, name: &str) -> &'b DynamicRelation<Prov> {
    if let Some(rel) = self.dynamic_relations.get(name) {
      rel
    } else {
      panic!("Cannot find dynamic relation `{}`", name)
    }
  }

  fn unsafe_get_input_dynamic_collection(&self, name: &str) -> DynamicCollectionRef<'a, Prov> {
    if let Some(rel) = self.input_dynamic_collections.get(name) {
      rel.clone()
    } else {
      panic!("Cannot find input dynamic collection `{}`", name)
    }
  }

  fn unsafe_get_input_dynamic_index_vec_collection(&self, name: &str) -> &'a DynamicIndexedVecCollection<Prov> {
    match self.unsafe_get_input_dynamic_collection(name) {
      DynamicCollectionRef::IndexedVec(i) => i,
      _ => panic!("Relation `{}` is not stored as indexed vector; aborting", name)
    }
  }

  fn build_dynamic_update(
    &'a self,
    env: &'a RuntimeEnvironment,
    ctx: &'a Prov,
    update: &'a ram::Update,
  ) -> DynamicUpdate<'a, Prov> {
    DynamicUpdate {
      target: self.unsafe_get_dynamic_relation(&update.target),
      dataflow: self.build_dynamic_dataflow(env, ctx, &update.dataflow),
    }
  }

  fn build_dynamic_dataflow(
    &'a self,
    env: &'a RuntimeEnvironment,
    ctx: &'a Prov,
    dataflow: &'a ram::Dataflow,
  ) -> DynamicDataflow<'a, Prov> {
    match dataflow {
      ram::Dataflow::Unit(t) => {
        if self.is_first_iteration() {
          DynamicDataflow::recent_unit(ctx, t.clone())
        } else {
          DynamicDataflow::stable_unit(ctx, t.clone())
        }
      }
      ram::Dataflow::UntaggedVec(v) => {
        let internal_tuple = v
          .iter()
          .map(|t| {
            env
              .internalize_tuple(t)
              .expect("[Internal Error] Cannot internalize tuple")
          })
          .collect();
        DynamicDataflow::untagged_vec(ctx, internal_tuple)
      }
      ram::Dataflow::Relation(c) => {
        if self.input_dynamic_collections.contains_key(c) {
          self.build_dynamic_collection(c)
        } else {
          DynamicDataflow::dynamic_relation(self.unsafe_get_dynamic_relation(c))
        }
      }
      ram::Dataflow::ForeignPredicateGround(p, a) => {
        let internal_values = a
          .iter()
          .map(|v| {
            env
              .internalize_value(v)
              .expect("[Internal Error] Cannot internalize value")
          })
          .collect();
        DynamicDataflow::foreign_predicate_ground(p.clone(), internal_values, self.is_first_iteration(), ctx, env)
      }
      ram::Dataflow::ForeignPredicateConstraint(d, p, a) => {
        // NOTE: `a` contains accessors which do not need to be internalized
        self
          .build_dynamic_dataflow(env, ctx, d)
          .foreign_predicate_constraint(p.clone(), a.clone(), ctx, env)
      }
      ram::Dataflow::ForeignPredicateJoin(d, p, a) => {
        // NOTE: `a` contains accessors which do not need to be internalized
        self
          .build_dynamic_dataflow(env, ctx, d)
          .foreign_predicate_join(p.clone(), a.clone(), ctx, env)
      }
      ram::Dataflow::OverwriteOne(d) => self.build_dynamic_dataflow(env, ctx, d).overwrite_one(ctx),
      ram::Dataflow::Exclusion(d1, d2) => {
        self
          .build_dynamic_dataflow(env, ctx, d1)
          .dynamic_exclusion(self.build_dynamic_dataflow(env, ctx, d2), ctx, env)
      }
      ram::Dataflow::Sorted(d) => {
        self.build_dynamic_dataflow(env, ctx, d).sorted()
      }
      ram::Dataflow::Filter(d, e) => {
        let internal_filter = env
          .internalize_expr(e)
          .expect("[Internal Error] Cannot internalize expression");
        self.build_dynamic_dataflow(env, ctx, d).filter(internal_filter, env)
      }
      ram::Dataflow::Find(d, k) => {
        let internal_key = env
          .internalize_tuple(k)
          .expect("[Internal Error] Cannot internalize tuple");
        self.build_dynamic_dataflow(env, ctx, d).find(internal_key)
      }
      ram::Dataflow::Project(d, e) => {
        let internal_expr = env
          .internalize_expr(e)
          .expect("[Internal Error] Cannot internalize expression");
        self.build_dynamic_dataflow(env, ctx, d).project(internal_expr, env)
      }
      ram::Dataflow::Intersect(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(env, ctx, d1);
        let r2 = self.build_dynamic_dataflow(env, ctx, d2);
        r1.intersect(r2, ctx)
      }
      ram::Dataflow::Join(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(env, ctx, d1);
        let r2 = self.build_dynamic_dataflow(env, ctx, d2);
        r1.join(r2, ctx)
      }
      ram::Dataflow::Product(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(env, ctx, d1);
        let r2 = self.build_dynamic_dataflow(env, ctx, d2);
        r1.product(r2, ctx)
      }
      ram::Dataflow::Union(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(env, ctx, d1);
        let r2 = self.build_dynamic_dataflow(env, ctx, d2);
        r1.union(r2)
      }
      ram::Dataflow::Difference(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(env, ctx, d1);
        let r2 = self.build_dynamic_dataflow(env, ctx, d2);
        r1.difference(r2, ctx)
      }
      ram::Dataflow::Antijoin(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(env, ctx, d1);
        let r2 = self.build_dynamic_dataflow(env, ctx, d2);
        r1.antijoin(r2, ctx)
      }
      ram::Dataflow::JoinIndexedVec(d1, s) => {
        let r1 = self.build_dynamic_dataflow(env, ctx, d1);
        let r2 = self.unsafe_get_input_dynamic_index_vec_collection(s);
        r1.join_indexed_vec(r2, ctx)
      }
      ram::Dataflow::Reduce(a) => {
        let op = env
          .aggregate_registry
          .instantiate_aggregator::<Prov>(&a.aggregator, a.aggregate_info.clone())
          .expect(&format!("cannot instantiate aggregator `{}`", a.aggregator));
        match &a.group_by {
          ram::ReduceGroupByType::None => {
            let c = self.build_dynamic_collection(&a.predicate);
            DynamicDataflow::new(DynamicAggregationSingleGroupDataflow::new(op, c, ctx, env))
          }
          ram::ReduceGroupByType::Implicit => {
            let c = self.build_dynamic_collection(&a.predicate);
            DynamicDataflow::new(DynamicAggregationImplicitGroupDataflow::new(op, c, ctx, env))
          }
          ram::ReduceGroupByType::Join(other) => {
            let c = self.build_dynamic_collection(&a.predicate);
            let g = self.build_dynamic_collection(&other);
            DynamicDataflow::new(DynamicAggregationJoinGroupDataflow::new(op, g, c, ctx, env))
          }
        }
      }
    }
  }

  fn build_dynamic_collection(&self, r: &str) -> DynamicDataflow<Prov> {
    let col = self.unsafe_get_input_dynamic_collection(r);
    DynamicDataflow::dynamic_collection(col, self.is_first_iteration())
  }
}

struct DynamicUpdate<'a, Prov: Provenance> {
  target: &'a DynamicRelation<Prov>,
  dataflow: DynamicDataflow<'a, Prov>,
}
