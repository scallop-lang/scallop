use std::collections::*;

use crate::compiler::ram;
use crate::runtime::env::*;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;

use super::dataflow::*;
use super::*;

pub struct DynamicIteration<'a, Prov: Provenance> {
  pub iter_num: usize,
  pub input_dynamic_collections: HashMap<String, &'a DynamicCollection<Prov>>,
  pub dynamic_relations: HashMap<String, DynamicRelation<Prov>>,
  pub output_relations: Vec<String>,
  pub updates: Vec<ram::Update>,
}

impl<'a, Prov: Provenance> DynamicIteration<'a, Prov> {
  pub fn new() -> Self {
    Self {
      iter_num: 0,
      input_dynamic_collections: HashMap::new(),
      dynamic_relations: HashMap::new(),
      output_relations: Vec::new(),
      updates: Vec::new(),
    }
  }

  pub fn is_first_iteration(&self) -> bool {
    self.iter_num == 0
  }

  pub fn step(&mut self) {
    self.iter_num += 1;
  }

  pub fn add_input_dynamic_collection(&mut self, name: &str, col: &'a DynamicCollection<Prov>) {
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

  pub fn get_dynamic_relation<'c>(&'c mut self, name: &str) -> Option<&'c DynamicRelation<Prov>> {
    self.dynamic_relations.get(name).map(|r| r)
  }

  pub fn get_dynamic_relation_unsafe<'c>(&'c mut self, name: &str) -> &'c DynamicRelation<Prov> {
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

  pub fn add_output_relation(&mut self, name: &str) {
    self.output_relations.push(name.to_string())
  }

  pub fn run(&'a mut self, ctx: &Prov, runtime: &RuntimeEnvironment) -> HashMap<String, DynamicCollection<Prov>> {
    // Iterate until fixpoint
    while self.need_to_iterate(ctx, &runtime.iter_limit) {
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
    for name in &self.output_relations {
      let col = self.dynamic_relations.remove(name).unwrap().complete(ctx);
      result.insert(name.clone(), col);
    }
    result
  }

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

  fn need_to_iterate(&mut self, ctx: &Prov, iter_limit: &Option<usize>) -> bool {
    // Check if it has been changed
    if self.changed(ctx) || self.is_first_iteration() {
      // Check iter count; if reaching limit then we need to stop
      if let Some(iter_limit) = iter_limit {
        if self.iter_num > *iter_limit {
          self.changed(ctx);
          return false;
        }
      }

      // If not reaching limit then we need to iterate
      return true;
    }

    // If it is no longer changing, but we are still less than expected iter limit, continue
    if let Some(iter_limit) = iter_limit {
      if self.iter_num < *iter_limit {
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
    while self.need_to_iterate_with_monitor(ctx, &runtime.iter_limit, m) {
      // !SPECIAL MONITORING!
      m.observe_stratum_iteration(self.iter_num);

      // Perform updates
      for update in &self.updates {
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
    for name in &self.output_relations {
      let col = self.dynamic_relations.remove(name).unwrap().complete(ctx);
      result.insert(name.clone(), col);
    }
    result
  }

  fn need_to_iterate_with_monitor<M>(&mut self, ctx: &Prov, iter_limit: &Option<usize>, m: &M) -> bool
  where
    M: Monitor<Prov>,
  {
    // Check if it has been changed
    if self.changed(ctx) || self.is_first_iteration() {
      // Check iter count; if reaching limit then we need to stop
      if let Some(iter_limit) = iter_limit {
        if self.iter_num > *iter_limit {
          // !SPECIAL MONITORING!
          m.observe_hitting_iteration_limit();

          self.changed(ctx);
          return false;
        }
      }

      // If not reaching limit then we need to iterate
      return true;
    }

    // If it is no longer changing, but we are still less than expected iter limit, continue
    if let Some(iter_limit) = iter_limit {
      if self.iter_num < *iter_limit {
        return true;
      }
    }

    // !SPECIAL MONITORING!
    m.observe_converging();

    // Finally, stop
    return false;
  }

  fn changed(&mut self, ctx: &Prov) -> bool {
    let mut changed = false;
    for (_, relation) in &mut self.dynamic_relations {
      if relation.changed(ctx) {
        changed = true;
      }
    }
    changed
  }

  fn unsafe_get_dynamic_relation(&'a self, name: &str) -> &'a DynamicRelation<Prov> {
    if self.dynamic_relations.contains_key(name) {
      &self.dynamic_relations[name]
    } else {
      panic!("Cannot find dynamic relation `{}`", name)
    }
  }

  fn unsafe_get_input_dynamic_collection(&'a self, name: &str) -> &'a DynamicCollection<Prov> {
    if self.input_dynamic_collections.contains_key(name) {
      self.input_dynamic_collections[name]
    } else {
      panic!("Cannot find input dynamic collection `{}`", name)
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
      ram::Dataflow::Exclusion(d1, d2) => self
        .build_dynamic_dataflow(env, ctx, d1)
        .dynamic_exclusion(self.build_dynamic_dataflow(env, ctx, d2), ctx, env),
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
      ram::Dataflow::Reduce(a) => {
        let op = a.op.clone().into();
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
