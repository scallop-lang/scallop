use std::collections::*;

use crate::compiler::ram::*;
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
  pub updates: Vec<Update>,
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

  pub fn get_dynamic_relation<'c>(&'c mut self, name: &str) -> Option<&'c DynamicRelation<Prov>> {
    self.dynamic_relations.get(name).map(|r| r)
  }

  pub fn get_dynamic_relation_unsafe<'c>(&'c mut self, name: &str) -> &'c DynamicRelation<Prov> {
    self.dynamic_relations.get(name).map(|r| r).unwrap()
  }

  pub fn add_update_dataflow(&mut self, target: &str, dataflow: Dataflow) {
    self.add_update(Update {
      target: target.to_string(),
      dataflow,
    });
  }

  pub fn add_update(&mut self, update: Update) {
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
        let dyn_update = self.build_dynamic_update(ctx, update);
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
        let dyn_update = self.build_dynamic_update(ctx, update);
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

  fn build_dynamic_update(&'a self, ctx: &'a Prov, update: &Update) -> DynamicUpdate<'a, Prov> {
    DynamicUpdate {
      target: self.unsafe_get_dynamic_relation(&update.target),
      dataflow: self.build_dynamic_dataflow(ctx, &update.dataflow),
    }
  }

  fn build_dynamic_dataflow(&'a self, ctx: &'a Prov, dataflow: &Dataflow) -> DynamicDataflow<'a, Prov> {
    match dataflow {
      Dataflow::Unit(t) => {
        if self.is_first_iteration() {
          DynamicDataflow::recent_unit(ctx, t.clone())
        } else {
          DynamicDataflow::stable_unit(ctx, t.clone())
        }
      }
      Dataflow::Relation(c) => {
        if self.input_dynamic_collections.contains_key(c) {
          self.build_dynamic_collection(c)
        } else {
          self.unsafe_get_dynamic_relation(c).into()
        }
      }
      Dataflow::OverwriteOne(d) => self.build_dynamic_dataflow(ctx, d).overwrite_one(ctx),
      Dataflow::Filter(d, e) => self.build_dynamic_dataflow(ctx, d).filter(e.clone()),
      Dataflow::Find(d, k) => self.build_dynamic_dataflow(ctx, d).find(k.clone()),
      Dataflow::Project(d, e) => self.build_dynamic_dataflow(ctx, d).project(e.clone()),
      Dataflow::Intersect(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(ctx, d1);
        let r2 = self.build_dynamic_dataflow(ctx, d2);
        r1.intersect(r2, ctx)
      }
      Dataflow::Join(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(ctx, d1);
        let r2 = self.build_dynamic_dataflow(ctx, d2);
        r1.join(r2, ctx)
      }
      Dataflow::Product(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(ctx, d1);
        let r2 = self.build_dynamic_dataflow(ctx, d2);
        r1.product(r2, ctx)
      }
      Dataflow::Union(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(ctx, d1);
        let r2 = self.build_dynamic_dataflow(ctx, d2);
        r1.union(r2)
      }
      Dataflow::Difference(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(ctx, d1);
        let r2 = self.build_dynamic_dataflow(ctx, d2);
        r1.difference(r2, ctx)
      }
      Dataflow::Antijoin(d1, d2) => {
        let r1 = self.build_dynamic_dataflow(ctx, d1);
        let r2 = self.build_dynamic_dataflow(ctx, d2);
        r1.antijoin(r2, ctx)
      }
      Dataflow::Reduce(a) => {
        let op = a.op.clone().into();
        match &a.group_by {
          ReduceGroupByType::None => {
            let c = self.build_dynamic_collection(&a.predicate);
            DynamicAggregationDataflow::single(op, c, ctx).into()
          }
          ReduceGroupByType::Implicit => {
            let c = self.build_dynamic_collection(&a.predicate);
            DynamicAggregationDataflow::implicit(op, c, ctx).into()
          }
          ReduceGroupByType::Join(other) => {
            let c = self.build_dynamic_collection(&a.predicate);
            let g = self.build_dynamic_collection(&other);
            DynamicAggregationDataflow::join(op, g, c, ctx).into()
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
