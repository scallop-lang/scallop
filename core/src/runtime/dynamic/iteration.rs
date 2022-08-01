use std::collections::*;

use crate::compiler::ram::*;
use crate::runtime::provenance::*;

use super::dataflow::*;
use super::*;

pub struct DynamicIteration<'a, T: Tag> {
  pub iter_num: usize,
  pub input_dynamic_collections: HashMap<String, &'a DynamicCollection<T>>,
  pub dynamic_relations: HashMap<String, DynamicRelation<T>>,
  pub output_relations: Vec<String>,
  pub updates: Vec<Update>,
}

impl<'a, T: Tag> DynamicIteration<'a, T> {
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

  pub fn add_input_dynamic_collection(&mut self, name: &str, col: &'a DynamicCollection<T>) {
    self.input_dynamic_collections.insert(name.to_string(), col);
  }

  pub fn create_dynamic_relation(&mut self, name: &str) {
    self
      .dynamic_relations
      .insert(name.to_string(), DynamicRelation::<T>::new());
  }

  pub fn get_dynamic_relation<'b>(&'b mut self, name: &str) -> Option<&'b DynamicRelation<T>> {
    self.dynamic_relations.get(name).map(|r| r)
  }

  pub fn get_dynamic_relation_unsafe<'b>(&'b mut self, name: &str) -> &'b DynamicRelation<T> {
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

  pub fn run(&'a mut self, ctx: &T::Context) -> HashMap<String, DynamicCollection<T>> {
    self.run_with_iter_limit(ctx, None)
  }

  pub fn run_with_iter_limit(
    &'a mut self,
    ctx: &T::Context,
    iter_limit: Option<usize>,
  ) -> HashMap<String, DynamicCollection<T>> {
    // Iterate until fixpoint
    while self.changed(ctx) || self.is_first_iteration() {
      // Check iter count
      if let Some(iter_limit) = iter_limit {
        if self.iter_num > iter_limit {
          self.changed(ctx);
          break;
        }
      }

      // Perform updates
      for update in &self.updates {
        let dyn_update = self.build_dynamic_update(ctx, update);
        dyn_update.target.insert_dataflow_recent(ctx, &dyn_update.dataflow);
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

  fn changed(&mut self, ctx: &T::Context) -> bool {
    let mut changed = false;
    for (_, relation) in &mut self.dynamic_relations {
      if relation.changed(ctx) {
        changed = true;
      }
    }
    changed
  }

  fn unsafe_get_dynamic_relation(&'a self, name: &str) -> &'a DynamicRelation<T> {
    if self.dynamic_relations.contains_key(name) {
      &self.dynamic_relations[name]
    } else {
      panic!("Cannot find dynamic relation `{}`", name)
    }
  }

  fn unsafe_get_input_dynamic_collection(&'a self, name: &str) -> &'a DynamicCollection<T> {
    if self.input_dynamic_collections.contains_key(name) {
      self.input_dynamic_collections[name]
    } else {
      panic!("Cannot find input dynamic collection `{}`", name)
    }
  }

  fn build_dynamic_update(&'a self, ctx: &'a T::Context, update: &Update) -> DynamicUpdate<'a, T> {
    DynamicUpdate {
      target: self.unsafe_get_dynamic_relation(&update.target),
      dataflow: self.build_dynamic_dataflow(ctx, &update.dataflow),
    }
  }

  fn build_dynamic_dataflow(&'a self, ctx: &'a T::Context, dataflow: &Dataflow) -> DynamicDataflow<'a, T> {
    match dataflow {
      Dataflow::Unit => {
        if self.is_first_iteration() {
          DynamicDataflow::recent_unit(ctx)
        } else {
          DynamicDataflow::stable_unit(ctx)
        }
      }
      Dataflow::Relation(c) => {
        if self.input_dynamic_collections.contains_key(c) {
          self.build_dynamic_collection(c)
        } else {
          self.unsafe_get_dynamic_relation(c).into()
        }
      }
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

  fn build_dynamic_collection(&self, r: &str) -> DynamicDataflow<T> {
    let col = self.unsafe_get_input_dynamic_collection(r);
    DynamicDataflow::dynamic_collection(col, self.is_first_iteration())
  }
}

struct DynamicUpdate<'a, T: Tag> {
  target: &'a DynamicRelation<T>,
  dataflow: DynamicDataflow<'a, T>,
}
