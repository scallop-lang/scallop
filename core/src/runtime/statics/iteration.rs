use super::*;
use crate::runtime::provenance::*;

pub struct StaticIteration<'a, T: Tag> {
  pub iter_num: usize,
  pub relations: Vec<Box<dyn StaticRelationTrait<T>>>,
  pub provenance_context: &'a mut T::Context,
}

impl<'a, T: Tag> StaticIteration<'a, T> {
  /// Create a new Iteration
  pub fn new(provenance_context: &'a mut T::Context) -> Self {
    Self {
      provenance_context,
      relations: Vec::new(),
      iter_num: 0,
    }
  }

  pub fn is_first_iteration(&self) -> bool {
    self.iter_num == 0
  }

  pub fn step(&mut self) {
    self.iter_num += 1;
  }

  pub fn changed(&mut self) -> bool {
    let mut result = false;
    for relation in self.relations.iter_mut() {
      if relation.changed(&self.provenance_context) {
        result = true;
      }
    }
    result
  }

  pub fn create_relation<Tup>(&mut self) -> StaticRelation<Tup, T>
  where
    Tup: StaticTupleTrait + 'static,
  {
    let relation = StaticRelation::new();
    self.relations.push(Box::new(relation.clone()));
    relation
  }

  pub fn insert_dataflow<D, Tup>(&self, r: &StaticRelation<Tup, T>, data: D)
  where
    D: dataflow::Dataflow<Tup, T>,
    Tup: StaticTupleTrait,
  {
    r.insert_dataflow_recent(&self.provenance_context, data)
  }

  pub fn product<D1, D2, T1, T2>(&self, v1: D1, v2: D2) -> dataflow::Product<D1, D2, T1, T2, T>
  where
    T1: StaticTupleTrait,
    T2: StaticTupleTrait,
    D1: dataflow::Dataflow<T1, T>,
    D2: dataflow::Dataflow<T2, T>,
  {
    dataflow::product(v1, v2, &self.provenance_context)
  }

  pub fn intersect<D1, D2, Tup>(&self, v1: D1, v2: D2) -> dataflow::Intersection<D1, D2, Tup, T>
  where
    Tup: StaticTupleTrait,
    D1: dataflow::Dataflow<Tup, T>,
    D2: dataflow::Dataflow<Tup, T>,
  {
    dataflow::intersect(v1, v2, &self.provenance_context)
  }

  pub fn union<D1, D2, Tup>(&self, v1: D1, v2: D2) -> dataflow::Union<D1, D2, Tup, T>
  where
    Tup: StaticTupleTrait,
    D1: dataflow::Dataflow<Tup, T>,
    D2: dataflow::Dataflow<Tup, T>,
  {
    dataflow::union(v1, v2, &self.provenance_context)
  }

  pub fn join<D1, D2, K, T1, T2>(&self, v1: D1, v2: D2) -> dataflow::Join<D1, D2, K, T1, T2, T>
  where
    K: StaticTupleTrait,
    T1: StaticTupleTrait,
    T2: StaticTupleTrait,
    D1: dataflow::Dataflow<(K, T1), T>,
    D2: dataflow::Dataflow<(K, T2), T>,
  {
    dataflow::join(v1, v2, &self.provenance_context)
  }

  pub fn difference<D1, D2, Tup>(&self, v1: D1, v2: D2) -> dataflow::Difference<D1, D2, Tup, T>
  where
    Tup: StaticTupleTrait,
    D1: dataflow::Dataflow<Tup, T>,
    D2: dataflow::Dataflow<Tup, T>,
  {
    dataflow::difference(v1, v2, &self.provenance_context)
  }

  pub fn antijoin<D1, D2, K, T1>(&self, v1: D1, v2: D2) -> dataflow::Antijoin<D1, D2, K, T1, T>
  where
    K: StaticTupleTrait,
    T1: StaticTupleTrait,
    D1: dataflow::Dataflow<(K, T1), T>,
    D2: dataflow::Dataflow<K, T>,
  {
    dataflow::antijoin(v1, v2, &self.provenance_context)
  }

  pub fn aggregate<A, D, T1>(&self, agg: A, d: D) -> dataflow::AggregationSingleGroup<A, D, T1, T>
  where
    A: Aggregator<T1, T>,
    T1: StaticTupleTrait,
    D: dataflow::Dataflow<T1, T>,
  {
    dataflow::AggregationSingleGroup::new(agg, d, &self.provenance_context)
  }

  pub fn aggregate_implicit_group<A, D, K, T1>(
    &self,
    agg: A,
    d: D,
  ) -> dataflow::AggregationImplicitGroup<A, D, K, T1, T>
  where
    A: Aggregator<T1, T>,
    K: StaticTupleTrait,
    T1: StaticTupleTrait,
    D: dataflow::Dataflow<(K, T1), T>,
  {
    dataflow::AggregationImplicitGroup::new(agg, d, &self.provenance_context)
  }

  pub fn aggregate_join_group<A, D1, D2, K, T1, T2>(
    &self,
    agg: A,
    v1: D1,
    v2: D2,
  ) -> dataflow::AggregationJoinGroup<A, D1, D2, K, T1, T2, T>
  where
    A: Aggregator<T2, T>,
    K: StaticTupleTrait,
    T1: StaticTupleTrait,
    T2: StaticTupleTrait,
    D1: dataflow::Dataflow<(K, T1), T>,
    D2: dataflow::Dataflow<(K, T2), T>,
  {
    dataflow::AggregationJoinGroup::new(agg, v1, v2, &self.provenance_context)
  }

  pub fn complete<Tup>(&self, r: &StaticRelation<Tup, T>) -> StaticCollection<Tup, T>
  where
    Tup: StaticTupleTrait,
  {
    r.complete(&self.provenance_context)
  }
}
