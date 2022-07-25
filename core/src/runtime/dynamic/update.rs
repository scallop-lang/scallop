use std::collections::*;

use super::*;
use crate::common::expr::Expr;
use crate::common::tuple::Tuple;
use crate::compiler::ram::{self, ReduceGroupByType};

#[derive(Clone)]
pub enum Dataflow {
  Unit,
  DynamicCollection(String),
  DynamicRelation(String),
  Filter(Box<Dataflow>, Expr),
  Find(Box<Dataflow>, Tuple),
  Project(Box<Dataflow>, Expr),
  Intersect(Box<Dataflow>, Box<Dataflow>),
  Join(Box<Dataflow>, Box<Dataflow>),
  Product(Box<Dataflow>, Box<Dataflow>),
  Union(Box<Dataflow>, Box<Dataflow>),
  Difference(Box<Dataflow>, Box<Dataflow>),
  Antijoin(Box<Dataflow>, Box<Dataflow>),
  Aggregation(Groups, DynamicAggregateOp),
}

impl Dataflow {
  pub fn from_ram(df: &ram::Dataflow, dyn_relas: &HashSet<String>) -> Self {
    match df {
      ram::Dataflow::Unit => Dataflow::Unit,
      ram::Dataflow::Union(d1, d2) => {
        let d1 = Self::from_ram(d1, dyn_relas);
        let d2 = Self::from_ram(d2, dyn_relas);
        Dataflow::union(d1, d2)
      }
      ram::Dataflow::Join(d1, d2) => {
        let d1 = Self::from_ram(d1, dyn_relas);
        let d2 = Self::from_ram(d2, dyn_relas);
        Dataflow::join(d1, d2)
      }
      ram::Dataflow::Intersect(d1, d2) => {
        let d1 = Self::from_ram(d1, dyn_relas);
        let d2 = Self::from_ram(d2, dyn_relas);
        Dataflow::intersect(d1, d2)
      }
      ram::Dataflow::Product(d1, d2) => {
        let d1 = Self::from_ram(d1, dyn_relas);
        let d2 = Self::from_ram(d2, dyn_relas);
        Dataflow::product(d1, d2)
      }
      ram::Dataflow::Antijoin(d1, d2) => {
        let d1 = Self::from_ram(d1, dyn_relas);
        let d2 = Self::from_ram(d2, dyn_relas);
        Dataflow::antijoin(d1, d2)
      }
      ram::Dataflow::Difference(d1, d2) => {
        let d1 = Self::from_ram(d1, dyn_relas);
        let d2 = Self::from_ram(d2, dyn_relas);
        Dataflow::difference(d1, d2)
      }
      ram::Dataflow::Project(d, p) => {
        let d = Self::from_ram(d, dyn_relas);
        Dataflow::project(d, p.clone())
      }
      ram::Dataflow::Filter(d, f) => {
        let d = Self::from_ram(d, dyn_relas);
        Dataflow::filter(d, f.clone())
      }
      ram::Dataflow::Find(d, t) => {
        let d = Self::from_ram(d, dyn_relas);
        Dataflow::find(d, t.clone())
      }
      ram::Dataflow::Reduce(r) => {
        let group = match &r.group_by {
          ReduceGroupByType::None => Groups::SingleCollection(r.predicate.clone()),
          ReduceGroupByType::Implicit => Groups::GroupedByKey(r.predicate.clone()),
          ReduceGroupByType::Join(group_by_pred) => {
            Groups::GroupByJoinCollection(group_by_pred.clone(), r.predicate.clone())
          }
        };
        group.aggregate(r.op.clone())
      }
      ram::Dataflow::Relation(r) => {
        if dyn_relas.contains(r) {
          Dataflow::DynamicRelation(r.clone())
        } else {
          Dataflow::DynamicCollection(r.clone())
        }
      }
    }
  }

  pub fn dynamic_collection(name: &str) -> Self {
    Self::DynamicCollection(name.to_string())
  }

  pub fn dynamic_relation(name: &str) -> Self {
    Self::DynamicRelation(name.to_string())
  }

  pub fn filter<E: Into<Expr>>(self, expr: E) -> Self {
    Self::Filter(Box::new(self), expr.into())
  }

  pub fn find<T: Into<Tuple>>(self, tuple: T) -> Self {
    Self::Find(Box::new(self), tuple.into())
  }

  pub fn project<E: Into<Expr>>(self, expr: E) -> Self {
    Self::Project(Box::new(self), expr.into())
  }

  pub fn intersect(self, other: Self) -> Self {
    Self::Intersect(Box::new(self), Box::new(other))
  }

  pub fn join(self, other: Self) -> Self {
    Self::Join(Box::new(self), Box::new(other))
  }

  pub fn product(self, other: Self) -> Self {
    Self::Product(Box::new(self), Box::new(other))
  }

  pub fn union(self, other: Self) -> Self {
    Self::Union(Box::new(self), Box::new(other))
  }

  pub fn difference(self, other: Self) -> Self {
    Self::Difference(Box::new(self), Box::new(other))
  }

  pub fn antijoin(self, other: Self) -> Self {
    Self::Antijoin(Box::new(self), Box::new(other))
  }
}

#[derive(Clone)]
pub enum Groups {
  SingleCollection(String),
  GroupedByKey(String),
  GroupByJoinCollection(String, String),
}

impl Groups {
  pub fn single_collection(name: &str) -> Self {
    Self::SingleCollection(name.to_string())
  }

  pub fn keyed_groups(name: &str) -> Self {
    Self::GroupedByKey(name.to_string())
  }

  pub fn group_by_join_collection(key_name: &str, name: &str) -> Self {
    Self::GroupByJoinCollection(key_name.to_string(), name.to_string())
  }

  pub fn aggregate(self, op: DynamicAggregateOp) -> Dataflow {
    Dataflow::Aggregation(self, op)
  }
}

#[derive(Clone)]
pub struct Update {
  pub target: String,
  pub dataflow: Dataflow,
}

impl Update {
  pub fn new(target: &str, dataflow: Dataflow) -> Self {
    Self {
      target: target.to_string(),
      dataflow,
    }
  }

  pub fn from_ram(update: &ram::Update, dyn_relas: &HashSet<String>) -> Self {
    Self {
      target: update.target.clone(),
      dataflow: Dataflow::from_ram(&update.dataflow, dyn_relas),
    }
  }
}
