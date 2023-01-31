use std::collections::HashSet;
use std::fmt::{Debug, Display};

use rand::distributions::WeightedIndex;
use rand::prelude::*;

use super::*;

use crate::common::tuples::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::statics::*;

pub trait Provenance: Clone + 'static {
  type Tag: Tag;

  type InputTag: Clone + Debug;

  type OutputTag: Clone + Debug + Display;

  fn name() -> &'static str;

  fn tagging_fn(&mut self, ext_tag: Self::InputTag) -> Self::Tag;

  fn tagging_optional_fn(&mut self, ext_tag: Option<Self::InputTag>) -> Self::Tag {
    match ext_tag {
      Some(et) => self.tagging_fn(et),
      None => self.one(),
    }
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag;

  fn discard(&self, t: &Self::Tag) -> bool;

  fn zero(&self) -> Self::Tag;

  fn one(&self) -> Self::Tag;

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag;

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool;

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag;

  fn negate(&self, _: &Self::Tag) -> Option<Self::Tag> {
    None
  }

  fn minus(&self, t1: &Self::Tag, t2: &Self::Tag) -> Option<Self::Tag> {
    self.negate(t2).map(|neg_t2| self.mult(t1, &neg_t2))
  }

  fn weight(&self, _: &Self::Tag) -> f64 {
    1.0
  }

  fn dynamic_count(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    vec![DynamicElement::new(batch.len(), self.one())]
  }

  fn dynamic_sum(&self, ty: &ValueType, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let s = ty.sum(batch.iter_tuples());
    vec![DynamicElement::new(s, self.one())]
  }

  fn dynamic_prod(&self, ty: &ValueType, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let p = ty.prod(batch.iter_tuples());
    vec![DynamicElement::new(p, self.one())]
  }

  fn dynamic_min(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    batch.first().into_iter().cloned().collect()
  }

  fn dynamic_max(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    batch.last().into_iter().cloned().collect()
  }

  fn dynamic_argmin(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    batch
      .iter_tuples()
      .arg_minimum()
      .into_iter()
      .map(|t| DynamicElement::new(t, self.one()))
      .collect()
  }

  fn dynamic_argmax(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    batch
      .iter_tuples()
      .arg_maximum()
      .into_iter()
      .map(|t| DynamicElement::new(t, self.one()))
      .collect()
  }

  fn dynamic_exists(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    vec![DynamicElement::new(!batch.is_empty(), self.one())]
  }

  fn dynamic_top_k(&self, k: usize, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let ids = aggregate_top_k_helper(batch.len(), k, |id| self.weight(&batch[id].tag));
    ids.into_iter().map(|id| batch[id].clone()).collect()
  }

  fn dynamic_categorical_k(
    &self,
    k: usize,
    batch: DynamicElements<Self>,
    rt: &RuntimeEnvironment,
  ) -> DynamicElements<Self> {
    if batch.len() <= k {
      batch
    } else {
      let weights = batch.iter().map(|e| self.weight(&e.tag)).collect::<Vec<_>>();
      let dist = WeightedIndex::new(&weights).unwrap();
      let sampled_ids = (0..k)
        .map(|_| dist.sample(&mut *rt.rng.lock().unwrap()))
        .collect::<HashSet<_>>();
      batch
        .into_iter()
        .enumerate()
        .filter_map(|(i, e)| if sampled_ids.contains(&i) { Some(e) } else { None })
        .collect()
    }
  }

  fn static_count<T: StaticTupleTrait>(&self, batch: StaticElements<T, Self>) -> StaticElements<usize, Self> {
    vec![StaticElement::new(batch.len(), self.one())]
  }

  fn static_sum<T: StaticTupleTrait + SumType>(&self, batch: StaticElements<T, Self>) -> StaticElements<T, Self> {
    vec![StaticElement::new(
      <T as SumType>::sum(batch.iter_tuples().cloned()),
      self.one(),
    )]
  }

  fn static_prod<T: StaticTupleTrait + ProdType>(&self, batch: StaticElements<T, Self>) -> StaticElements<T, Self> {
    vec![StaticElement::new(
      <T as ProdType>::prod(batch.iter_tuples().cloned()),
      self.one(),
    )]
  }

  fn static_max<T: StaticTupleTrait>(&self, batch: StaticElements<T, Self>) -> StaticElements<T, Self> {
    batch.last().into_iter().cloned().collect()
  }

  fn static_min<T: StaticTupleTrait>(&self, batch: StaticElements<T, Self>) -> StaticElements<T, Self> {
    batch.first().into_iter().cloned().collect()
  }

  fn static_argmax<T1: StaticTupleTrait, T2: StaticTupleTrait>(
    &self,
    batch: StaticElements<(T1, T2), Self>,
  ) -> StaticElements<(T1, T2), Self> {
    static_argmax(batch.into_iter().map(|e| e.tuple()))
      .into_iter()
      .map(|t| StaticElement::new(t, self.one()))
      .collect()
  }

  fn static_argmin<T1: StaticTupleTrait, T2: StaticTupleTrait>(
    &self,
    batch: StaticElements<(T1, T2), Self>,
  ) -> StaticElements<(T1, T2), Self> {
    static_argmin(batch.into_iter().map(|e| e.tuple()))
      .into_iter()
      .map(|t| StaticElement::new(t, self.one()))
      .collect()
  }

  fn static_exists<T: StaticTupleTrait>(&self, batch: StaticElements<T, Self>) -> StaticElements<bool, Self> {
    vec![StaticElement::new(!batch.is_empty(), self.one())]
  }

  fn static_top_k<T: StaticTupleTrait>(&self, k: usize, batch: StaticElements<T, Self>) -> StaticElements<T, Self> {
    let ids = aggregate_top_k_helper(batch.len(), k, |id| self.weight(&batch[id].tag));
    ids.into_iter().map(|id| batch[id].clone()).collect()
  }

  fn static_categorical_k<T: StaticTupleTrait>(
    &self,
    k: usize,
    batch: StaticElements<T, Self>,
    rt: &RuntimeEnvironment,
  ) -> StaticElements<T, Self> {
    if batch.len() <= k {
      batch
    } else {
      let weights = batch.iter().map(|e| self.weight(&e.tag)).collect::<Vec<_>>();
      let dist = WeightedIndex::new(&weights).unwrap();
      let sampled_ids = (0..k)
        .map(|_| dist.sample(&mut *rt.rng.lock().unwrap()))
        .collect::<HashSet<_>>();
      batch
        .into_iter()
        .enumerate()
        .filter_map(|(i, e)| if sampled_ids.contains(&i) { Some(e) } else { None })
        .collect()
    }
  }
}

pub type OutputTagOf<C> = <C as Provenance>::OutputTag;

pub type InputTagOf<C> = <C as Provenance>::InputTag;
