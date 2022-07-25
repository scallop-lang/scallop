use std::fmt::{Debug, Display};

use super::unit::Unit;
use crate::runtime::dynamic::*;

pub enum Proceeding {
  Stable,
  Recent,
}

pub trait ProvenanceContext: Clone {
  type Tag: Tag<Context = Self>;

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

  fn tagging_disjunction_fn(&mut self, tags: Vec<Self::InputTag>) -> Vec<Self::Tag> {
    tags.into_iter().map(|t| self.tagging_fn(t)).collect()
  }

  fn tagging_disjunction_optional_fn(
    &mut self,
    tags: Vec<Option<Self::InputTag>>,
  ) -> Vec<Self::Tag> {
    // First generate disjunctive tags for those with a tag
    let disj_tags = self.tagging_disjunction_fn(tags.iter().filter_map(|t| t.clone()).collect());

    // Then fill in self.one() for the None tags
    let mut disj_tag_index = 0;
    let mut all_return_tags = vec![];
    for tag in tags {
      if let Some(_) = tag {
        all_return_tags.push(disj_tags[disj_tag_index].clone());
        disj_tag_index += 1;
      } else {
        all_return_tags.push(self.one());
      }
    }
    all_return_tags
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag;

  fn discard(&self, t: &Self::Tag) -> bool;

  fn zero(&self) -> Self::Tag;

  fn one(&self) -> Self::Tag;

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag;

  fn add_with_proceeding(&self, t1: &Self::Tag, t2: &Self::Tag) -> (Self::Tag, Proceeding) {
    (self.add(t1, t2), Proceeding::Stable)
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag;

  fn negate(&self, _: &Self::Tag) -> Option<Self::Tag> {
    None
  }

  fn minus(&self, t1: &Self::Tag, t2: &Self::Tag) -> Option<Self::Tag> {
    self.negate(t2).map(|neg_t2| self.mult(t1, &neg_t2))
  }

  /// Aggregate dynamically on a set of elements
  ///
  /// Precondition:
  /// - the batch elements are normalized (no duplicate tuples)
  /// - the batch elements are sorted
  fn dynamic_aggregate(
    &self,
    op: &DynamicAggregateOp,
    batch: DynamicElements<Self::Tag>,
  ) -> DynamicElements<Self::Tag> {
    use DynamicAggregateOp::*;
    match op {
      Count(c) => self.dynamic_count(c, batch),
      Sum(c) => self.dynamic_sum(c, batch),
      Prod(c) => self.dynamic_prod(c, batch),
      Min(c) => self.dynamic_min(c, batch),
      Max(c) => self.dynamic_max(c, batch),
      Exists(c) => self.dynamic_exists(c, batch),
      Unique(c) => self.dynamic_unique(c, batch),
    }
  }

  fn dynamic_count<'a>(
    &self,
    op: &DynamicCountOp,
    batch: DynamicElements<Self::Tag>,
  ) -> DynamicElements<Self::Tag> {
    op.aggregate(batch, self)
      .into_iter()
      .map(|tuple| DynamicElement::new(tuple, self.one()))
      .collect()
  }

  fn dynamic_sum<'a>(
    &self,
    op: &DynamicSumOp,
    batch: DynamicElements<Self::Tag>,
  ) -> DynamicElements<Self::Tag> {
    op.aggregate(batch, self)
      .into_iter()
      .map(|tuple| DynamicElement::new(tuple, self.one()))
      .collect()
  }

  fn dynamic_prod<'a>(
    &self,
    op: &DynamicProdOp,
    batch: DynamicElements<Self::Tag>,
  ) -> DynamicElements<Self::Tag> {
    op.aggregate(batch, self)
      .into_iter()
      .map(|tuple| DynamicElement::new(tuple, self.one()))
      .collect()
  }

  fn dynamic_min<'a>(
    &self,
    op: &DynamicMinOp,
    batch: DynamicElements<Self::Tag>,
  ) -> DynamicElements<Self::Tag> {
    op.aggregate(batch, self)
      .into_iter()
      .map(|tuple| DynamicElement::new(tuple, self.one()))
      .collect()
  }

  fn dynamic_max<'a>(
    &self,
    op: &DynamicMaxOp,
    batch: DynamicElements<Self::Tag>,
  ) -> DynamicElements<Self::Tag> {
    op.aggregate(batch, self)
      .into_iter()
      .map(|tuple| DynamicElement::new(tuple, self.one()))
      .collect()
  }

  fn dynamic_exists<'a>(
    &self,
    op: &DynamicExistsOp,
    batch: DynamicElements<Self::Tag>,
  ) -> DynamicElements<Self::Tag> {
    op.aggregate(batch, self)
      .into_iter()
      .map(|tuple| DynamicElement::new(tuple, self.one()))
      .collect()
  }

  fn dynamic_unique<'a>(
    &self,
    op: &DynamicUniqueOp,
    batch: DynamicElements<Self::Tag>,
  ) -> DynamicElements<Self::Tag> {
    op.aggregate(batch, self)
      .into_iter()
      .map(|tuple| DynamicElement::new(tuple, self.one()))
      .collect()
  }
}

pub type OutputTagOf<C> = <C as ProvenanceContext>::OutputTag;

pub type InputTagOf<C> = <C as ProvenanceContext>::InputTag;

pub trait Tag: Clone + Debug + Display + 'static {
  type Context: ProvenanceContext<Tag = Self>;
}

#[derive(Clone)]
pub struct Tagged<Tuple: Clone + Ord + Sized, T: Tag> {
  pub tuple: Tuple,
  pub tag: T,
}

impl<Tup: Clone + Ord + Sized, T: Tag> PartialEq for Tagged<Tup, T> {
  fn eq(&self, other: &Self) -> bool {
    self.tuple == other.tuple
  }
}

impl<Tup: Clone + Ord + Sized, T: Tag> Eq for Tagged<Tup, T> {}

impl<Tup: Clone + Ord + Sized, T: Tag> PartialOrd for Tagged<Tup, T> {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.tuple.partial_cmp(&other.tuple)
  }
}

impl<Tup: Clone + Ord + Sized, T: Tag> Ord for Tagged<Tup, T> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.tuple.cmp(&other.tuple)
  }
}

impl<Tup, T> std::fmt::Debug for Tagged<Tup, T>
where
  Tup: Clone + Ord + Sized + std::fmt::Debug,
  T: Tag,
{
  default fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("")
      .field(&self.tuple)
      .field(&self.tag)
      .finish()
  }
}

impl<Tup: Clone + Ord + Sized + std::fmt::Debug> std::fmt::Debug for Tagged<Tup, Unit> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self.tuple, f)
  }
}

impl<Tup, T> std::fmt::Display for Tagged<Tup, T>
where
  Tup: Clone + Ord + Sized + std::fmt::Display,
  T: Tag,
{
  default fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}::{}", self.tag, self.tuple))
  }
}

impl<Tup: Clone + Ord + Sized + std::fmt::Display> std::fmt::Display for Tagged<Tup, Unit> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Display::fmt(&self.tuple, f)
  }
}
