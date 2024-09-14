use std::fmt::{Debug, Display};

use super::*;

/// A provenance
pub trait Provenance: Clone + 'static {
  /// The input tag space of the provenance
  type InputTag: Clone + Debug + StaticInputTag;

  /// The (internal) tag space of the provenance
  type Tag: Tag;

  /// The output tag space of the provenance
  type OutputTag: Clone + Debug + Display;

  /// The name of the provenance
  fn name(&self) -> String;

  /// Converting input tag to internal tag
  fn tagging_fn(&self, ext_tag: Self::InputTag) -> Self::Tag;

  /// Converting a maybe input tag to internal tag;
  /// if the input tag does not exist, we use the `one` tag.
  ///
  /// Custom provenance may overwrite this to get special behavior when
  /// there is no input tag
  fn tagging_optional_fn(&self, ext_tag: Option<Self::InputTag>) -> Self::Tag {
    match ext_tag {
      Some(et) => self.tagging_fn(et),
      None => self.one(),
    }
  }

  /// Convert the internal tag to the output tag
  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag;

  /// Check if we want to discard a fact with the given tag
  fn discard(&self, t: &Self::Tag) -> bool;

  /// The `zero` element in the internal tag space
  fn zero(&self) -> Self::Tag;

  /// The `one` element in the internal tag space
  fn one(&self) -> Self::Tag;

  /// Adding two tags
  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag;

  /// Multiply two tags
  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag;

  /// Negate a tag.
  ///
  /// If `None` is returned, the tuple will be discarded/removed.
  /// By default (if not implemented), negating a tag results in `None`,
  #[allow(unused)]
  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    None
  }

  fn minus(&self, t1: &Self::Tag, t2: &Self::Tag) -> Option<Self::Tag> {
    self.negate(t2).map(|neg_t2| self.mult(t1, &neg_t2))
  }

  /// Check if a tag has saturated given its old and new versions
  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool;

  /// Get the weight of a tag
  ///
  /// By default (if not implemented), every tag are weighted equally by having a weight of 1
  #[allow(unused)]
  fn weight(&self, tag: &Self::Tag) -> f64 {
    1.0
  }
}

pub type OutputTagOf<C> = <C as Provenance>::OutputTag;

pub type InputTagOf<C> = <C as Provenance>::InputTag;
