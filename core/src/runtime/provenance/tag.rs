use std::fmt::{Debug, Display};

use super::*;

pub trait Tag: Clone + Debug + Display + 'static {}

#[derive(Clone)]
pub struct Tagged<Tuple: Clone + Ord + Sized, Prov: Provenance> {
  pub tuple: Tuple,
  pub tag: Prov::Tag,
}

impl<Tup: Clone + Ord + Sized, Prov: Provenance> PartialEq for Tagged<Tup, Prov> {
  fn eq(&self, other: &Self) -> bool {
    self.tuple == other.tuple
  }
}

impl<Tup: Clone + Ord + Sized, Prov: Provenance> Eq for Tagged<Tup, Prov> {}

impl<Tup: Clone + Ord + Sized, Prov: Provenance> PartialOrd for Tagged<Tup, Prov> {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.tuple.partial_cmp(&other.tuple)
  }
}

impl<Tup: Clone + Ord + Sized, Prov: Provenance> Ord for Tagged<Tup, Prov> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.tuple.cmp(&other.tuple)
  }
}

impl<Tup, Prov> std::fmt::Debug for Tagged<Tup, Prov>
where
  Tup: Clone + Ord + Sized + std::fmt::Debug,
  Prov: Provenance,
{
  default fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("").field(&self.tuple).field(&self.tag).finish()
  }
}

impl<Tup: Clone + Ord + Sized + std::fmt::Debug> std::fmt::Debug for Tagged<Tup, unit::UnitProvenance> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self.tuple, f)
  }
}

impl<Tup, Prov> std::fmt::Display for Tagged<Tup, Prov>
where
  Tup: Clone + Ord + Sized + std::fmt::Display,
  Prov: Provenance,
{
  default fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{}::{}", self.tag, self.tuple))
  }
}

impl<Tup: Clone + Ord + Sized + std::fmt::Display> std::fmt::Display for Tagged<Tup, unit::UnitProvenance> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Display::fmt(&self.tuple, f)
  }
}

impl Tag for bool {}

impl Tag for usize {}

impl Tag for f64 {}

impl Tag for f32 {}
