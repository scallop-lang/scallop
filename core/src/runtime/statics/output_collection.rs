use crate::common::tuple::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::StaticTupleTrait;

#[derive(Clone)]
pub struct StaticOutputCollection<Tup: StaticTupleTrait, Prov: Provenance> {
  pub elements: Vec<(OutputTagOf<Prov>, Tup)>,
}

impl<Tup: StaticTupleTrait, Prov: Provenance> StaticOutputCollection<Tup, Prov> {
  /// Whether the collection is empty
  pub fn is_empty(&self) -> bool {
    self.elements.is_empty()
  }

  /// Get the number of elements inside of the collection
  pub fn len(&self) -> usize {
    self.elements.len()
  }

  pub fn iter(&self) -> impl Iterator<Item = &(OutputTagOf<Prov>, Tup)> {
    self.elements.iter()
  }

  pub fn ith_tuple(&self, i: usize) -> Option<&Tup> {
    self.elements.get(i).map(|e| &e.1)
  }

  pub fn ith_tag(&self, i: usize) -> Option<&OutputTagOf<Prov>> {
    self.elements.get(i).map(|e| &e.0)
  }

  pub fn to_dynamic_vec(&self) -> Vec<(OutputTagOf<Prov>, Tuple)>
  where
    Tup: Into<Tuple>,
  {
    self
      .elements
      .iter()
      .map(|(tag, tup)| (tag.clone(), tup.clone().into()))
      .collect()
  }
}

impl<I, Prov, Tup> From<I> for StaticOutputCollection<Tup, Prov>
where
  Prov: Provenance,
  I: Iterator<Item = (OutputTagOf<Prov>, Tup)>,
  Tup: StaticTupleTrait,
{
  fn from(i: I) -> Self {
    Self { elements: i.collect() }
  }
}

impl<Prov: Provenance, Tup: StaticTupleTrait> std::fmt::Debug for StaticOutputCollection<Tup, Prov> {
  default fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, (tag, tuple)) in self.elements.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{:?}::{:?}", tag, tuple))?;
    }
    f.write_str("}")
  }
}

impl<Tup: StaticTupleTrait> std::fmt::Debug for StaticOutputCollection<Tup, unit::UnitProvenance> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, (_, tuple)) in self.elements.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{:?}", tuple))?;
    }
    f.write_str("}")
  }
}

impl<Prov: Provenance, Tup: StaticTupleTrait> std::fmt::Display for StaticOutputCollection<Tup, Prov> {
  default fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, (tag, tuple)) in self.elements.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{}::{:?}", tag, tuple))?;
    }
    f.write_str("}")
  }
}

impl<Tup: StaticTupleTrait> std::fmt::Display for StaticOutputCollection<Tup, unit::UnitProvenance> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, (_, tuple)) in self.elements.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{:?}", tuple))?;
    }
    f.write_str("}")
  }
}
