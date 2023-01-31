use crate::common::tuple::Tuple;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub struct DynamicOutputCollection<Prov: Provenance> {
  pub elements: Vec<(OutputTagOf<Prov>, Tuple)>,
}

impl<Prov: Provenance> DynamicOutputCollection<Prov> {
  pub fn empty() -> Self {
    Self { elements: vec![] }
  }

  /// Whether the collection is empty
  pub fn is_empty(&self) -> bool {
    self.elements.is_empty()
  }

  /// Get the number of elements inside of the collection
  pub fn len(&self) -> usize {
    self.elements.len()
  }

  pub fn iter(&self) -> impl Iterator<Item = &(OutputTagOf<Prov>, Tuple)> {
    self.elements.iter()
  }

  pub fn ith_tuple(&self, i: usize) -> Option<&Tuple> {
    self.elements.get(i).map(|e| &e.1)
  }

  pub fn ith_tag(&self, i: usize) -> Option<&OutputTagOf<Prov>> {
    self.elements.get(i).map(|e| &e.0)
  }

  pub fn extend<I>(&mut self, iter: I)
  where
    I: Iterator<Item = (Prov::OutputTag, Tuple)>,
  {
    self.elements.extend(iter)
  }
}

impl<I, Prov> From<I> for DynamicOutputCollection<Prov>
where
  Prov: Provenance,
  I: Iterator<Item = (OutputTagOf<Prov>, Tuple)>,
{
  fn from(i: I) -> Self {
    Self { elements: i.collect() }
  }
}

impl<Prov: Provenance> std::fmt::Debug for DynamicOutputCollection<Prov> {
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

impl std::fmt::Debug for DynamicOutputCollection<unit::UnitProvenance> {
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

impl<Prov: Provenance> std::fmt::Display for DynamicOutputCollection<Prov> {
  default fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, (tag, tuple)) in self.elements.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{}::{}", tag, tuple))?;
    }
    f.write_str("}")
  }
}

impl std::fmt::Display for DynamicOutputCollection<unit::UnitProvenance> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, (_, tuple)) in self.elements.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      f.write_fmt(format_args!("{}", tuple))?;
    }
    f.write_str("}")
  }
}
