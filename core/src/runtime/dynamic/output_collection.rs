use crate::common::tuple::Tuple;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub struct DynamicOutputCollection<T: Tag> {
  pub elements: Vec<(OutputTagOf<T::Context>, Tuple)>,
}

impl<T: Tag> DynamicOutputCollection<T> {
  /// Whether the collection is empty
  pub fn is_empty(&self) -> bool {
    self.elements.is_empty()
  }

  /// Get the number of elements inside of the collection
  pub fn len(&self) -> usize {
    self.elements.len()
  }

  pub fn iter(&self) -> impl Iterator<Item = &(OutputTagOf<T::Context>, Tuple)> {
    self.elements.iter()
  }

  pub fn ith_tuple(&self, i: usize) -> Option<&Tuple> {
    self.elements.get(i).map(|e| &e.1)
  }

  pub fn ith_tag(&self, i: usize) -> Option<&OutputTagOf<T::Context>> {
    self.elements.get(i).map(|e| &e.0)
  }
}

impl<I, T> From<I> for DynamicOutputCollection<T>
where
  T: Tag,
  I: Iterator<Item = (OutputTagOf<T::Context>, Tuple)>,
{
  fn from(i: I) -> Self {
    Self { elements: i.collect() }
  }
}

impl<T: Tag> std::fmt::Debug for DynamicOutputCollection<T> {
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

impl std::fmt::Debug for DynamicOutputCollection<unit::Unit> {
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

impl<T: Tag> std::fmt::Display for DynamicOutputCollection<T> {
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

impl std::fmt::Display for DynamicOutputCollection<unit::Unit> {
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
