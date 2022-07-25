use crate::runtime::provenance::*;
use crate::runtime::statics::StaticTupleTrait;

#[derive(Clone)]
pub struct StaticOutputCollection<Tup: StaticTupleTrait, T: Tag> {
  pub elements: Vec<(OutputTagOf<T::Context>, Tup)>,
}

impl<Tup: StaticTupleTrait, T: Tag> StaticOutputCollection<Tup, T> {
  /// Whether the collection is empty
  pub fn is_empty(&self) -> bool {
    self.elements.is_empty()
  }

  /// Get the number of elements inside of the collection
  pub fn len(&self) -> usize {
    self.elements.len()
  }

  pub fn iter(&self) -> impl Iterator<Item = &(OutputTagOf<T::Context>, Tup)> {
    self.elements.iter()
  }

  pub fn ith_tuple(&self, i: usize) -> Option<&Tup> {
    self.elements.get(i).map(|e| &e.1)
  }

  pub fn ith_tag(&self, i: usize) -> Option<&OutputTagOf<T::Context>> {
    self.elements.get(i).map(|e| &e.0)
  }
}

impl<I, T, Tup> From<I> for StaticOutputCollection<Tup, T>
where
  T: Tag,
  I: Iterator<Item = (OutputTagOf<T::Context>, Tup)>,
  Tup: StaticTupleTrait,
{
  fn from(i: I) -> Self {
    Self {
      elements: i.collect(),
    }
  }
}

impl<T: Tag, Tup: StaticTupleTrait> std::fmt::Debug for StaticOutputCollection<Tup, T> {
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

impl<Tup: StaticTupleTrait> std::fmt::Debug for StaticOutputCollection<Tup, unit::Unit> {
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

impl<T: Tag, Tup: StaticTupleTrait> std::fmt::Display for StaticOutputCollection<Tup, T> {
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

impl<Tup: StaticTupleTrait> std::fmt::Display for StaticOutputCollection<Tup, unit::Unit> {
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
