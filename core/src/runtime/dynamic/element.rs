use crate::common::element::*;
use crate::common::tuple::*;
use crate::runtime::provenance::*;

pub type DynamicElement<T> = Tagged<Tuple, T>;

impl<T: Tag> DynamicElement<T> {
  pub fn new<T1: Into<Tuple>>(tuple: T1, tag: T) -> Self {
    Self {
      tuple: tuple.into(),
      tag,
    }
  }
}

impl<T: Tag> Element<T> for DynamicElement<T> {
  fn tag(&self) -> &T {
    &self.tag
  }
}

pub type DynamicElements<T> = Vec<DynamicElement<T>>;

pub trait DynamicTupleIterator<'a, T: Tag> {
  type Output: Iterator<Item = &'a Tuple>;

  fn iter_tuples(&'a self) -> Self::Output;
}

impl<'a, T: Tag> DynamicTupleIterator<'a, T> for DynamicElements<T> {
  type Output = DynamicElementsTupleIterator<'a, T>;

  fn iter_tuples(&'a self) -> Self::Output {
    DynamicElementsTupleIterator { elements: self.iter() }
  }
}

pub struct DynamicElementsTupleIterator<'a, T: Tag> {
  elements: std::slice::Iter<'a, DynamicElement<T>>,
}

impl<'a, T: Tag> Iterator for DynamicElementsTupleIterator<'a, T> {
  type Item = &'a Tuple;

  fn next(&mut self) -> Option<Self::Item> {
    self.elements.next().map(|e| &e.tuple)
  }
}

impl<'a, T: Tag> DynamicTupleIterator<'a, T> for Vec<&'a DynamicElement<T>> {
  type Output = DynamicElementsTupleRefIterator<'a, T>;

  fn iter_tuples(&'a self) -> DynamicElementsTupleRefIterator<'a, T> {
    DynamicElementsTupleRefIterator { elements: self.iter() }
  }
}

pub struct DynamicElementsTupleRefIterator<'a, T: Tag> {
  elements: std::slice::Iter<'a, &'a DynamicElement<T>>,
}

impl<'a, T: Tag> Iterator for DynamicElementsTupleRefIterator<'a, T> {
  type Item = &'a Tuple;

  fn next(&mut self) -> Option<Self::Item> {
    self.elements.next().map(|e| &e.tuple)
  }
}
