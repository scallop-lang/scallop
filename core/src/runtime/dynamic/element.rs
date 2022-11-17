use crate::common::element::*;
use crate::common::tuple::*;
use crate::runtime::provenance::*;

pub type DynamicElement<Prov> = Tagged<Tuple, Prov>;

impl<Prov: Provenance> DynamicElement<Prov> {
  pub fn new<T1: Into<Tuple>>(tuple: T1, tag: Prov::Tag) -> Self {
    Self {
      tuple: tuple.into(),
      tag,
    }
  }
}

impl<Prov: Provenance> Element<Prov> for DynamicElement<Prov> {
  fn tag(&self) -> &Prov::Tag {
    &self.tag
  }
}

pub type DynamicElements<Prov> = Vec<DynamicElement<Prov>>;

pub trait DynamicTupleIterator<'a, Prov: Provenance> {
  type Output: Iterator<Item = &'a Tuple>;

  fn iter_tuples(&'a self) -> Self::Output;
}

impl<'a, Prov: Provenance + 'static> DynamicTupleIterator<'a, Prov> for DynamicElements<Prov> {
  type Output = DynamicElementsTupleIterator<'a, Prov>;

  fn iter_tuples(&'a self) -> Self::Output {
    DynamicElementsTupleIterator { elements: self.iter() }
  }
}

pub struct DynamicElementsTupleIterator<'a, Prov: Provenance> {
  elements: std::slice::Iter<'a, DynamicElement<Prov>>,
}

impl<'a, Prov: Provenance> Iterator for DynamicElementsTupleIterator<'a, Prov> {
  type Item = &'a Tuple;

  fn next(&mut self) -> Option<Self::Item> {
    self.elements.next().map(|e| &e.tuple)
  }
}

impl<'a, Prov: Provenance> DynamicTupleIterator<'a, Prov> for Vec<&'a DynamicElement<Prov>> {
  type Output = DynamicElementsTupleRefIterator<'a, Prov>;

  fn iter_tuples(&'a self) -> DynamicElementsTupleRefIterator<'a, Prov> {
    DynamicElementsTupleRefIterator { elements: self.iter() }
  }
}

pub struct DynamicElementsTupleRefIterator<'a, Prov: Provenance> {
  elements: std::slice::Iter<'a, &'a DynamicElement<Prov>>,
}

impl<'a, Prov: Provenance> Iterator for DynamicElementsTupleRefIterator<'a, Prov> {
  type Item = &'a Tuple;

  fn next(&mut self) -> Option<Self::Item> {
    self.elements.next().map(|e| &e.tuple)
  }
}
