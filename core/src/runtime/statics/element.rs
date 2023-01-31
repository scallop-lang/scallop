use crate::common::element::*;
use crate::runtime::provenance::*;

use super::*;

pub type StaticElement<Tup, Prov> = Tagged<StaticTupleWrapper<Tup>, Prov>;

impl<Tup, Prov> StaticElement<Tup, Prov>
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  pub fn new(tuple: Tup, tag: Prov::Tag) -> Self {
    Self {
      tuple: StaticTupleWrapper::new(tuple),
      tag,
    }
  }

  pub fn tuple(self) -> Tup {
    self.tuple.into()
  }
}

impl<Tup: StaticTupleTrait, Prov: Provenance> Element<Prov> for StaticElement<Tup, Prov> {
  fn tag(&self) -> &Prov::Tag {
    &self.tag
  }
}

impl<Tup: StaticTupleTrait, Prov: Provenance> Into<(Tup, Prov::Tag)> for StaticElement<Tup, Prov> {
  fn into(self) -> (Tup, Prov::Tag) {
    let Self { tuple, tag } = self;
    (tuple.into(), tag)
  }
}

pub type StaticElements<Tup, Prov> = Vec<StaticElement<Tup, Prov>>;

pub trait StaticTupleIterator<'a, Tup: 'static + StaticTupleTrait> {
  type Output: Iterator<Item = &'a Tup>;

  fn iter_tuples(&'a self) -> Self::Output;
}

impl<'a, Tup: 'static + StaticTupleTrait, Prov: 'static + Provenance> StaticTupleIterator<'a, Tup>
  for StaticElements<Tup, Prov>
{
  type Output = StaticElementsTupleIterator<'a, Tup, Prov>;

  fn iter_tuples(&'a self) -> Self::Output {
    StaticElementsTupleIterator { elements: self.iter() }
  }
}

pub struct StaticElementsTupleIterator<'a, Tup: StaticTupleTrait, Prov: Provenance> {
  elements: std::slice::Iter<'a, StaticElement<Tup, Prov>>,
}

impl<'a, Tup: StaticTupleTrait, Prov: Provenance> Iterator for StaticElementsTupleIterator<'a, Tup, Prov> {
  type Item = &'a Tup;

  fn next(&mut self) -> Option<Self::Item> {
    self.elements.next().map(|e| e.tuple.get())
  }
}

impl<'a, Tup: 'static + StaticTupleTrait, Prov: Provenance> StaticTupleIterator<'a, Tup>
  for Vec<&'a StaticElement<Tup, Prov>>
{
  type Output = StaticElementsRefTupleIterator<'a, Tup, Prov>;

  fn iter_tuples(&'a self) -> Self::Output {
    StaticElementsRefTupleIterator { elements: self.iter() }
  }
}

pub struct StaticElementsRefTupleIterator<'a, Tup: StaticTupleTrait, Prov: Provenance> {
  elements: std::slice::Iter<'a, &'a StaticElement<Tup, Prov>>,
}

impl<'a, Tup: StaticTupleTrait, Prov: Provenance> Iterator for StaticElementsRefTupleIterator<'a, Tup, Prov> {
  type Item = &'a Tup;

  fn next(&mut self) -> Option<Self::Item> {
    self.elements.next().map(|e| e.tuple.get())
  }
}
