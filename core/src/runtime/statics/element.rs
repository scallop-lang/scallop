use crate::common::element::*;
use crate::runtime::provenance::*;

use super::*;

pub type StaticElement<Tup, Tag> = Tagged<StaticTupleWrapper<Tup>, Tag>;

impl<A, B> StaticElement<A, B>
where
  A: StaticTupleTrait,
  B: Tag,
{
  pub fn new(tuple: A, tag: B) -> Self {
    Self {
      tuple: StaticTupleWrapper::new(tuple),
      tag,
    }
  }

  pub fn tuple(self) -> A {
    self.tuple.into()
  }
}

impl<Tup: StaticTupleTrait, T: Tag> Element<T> for StaticElement<Tup, T> {
  fn tag(&self) -> &T {
    &self.tag
  }
}

impl<A: StaticTupleTrait, B: Tag> Into<(A, B)> for StaticElement<A, B> {
  fn into(self) -> (A, B) {
    let Self { tuple, tag } = self;
    (tuple.into(), tag)
  }
}

pub type StaticElements<Tup, Tag> = Vec<StaticElement<Tup, Tag>>;

pub trait StaticTupleIterator<'a, Tup: 'static + StaticTupleTrait> {
  type Output: Iterator<Item = &'a Tup>;

  fn iter_tuples(&'a self) -> Self::Output;
}

impl<'a, Tup: 'static + StaticTupleTrait, T: Tag> StaticTupleIterator<'a, Tup> for StaticElements<Tup, T> {
  type Output = StaticElementsTupleIterator<'a, Tup, T>;

  fn iter_tuples(&'a self) -> Self::Output {
    StaticElementsTupleIterator { elements: self.iter() }
  }
}

pub struct StaticElementsTupleIterator<'a, Tup: StaticTupleTrait, T: Tag> {
  elements: std::slice::Iter<'a, StaticElement<Tup, T>>,
}

impl<'a, Tup: StaticTupleTrait, T: Tag> Iterator for StaticElementsTupleIterator<'a, Tup, T> {
  type Item = &'a Tup;

  fn next(&mut self) -> Option<Self::Item> {
    self.elements.next().map(|e| e.tuple.get())
  }
}

impl<'a, Tup: 'static + StaticTupleTrait, T: Tag> StaticTupleIterator<'a, Tup> for Vec<&'a StaticElement<Tup, T>> {
  type Output = StaticElementsRefTupleIterator<'a, Tup, T>;

  fn iter_tuples(&'a self) -> Self::Output {
    StaticElementsRefTupleIterator { elements: self.iter() }
  }
}

pub struct StaticElementsRefTupleIterator<'a, Tup: StaticTupleTrait, T: Tag> {
  elements: std::slice::Iter<'a, &'a StaticElement<Tup, T>>,
}

impl<'a, Tup: StaticTupleTrait, T: Tag> Iterator for StaticElementsRefTupleIterator<'a, Tup, T> {
  type Item = &'a Tup;

  fn next(&mut self) -> Option<Self::Item> {
    self.elements.next().map(|e| e.tuple.get())
  }
}
