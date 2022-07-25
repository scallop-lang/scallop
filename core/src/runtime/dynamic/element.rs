use crate::common::tuple::Tuple;
use crate::runtime::provenance::*;

pub type DynamicElement<T> = Tagged<Tuple, T>;

impl<T: Tag> DynamicElement<T> {
  pub fn new(tuple: Tuple, tag: T) -> Self {
    Self { tuple, tag }
  }
}

pub type DynamicElements<T> = Vec<DynamicElement<T>>;
