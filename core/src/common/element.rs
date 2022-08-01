use crate::runtime::provenance::*;

pub trait Element<T: Tag> {
  fn tag(&self) -> &T;
}
