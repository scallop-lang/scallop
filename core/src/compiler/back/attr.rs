use std::any::TypeId;
use std::collections::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Attributes {
  attrs: BTreeMap<TypeId, Attribute>,
}

impl From<Vec<Attribute>> for Attributes {
  fn from(attrs: Vec<Attribute>) -> Self {
    Self {
      attrs: attrs.into_iter().map(|attr| (attr.get_type_id(), attr)).collect(),
    }
  }
}

impl<T> From<T> for Attributes where T: AttributeTrait {
  fn from(t: T) -> Self {
    let attr = Attribute::new(t);
    Self {
      attrs: std::iter::once((attr.get_type_id(), attr)).collect(),
    }
  }
}

impl Attributes {
  pub fn new() -> Self {
    Self {
      attrs: BTreeMap::new(),
    }
  }

  pub fn insert<T: AttributeTrait>(&mut self, t: T) {
    self.attrs.insert(TypeId::of::<T>(), Attribute::new(t));
  }

  pub fn get<T: AttributeTrait>(&self) -> Option<&T> {
    self.attrs.get(&TypeId::of::<T>()).and_then(|a| a.get::<T>())
  }

  pub fn iter(&self) -> impl Iterator<Item = &Attribute> {
    self.attrs.iter().map(|(_, a)| a)
  }
}

pub struct Attribute {
  attr: Box<dyn AttributeTrait>,
}

impl<T> From<T> for Attribute where T: AttributeTrait {
  fn from(t: T) -> Self {
    Self::new(t)
  }
}

impl Attribute {
  pub fn new<T: AttributeTrait>(t: T) -> Self {
    Self {
      attr: Box::new(t),
    }
  }

  pub fn name(&self) -> String {
    self.attr.name()
  }

  pub fn args(&self) -> Vec<String> {
    self.attr.args()
  }

  pub fn get_type_id(&self) -> TypeId {
    (*self.attr).type_id()
  }

  pub fn get<T: AttributeTrait>(&self) -> Option<&T> {
    let attr = self.attr.as_ref();
    attr.downcast_ref::<T>().ok()
  }
}

impl std::cmp::PartialEq for Attribute {
  fn eq(&self, other: &Self) -> bool {
    self.get_type_id() == other.get_type_id()
  }
}

impl std::fmt::Debug for Attribute {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Attribute").field("name", &self.name()).field("args", &self.args()).finish()
  }
}

impl std::fmt::Display for Attribute {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let args = self.args();
    if args.is_empty() {
      f.write_fmt(format_args!("@{}", self.name()))
    } else {
      f.write_fmt(format_args!("@{}({})", self.name(), args.join(", ")))
    }
  }
}

impl Clone for Attribute {
  fn clone(&self) -> Self {
    Self {
      attr: dyn_clone::clone_box(&*self.attr),
    }
  }
}

pub trait AttributeTrait: dyn_clone::DynClone + downcast::Any + Send + 'static {
  fn name(&self) -> String;

  fn args(&self) -> Vec<String> {
    vec![]
  }
}

downcast::downcast!(dyn AttributeTrait);
