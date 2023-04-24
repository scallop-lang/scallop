use crate::common::input_tag::*;

pub trait StaticInputTag: Sized {
  fn from_dynamic_input_tag(_: &DynamicInputTag) -> Option<Self>;
}

impl<T> StaticInputTag for T {
  default fn from_dynamic_input_tag(_: &DynamicInputTag) -> Option<Self> {
    None
  }
}
