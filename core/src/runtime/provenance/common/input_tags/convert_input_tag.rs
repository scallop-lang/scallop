pub trait ConvertFromInputTag<X: Sized>: Sized {
  fn from_input_tag(t: X) -> Option<Self>;
}
