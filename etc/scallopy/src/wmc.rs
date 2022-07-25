#[derive(Clone, Copy)]
pub enum WMCType {
  SDDTopDown,
  SDDBottomUp,
}

impl<'a> From<&'a str> for WMCType {
  fn from(s: &'a str) -> Self {
    if s == "top-down" {
      Self::SDDTopDown
    } else if s == "bottom-up" {
      Self::SDDBottomUp
    } else {
      panic!("Unknown wmc type `{}`", s)
    }
  }
}
