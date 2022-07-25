pub enum PredicateSet {
  None,
  All,
  Some(Vec<String>),
}

impl Default for PredicateSet {
  fn default() -> Self {
    Self::All
  }
}

impl PredicateSet {
  pub fn contains(&self, p: &String) -> bool {
    match self {
      Self::None => false,
      Self::All => true,
      Self::Some(ps) => ps.contains(p),
    }
  }
}
