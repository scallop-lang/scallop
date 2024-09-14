use super::*;

#[derive(Clone, Debug, PartialEq)]
pub enum NatWithInfty {
  Nat(usize),
  Infty,
}

impl NatWithInfty {
  pub fn as_usize(&self) -> Option<usize> {
    match self {
      Self::Nat(n) => Some(*n),
      Self::Infty => None,
    }
  }

  pub fn is_infty(&self) -> bool {
    match self {
      Self::Infty => true,
      _ => false,
    }
  }
}

impl std::cmp::PartialOrd for NatWithInfty {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    match (self, other) {
      (Self::Infty, Self::Infty) => None,
      (Self::Infty, Self::Nat(_)) => Some(std::cmp::Ordering::Greater),
      (Self::Nat(_), Self::Infty) => Some(std::cmp::Ordering::Less),
      (Self::Nat(n1), Self::Nat(n2)) => Some(n1.cmp(n2)),
    }
  }
}

impl std::fmt::Display for NatWithInfty {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Nat(n) => n.fmt(f),
      Self::Infty => f.write_str("+inf"),
    }
  }
}

impl From<usize> for NatWithInfty {
  fn from(value: usize) -> Self {
    Self::Nat(value)
  }
}

impl Tag for NatWithInfty {}

#[derive(Clone, Debug, Default)]
pub struct TropicalProvenance;

pub type TropicalSemiring = TropicalProvenance;

impl Provenance for TropicalProvenance {
  type Tag = NatWithInfty;

  type InputTag = usize;

  type OutputTag = usize;

  fn name(&self) -> String {
    format!("tropical")
  }

  fn tagging_fn(&self, ext_tag: Self::InputTag) -> Self::Tag {
    ext_tag.into()
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    if let Some(n) = t.as_usize() {
      n
    } else {
      std::usize::MAX
    }
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.is_infty()
  }

  fn zero(&self) -> Self::Tag {
    NatWithInfty::Infty
  }

  fn one(&self) -> Self::Tag {
    0.into()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    match t1.partial_cmp(t2) {
      Some(std::cmp::Ordering::Less) => t1.clone(),
      Some(std::cmp::Ordering::Greater) => t2.clone(),
      Some(std::cmp::Ordering::Equal) => t1.clone(),
      None => t1.clone(),
    }
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    match (t1, t2) {
      (NatWithInfty::Nat(n1), NatWithInfty::Nat(n2)) => NatWithInfty::Nat(n1 + n2),
      _ => NatWithInfty::Infty,
    }
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }
}
