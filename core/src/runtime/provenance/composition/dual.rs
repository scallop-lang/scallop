use super::*;

#[derive(Clone)]
pub struct DualProvenance<P1: Provenance, P2: Provenance> {
  pub p1: P1,
  pub p2: P2,
}

impl<P1: Provenance, P2: Provenance> Provenance for DualProvenance<P1, P2> {
  type Tag = DualTag<P1::Tag, P2::Tag>;

  type InputTag = DualTag<P1::InputTag, P2::InputTag>;

  type OutputTag = DualTag<P1::OutputTag, P2::OutputTag>;

  fn name(&self) -> String {
    format!("Dual({}, {})", self.p1.name(), self.p2.name())
  }

  fn tagging_fn(&self, ext_tag: Self::InputTag) -> Self::Tag {
    Self::Tag {
      t1: self.p1.tagging_fn(ext_tag.t1),
      t2: self.p2.tagging_fn(ext_tag.t2),
    }
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    Self::OutputTag {
      t1: self.p1.recover_fn(&t.t1),
      t2: self.p2.recover_fn(&t.t2),
    }
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    self.p1.discard(&t.t1) || self.p2.discard(&t.t2)
  }

  fn zero(&self) -> Self::Tag {
    Self::Tag {
      t1: self.p1.zero(),
      t2: self.p2.zero(),
    }
  }

  fn one(&self) -> Self::Tag {
    Self::Tag {
      t1: self.p1.one(),
      t2: self.p2.one(),
    }
  }

  fn add(&self, x: &Self::Tag, y: &Self::Tag) -> Self::Tag {
    Self::Tag {
      t1: self.p1.add(&x.t1, &y.t1),
      t2: self.p2.add(&x.t2, &y.t2),
    }
  }

  fn mult(&self, x: &Self::Tag, y: &Self::Tag) -> Self::Tag {
    Self::Tag {
      t1: self.p1.mult(&x.t1, &y.t1),
      t2: self.p2.mult(&x.t2, &y.t2),
    }
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    self.p1.saturated(&t_old.t1, &t_new.t1) && self.p2.saturated(&t_old.t2, &t_new.t2)
  }
}

#[derive(Clone)]
pub struct DualTag<T1, T2> {
  pub t1: T1,
  pub t2: T2,
}

impl<T1: Tag, T2: Tag> Tag for DualTag<T1, T2> {}

impl<T1: std::fmt::Debug, T2: std::fmt::Debug> std::fmt::Debug for DualTag<T1, T2> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_tuple("").field(&self.t1).field(&self.t2).finish()
  }
}

impl<T1: std::fmt::Display, T2: std::fmt::Display> std::fmt::Display for DualTag<T1, T2> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("({}, {})", self.t1, self.t2))
  }
}
