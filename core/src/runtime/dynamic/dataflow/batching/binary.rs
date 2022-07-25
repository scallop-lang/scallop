use super::super::*;

#[derive(Clone)]
pub enum BatchBinaryOp<'a, T: Tag> {
  Intersect(IntersectOp<'a, T>),
  Join(JoinOp<'a, T>),
  Product(ProductOp<'a, T>),
  Difference(DifferenceOp<'a, T>),
  Antijoin(AntijoinOp<'a, T>),
}

impl<'a, T: Tag> BatchBinaryOp<'a, T> {
  pub fn apply(&self, b1: DynamicBatch<'a, T>, b2: DynamicBatch<'a, T>) -> DynamicBatch<'a, T> {
    match self {
      Self::Intersect(i) => i.apply(b1, b2),
      Self::Join(j) => j.apply(b1, b2),
      Self::Product(p) => p.apply(b1, b2),
      Self::Difference(d) => d.apply(b1, b2),
      Self::Antijoin(a) => a.apply(b1, b2),
    }
  }
}
