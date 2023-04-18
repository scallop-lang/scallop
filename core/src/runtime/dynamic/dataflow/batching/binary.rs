use super::super::*;

#[derive(Clone)]
pub enum BatchBinaryOp<'a, Prov: Provenance> {
  Intersect(IntersectOp<'a, Prov>),
  Join(JoinOp<'a, Prov>),
  Product(ProductOp<'a, Prov>),
  Difference(DifferenceOp<'a, Prov>),
  Antijoin(AntijoinOp<'a, Prov>),
  Exclusion(ExclusionOp<'a, Prov>),
}

impl<'a, Prov: Provenance> BatchBinaryOp<'a, Prov> {
  pub fn apply(&self, b1: DynamicBatch<'a, Prov>, b2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    match self {
      Self::Intersect(i) => i.apply(b1, b2),
      Self::Join(j) => j.apply(b1, b2),
      Self::Product(p) => p.apply(b1, b2),
      Self::Difference(d) => d.apply(b1, b2),
      Self::Antijoin(a) => a.apply(b1, b2),
      Self::Exclusion(e) => e.apply(b1, b2),
    }
  }
}
