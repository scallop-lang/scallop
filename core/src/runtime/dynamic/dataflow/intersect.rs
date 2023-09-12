use super::*;

pub struct DynamicIntersectDataflow<'a, Prov: Provenance> {
  pub d1: DynamicDataflow<'a, Prov>,
  pub d2: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicIntersectDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicIntersectDataflow<'a, Prov> {
  fn iter_stable(&self) -> DynamicBatches<'a, Prov> {
    let op = IntersectOp { ctx: self.ctx };
    DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_stable(), op)
  }

  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let op = IntersectOp { ctx: self.ctx };
    DynamicBatches::chain(vec![
      DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_recent(), op.clone()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_stable(), op.clone()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_recent(), op.clone()),
    ])
  }
}

pub struct IntersectOp<'a, Prov: Provenance> {
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for IntersectOp<'a, Prov> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, Prov: Provenance> BatchBinaryOp<'a, Prov> for IntersectOp<'a, Prov> {
  fn apply(&self, mut i1: DynamicBatch<'a, Prov>, mut i2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    let i1_curr = i1.next_elem();
    let i2_curr = i2.next_elem();
    DynamicBatch::new(DynamicIntersectBatch {
      i1: i1,
      i1_curr,
      i2: i2,
      i2_curr,
      ctx: self.ctx,
    })
  }
}

pub struct DynamicIntersectBatch<'a, Prov: Provenance> {
  i1: DynamicBatch<'a, Prov>,
  i1_curr: Option<DynamicElement<Prov>>,
  i2: DynamicBatch<'a, Prov>,
  i2_curr: Option<DynamicElement<Prov>>,
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicIntersectBatch<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i1_curr: self.i1_curr.clone(),
      i2: self.i2.clone(),
      i2_curr: self.i2_curr.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicIntersectBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    use std::cmp::Ordering;
    loop {
      match (&self.i1_curr, &self.i2_curr) {
        (Some(i1_curr_elem), Some(i2_curr_elem)) => match i1_curr_elem.tuple.cmp(&i2_curr_elem.tuple) {
          Ordering::Less => {
            self.i1_curr = self.i1.search_until(&i2_curr_elem.tuple);
          }
          Ordering::Equal => {
            let tag = self.ctx.mult(&i1_curr_elem.tag, &i2_curr_elem.tag);
            let result = DynamicElement::new(i1_curr_elem.tuple.clone(), tag);
            self.i1_curr = self.i1.next_elem();
            self.i2_curr = self.i2.next_elem();
            return Some(result);
          }
          Ordering::Greater => {
            self.i2_curr = self.i2.search_until(&i1_curr_elem.tuple);
          }
        },
        _ => return None,
      }
    }
  }
}
