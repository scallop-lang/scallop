use super::utils::*;
use super::*;

/// Note: d2 should always be a DynamicCollection
pub struct DynamicAntijoinDataflow<'a, Prov: Provenance> {
  pub d1: DynamicDataflow<'a, Prov>,
  pub d2: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicAntijoinDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      d1: self.d1.clone(),
      d2: self.d2.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicAntijoinDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    let op = AntijoinOp { ctx: self.ctx };
    DynamicBatches::chain(vec![
      DynamicBatches::binary(self.d1.iter_stable(), self.d2.iter_recent(), op.clone()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_stable(), op.clone()),
      DynamicBatches::binary(self.d1.iter_recent(), self.d2.iter_recent(), op.clone()),
    ])
  }
}

pub struct AntijoinOp<'a, Prov: Provenance> {
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for AntijoinOp<'a, Prov> {
  fn clone(&self) -> Self {
    Self { ctx: self.ctx }
  }
}

impl<'a, Prov: Provenance> BatchBinaryOp<'a, Prov> for AntijoinOp<'a, Prov> {
  fn apply(&self, mut i1: DynamicBatch<'a, Prov>, mut i2: DynamicBatch<'a, Prov>) -> DynamicBatch<'a, Prov> {
    let i1_curr = i1.next_elem();
    let i2_curr = i2.next_elem();
    DynamicBatch::new(DynamicAntijoinBatch {
      i1: i1,
      i1_curr,
      i2: i2,
      i2_curr,
      curr_iter: None,
      ctx: self.ctx,
    })
  }
}

pub struct DynamicAntijoinBatch<'a, Prov: Provenance> {
  i1: DynamicBatch<'a, Prov>,
  i1_curr: Option<DynamicElement<Prov>>,
  i2: DynamicBatch<'a, Prov>,
  i2_curr: Option<DynamicElement<Prov>>,
  curr_iter: Option<JoinProductIterator<Prov>>,
  ctx: &'a Prov,
}

impl<'a, Prov: Provenance> Clone for DynamicAntijoinBatch<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      i1: self.i1.clone(),
      i1_curr: self.i1_curr.clone(),
      i2: self.i2.clone(),
      i2_curr: self.i2_curr.clone(),
      curr_iter: self.curr_iter.clone(),
      ctx: self.ctx,
    }
  }
}

impl<'a, Prov: Provenance> Batch<'a, Prov> for DynamicAntijoinBatch<'a, Prov> {
  fn next_elem(&mut self) -> Option<DynamicElement<Prov>> {
    use std::cmp::Ordering;
    loop {
      if let Some(curr_prod_iter) = &mut self.curr_iter {
        if let Some((e1, e2)) = curr_prod_iter.next() {
          let maybe_tag = self.ctx.minus(&e1.tag, &e2.tag);
          if let Some(tag) = maybe_tag {
            let result = DynamicElement::new(e1.tuple.clone(), tag);
            return Some(result);
          } else {
            continue;
          }
        } else {
          self.i1.step(curr_prod_iter.v1.len() - 1);
          self.i1_curr = self.i1.next_elem();
          self.i2.step(curr_prod_iter.v2.len() - 1);
          self.i2_curr = self.i2.next_elem();
          self.curr_iter = None;
        }
      }

      match (&self.i1_curr, &self.i2_curr) {
        (Some(i1_curr_elem), Some(i2_curr_elem)) => match i1_curr_elem.tuple[0].cmp(&i2_curr_elem.tuple) {
          Ordering::Less => {
            let result = i1_curr_elem.clone();
            self.i1_curr = self.i1.next_elem();
            return Some(result);
          }
          Ordering::Equal => {
            let key = &i1_curr_elem.tuple[0];
            let v1 = std::iter::once(i1_curr_elem.clone())
              .chain(self.i1.clone().take_while(|x| &x.tuple[0] == key))
              .collect::<Vec<_>>();
            let v2 = std::iter::once(i2_curr_elem.clone()).collect::<Vec<_>>();
            let iter = JoinProductIterator::new(v1, v2);
            self.curr_iter = Some(iter);
          }
          Ordering::Greater => self.i2_curr = self.i2.search_until(&i1_curr_elem.tuple[0]),
        },
        (Some(i1_curr_elem), None) => {
          let result = i1_curr_elem.clone();
          self.i1_curr = self.i1.next_elem();
          return Some(result);
        }
        _ => break None,
      }
    }
  }
}
