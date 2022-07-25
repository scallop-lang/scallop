use crate::common::expr::Expr;
use crate::common::tuple::Tuple;
use crate::common::value_type::{FromType, ValueType};
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DynamicAggregateOp {
  Count(DynamicCountOp),
  Sum(DynamicSumOp),
  Prod(DynamicProdOp),
  Min(DynamicMinOp),
  Max(DynamicMaxOp),
  Exists(DynamicExistsOp),
  Unique(DynamicUniqueOp),
}

impl DynamicAggregateOp {
  pub fn count(key: Expr) -> Self {
    Self::Count(DynamicCountOp { key })
  }

  pub fn sum(key: Expr, ty: ValueType) -> Self {
    Self::Sum(DynamicSumOp { key, ty })
  }

  pub fn sum_with_ty<T>(key: Expr) -> Self
  where
    ValueType: FromType<T>,
  {
    Self::Sum(DynamicSumOp { key, ty: <ValueType as FromType<T>>::from_type() })
  }

  pub fn prod(key: Expr, ty: ValueType) -> Self {
    Self::Prod(DynamicProdOp { key, ty })
  }

  pub fn prod_with_ty<T>(key: Expr) -> Self
  where
    ValueType: FromType<T>,
  {
    Self::Prod(DynamicProdOp { key, ty: <ValueType as FromType<T>>::from_type() })
  }

  pub fn min(arg: Option<Expr>, key: Expr) -> Self {
    Self::Min(DynamicMinOp { arg, key })
  }

  pub fn max(arg: Option<Expr>, key: Expr) -> Self {
    Self::Max(DynamicMaxOp { arg, key })
  }

  pub fn exists() -> Self {
    Self::Exists(DynamicExistsOp)
  }

  pub fn unique(key: Expr) -> Self {
    Self::Unique(DynamicUniqueOp { key })
  }

  pub fn aggregate<'a, T: Tag>(
    &self,
    batch: Vec<DynamicElement<T>>,
    ctx: &'a T::Context,
  ) -> Vec<Tuple> {
    match self {
      Self::Count(c) => c.aggregate(batch, ctx),
      Self::Sum(s) => s.aggregate(batch, ctx),
      Self::Prod(p) => p.aggregate(batch, ctx),
      Self::Min(m) => m.aggregate(batch, ctx),
      Self::Max(m) => m.aggregate(batch, ctx),
      Self::Exists(e) => e.aggregate(batch, ctx),
      Self::Unique(u) => u.aggregate(batch, ctx),
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicCountOp {
  pub key: Expr,
}

impl DynamicCountOp {
  pub fn aggregate<T: Tag>(&self, batch: Vec<DynamicElement<T>>, ctx: &T::Context) -> Vec<Tuple> {
    vec![project_batch_helper(batch, &self.key, ctx).len().into()]
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicSumOp {
  pub key: Expr,
  pub ty: ValueType,
}

impl DynamicSumOp {
  pub fn aggregate<T: Tag>(&self, batch: Vec<DynamicElement<T>>, _: &T::Context) -> Vec<Tuple> {
    vec![self.sum(batch)]
  }

  pub fn sum<T: Tag>(&self, batch: Vec<DynamicElement<T>>) -> Tuple {
    if batch.is_empty() {
      self.ty.zero().into()
    } else {
      use crate::common::value_type::ValueType::*;
      match self.ty {
        I8 => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_i8()).into(),
        I16 => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_i16()).into(),
        I32 => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_i32()).into(),
        I64 => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_i64()).into(),
        I128 => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_i128()).into(),
        ISize => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_isize()).into(),
        U8 => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_u8()).into(),
        U16 => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_u16()).into(),
        U32 => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_u32()).into(),
        U64 => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_u64()).into(),
        U128 => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_u128()).into(),
        USize => batch.iter().fold(0, |a, e| a + self.key.eval(&e.tuple).as_usize()).into(),
        F32 => batch.iter().fold(0.0, |a, e| a + self.key.eval(&e.tuple).as_f32()).into(),
        F64 => batch.iter().fold(0.0, |a, e| a + self.key.eval(&e.tuple).as_f64()).into(),
        _ => panic!("Cannot perform summation on type {:?}", &self.ty),
      }
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicProdOp {
  pub key: Expr,
  pub ty: ValueType,
}

impl DynamicProdOp {
  pub fn aggregate<T: Tag>(&self, batch: Vec<DynamicElement<T>>, _: &T::Context) -> Vec<Tuple> {
    vec![self.prod(batch)]
  }

  pub fn prod<T: Tag>(&self, batch: Vec<DynamicElement<T>>) -> Tuple {
    if batch.is_empty() {
      self.ty.zero().into()
    } else {
      use crate::common::value_type::ValueType::*;
      match self.ty {
        I8 => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_i8()).into(),
        I16 => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_i16()).into(),
        I32 => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_i32()).into(),
        I64 => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_i64()).into(),
        I128 => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_i128()).into(),
        ISize => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_isize()).into(),
        U8 => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_u8()).into(),
        U16 => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_u16()).into(),
        U32 => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_u32()).into(),
        U64 => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_u64()).into(),
        U128 => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_u128()).into(),
        USize => batch.iter().fold(1, |a, e| a * self.key.eval(&e.tuple).as_usize()).into(),
        F32 => batch.iter().fold(1.0, |a, e| a * self.key.eval(&e.tuple).as_f32()).into(),
        F64 => batch.iter().fold(1.0, |a, e| a * self.key.eval(&e.tuple).as_f64()).into(),
        _ => panic!("Cannot perform production on type {:?}", self.ty),
      }
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicMinOp {
  pub arg: Option<Expr>,
  pub key: Expr,
}

impl DynamicMinOp {
  pub fn aggregate<T: Tag>(&self, batch: Vec<DynamicElement<T>>, _: &T::Context) -> Vec<Tuple> {
    self.min(batch)
  }

  pub fn min<T: Tag>(
    &self,
    batch: DynamicElements<T>,
  ) -> Vec<Tuple> {
    let mut result = vec![];
    let mut min_value = None;
    for e in batch {
      let v = self.key.eval(&e.tuple);
      let t = match &self.arg {
        Some(arg_expr) => Tuple::Tuple(Box::new([arg_expr.eval(&e.tuple), v.clone()])),
        None => v.clone(),
      };
      if let Some(m) = &min_value {
        if &v == m {
          result.push(t);
        } else if &v < m {
          min_value = Some(v.clone());
          result.clear();
          result.push(t);
        }
      } else {
        min_value = Some(v.clone());
        result.push(t);
      }
    }
    return result;
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicMaxOp {
  pub arg: Option<Expr>,
  pub key: Expr,
}

impl DynamicMaxOp {
  pub fn aggregate<T: Tag>(&self, batch: Vec<DynamicElement<T>>, _: &T::Context) -> Vec<Tuple> {
    self.max(batch)
  }

  pub fn max<T: Tag>(
    &self,
    batch: Vec<DynamicElement<T>>,
  ) -> Vec<Tuple> {
    let mut result = vec![];
    let mut max_value = None;
    for e in batch {
      let v = self.key.eval(&e.tuple);
      let t = match &self.arg {
        Some(arg_expr) => Tuple::Tuple(Box::new([arg_expr.eval(&e.tuple), v.clone()])),
        None => v.clone(),
      };
      if let Some(m) = &max_value {
        if &v == m {
          result.push(t);
        } else if &v > m {
          max_value = Some(v.clone());
          result.clear();
          result.push(t);
        }
      } else {
        max_value = Some(v.clone());
        result.push(t);
      }
    }
    return result;
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicExistsOp;

impl DynamicExistsOp {
  pub fn aggregate<'a, T: Tag>(&self, batch: Vec<DynamicElement<T>>, _: &'a T::Context) -> Vec<Tuple> {
    Self::exists_batch_helper(batch)
  }

  fn exists_batch_helper<T: Tag>(batch: Vec<DynamicElement<T>>) -> Vec<Tuple> {
    vec![(!batch.is_empty()).into()]
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynamicUniqueOp {
  pub key: Expr,
}

impl DynamicUniqueOp {
  pub fn aggregate<'a, T: Tag>(&self, batch: Vec<DynamicElement<T>>, _: &'a T::Context) -> Vec<Tuple> {
    if let Some(e) = batch.get(0) {
      vec![self.key.eval(&e.tuple)]
    } else {
      vec![]
    }
  }
}

pub(crate) fn project_batch_helper<'a, T: Tag>(
  batch: DynamicElements<T>,
  key: &Expr,
  ctx: &'a T::Context,
) -> DynamicElements<T> {
  // Project using key expression
  let mut elems = batch
    .into_iter()
    .map(|elem| {
      let tuple = key.eval(&elem.tuple);
      DynamicElement::new(tuple, elem.tag)
    })
    .collect::<Vec<_>>();
  elems.sort();

  // Merge elements that are the same using `add` (deduplication)
  let mut new_elems: Vec<DynamicElement<T>> = vec![];
  let mut curr_elem: Option<DynamicElement<T>> = None;
  for old_elem in elems {
    if let Some(e) = &mut curr_elem {
      if e.tuple == old_elem.tuple {
        e.tag = ctx.add(&e.tag, &old_elem.tag);
      } else {
        new_elems.push(e.clone());
        curr_elem = Some(old_elem);
      }
    } else {
      curr_elem = Some(old_elem);
    }
  }
  if let Some(e) = curr_elem {
    new_elems.push(e);
  }

  // Return the new elements
  new_elems
}
