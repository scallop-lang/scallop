use crate::common::expr::Expr;
use crate::common::generic_tuple::GenericTuple;
use crate::common::tuple_access::TupleAccessor;
use crate::common::tuple_type::TupleType;

pub type Permutation = GenericTuple<usize>;

impl std::fmt::Display for Permutation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Value(i) => f.write_fmt(format_args!("{}", i)),
      Self::Tuple(t) => f.write_fmt(format_args!(
        "({})",
        t.iter().map(|e| { format!("{}", e) }).collect::<Vec<_>>().join(",")
      )),
    }
  }
}

impl Permutation {
  pub fn permute(&self, types: &TupleType) -> TupleType {
    if let TupleType::Tuple(ts) = types {
      match self {
        Self::Value(i) => ts[i.clone()].clone(),
        Self::Tuple(t) => TupleType::Tuple(t.iter().map(|e| e.permute(types)).collect()),
      }
    } else {
      panic!("[Internal Error] Types should be flattened")
    }
  }

  pub fn normalize(&self) -> Self {
    match self {
      Self::Tuple(ts) => Self::Tuple(
        ts.iter()
          .filter_map(|t| if t.is_empty() { None } else { Some(t.normalize()) })
          .collect(),
      ),
      Self::Value(v) => Self::Value(v.clone()),
    }
  }

  pub fn expr(&self) -> Expr {
    match self {
      Self::Value(i) => Expr::Access(TupleAccessor::from(i.clone())),
      Self::Tuple(t) => Expr::Tuple(t.iter().map(|e| e.expr()).collect()),
    }
  }

  pub fn order_preserving(&self) -> bool {
    self.order_preserving_helper(&mut 0)
  }

  fn order_preserving_helper(&self, counter: &mut usize) -> bool {
    match self {
      Self::Value(x) => {
        if x == counter {
          *counter += 1;
          true
        } else {
          false
        }
      }
      Self::Tuple(ts) => {
        for t in ts.iter() {
          if !t.order_preserving_helper(counter) {
            return false;
          }
        }
        true
      }
    }
  }
}
