use std::collections::*;

use super::*;

pub struct TensorRegistry {
  tensors: HashMap<TensorShape, Vec<Tensor>>,
}

impl TensorRegistry {
  pub fn new() -> Self {
    Self {
      tensors: HashMap::new(),
    }
  }

  pub fn register(&mut self, tensor: Tensor) -> TensorSymbol {
    let shape = tensor.shape();
    let tensors_under_shape = self.tensors.entry(shape.clone()).or_default();
    let id = tensors_under_shape.len();
    tensors_under_shape.push(tensor);
    TensorSymbol::new(shape, id)
  }

  pub fn get(&self, symbol: &TensorSymbol) -> Option<&Tensor> {
    self.tensors.get(&symbol.shape).and_then(|ts| ts.get(symbol.id))
  }

  #[cfg(feature = "torch-tensor")]
  pub fn eval_expr(&self, value: &TensorExpr) -> Tensor {
    match value {
      TensorExpr::Symbol(s) => self.get(s).expect("Cannot find symbol").clone(),
      TensorExpr::Float(f) => Tensor::new((*f).into()),
      TensorExpr::Add(v1, v2) => Tensor::new(self.eval_expr(v1).tensor + self.eval_expr(v2).tensor),
      TensorExpr::Sub(v1, v2) => Tensor::new(self.eval_expr(v1).tensor - self.eval_expr(v2).tensor),
      TensorExpr::Mul(v1, v2) => Tensor::new(self.eval_expr(v1).tensor * self.eval_expr(v2).tensor),
      TensorExpr::Dot(v1, v2) => Tensor::new(self.eval_expr(v1).tensor.dot(&self.eval_expr(v2).tensor)),
    }
  }

  #[cfg(not(feature = "torch-tensor"))]
  #[allow(unused)]
  pub fn eval_expr(&self, value: &TensorExpr) -> Tensor {
    panic!("{}", NO_TORCH_MSG)
  }

  pub fn eval(&self, value: &TensorValue) -> Tensor {
    self.eval_expr(&value.expr)
  }
}
