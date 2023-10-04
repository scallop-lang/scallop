use pyo3::prelude::*;
use std::collections::*;

use scallop_core::common::foreign_tensor::*;

use super::*;

#[derive(Clone)]
pub struct TorchTensorRegistry {
  tensors: HashMap<TensorShape, Vec<DynamicExternalTensor>>,
}

impl TorchTensorRegistry {
  pub fn new() -> Self {
    Self {
      tensors: HashMap::new(),
    }
  }

  fn get_torch(&self, symbol: &InternalTensorSymbol) -> TorchTensor {
    let dyn_tensor = self.get(symbol).expect("Tensor symbol not found");
    let torch_tensor = dyn_tensor.cast::<TorchTensor>().expect("Not a torch tensor");
    torch_tensor.clone()
  }

  fn eval_expr_torch(&self, value: &TensorExpr) -> TorchTensor {
    Python::with_gil(|py| match value {
      TensorExpr::Symbol(s) => self.get_torch(s),
      TensorExpr::Float(f) => {
        let builtins = PyModule::import(py, "torch").expect("Cannot import torch");
        let result: Py<PyAny> = builtins
          .getattr("tensor")
          .expect("Cannot get tensor")
          .call1((*f,))
          .expect("Cannot create tensor")
          .extract()
          .expect("Cannot convert");
        TorchTensor::new(result)
      }
      TensorExpr::Add(a, b) => {
        let (ta, tb) = (self.eval_expr_torch(&a), self.eval_expr_torch(&b));
        let result: Py<PyAny> = ta
          .internal()
          .getattr(py, "__add__")
          .expect("Cannot sum")
          .call1(py, (tb.internal(),))
          .expect("Cannot sum")
          .extract(py)
          .expect("Cannot convert");
        TorchTensor::new(result)
      }
      TensorExpr::Sub(a, b) => {
        let (ta, tb) = (self.eval_expr_torch(&a), self.eval_expr_torch(&b));
        let result: Py<PyAny> = ta
          .internal()
          .getattr(py, "__sub__")
          .expect("Cannot sub")
          .call1(py, (tb.internal(),))
          .expect("Cannot sub")
          .extract(py)
          .expect("Cannot convert");
        TorchTensor::new(result)
      }
      TensorExpr::Mul(a, b) => {
        let (ta, tb) = (self.eval_expr_torch(&a), self.eval_expr_torch(&b));
        let result: Py<PyAny> = ta
          .internal()
          .getattr(py, "__mul__")
          .expect("Cannot mul")
          .call1(py, (tb.internal(),))
          .expect("Cannot mul")
          .extract(py)
          .expect("Cannot convert");
        TorchTensor::new(result)
      }
      TensorExpr::Dot(a, b) => {
        let (ta, tb) = (self.eval_expr_torch(&a), self.eval_expr_torch(&b));
        let result: Py<PyAny> = ta
          .internal()
          .getattr(py, "dot")
          .expect("Cannot dot")
          .call1(py, (tb.internal(),))
          .expect("Cannot dot")
          .extract(py)
          .expect("Cannot convert");
        TorchTensor::new(result)
      }
    })
  }
}

impl TensorRegistry for TorchTensorRegistry {
  fn register(&mut self, ext_tensor: DynamicExternalTensor) -> InternalTensorSymbol {
    let shape = ext_tensor.shape();
    let tensors_under_shape = self.tensors.entry(shape.clone()).or_default();
    let id = tensors_under_shape.len();
    tensors_under_shape.push(ext_tensor);
    InternalTensorSymbol::new(shape, id)
  }

  fn get(&self, symbol: &InternalTensorSymbol) -> Option<&DynamicExternalTensor> {
    self.tensors.get(&symbol.shape).and_then(|ts| ts.get(symbol.id))
  }

  fn eval_expr(&self, value: &TensorExpr) -> DynamicExternalTensor {
    DynamicExternalTensor::new(self.eval_expr_torch(value))
  }
}
