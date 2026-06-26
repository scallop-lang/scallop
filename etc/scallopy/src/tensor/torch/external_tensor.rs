use pyo3::{Bound, Py, PyAny, Python};
use std::any::Any;

use scallop_core::common::foreign_tensor::*;

use super::super::*;

#[derive(Clone)]
pub struct TorchTensor {
  internal: Py<PyAny>,
}

impl TorchTensor {
  pub fn new(p: Py<PyAny>) -> Self {
    Self { internal: p }
  }

  pub fn internal(&self) -> Py<PyAny> {
    self.internal.clone()
  }
}

impl ExternalTensor for TorchTensor {
  fn shape(&self) -> TensorShape {
    Python::attach(|py| {
      let shape_tuple: Vec<i64> = self
        .internal
        .getattr(py, "shape")
        .expect("Cannot get `.shape` from object")
        .extract(py)
        .expect("`.shape` is not a tuple");
      TensorShape::from(shape_tuple)
    })
  }

  fn get_f64(&self) -> f64 {
    Python::attach(|py| {
      self
        .internal
        .call_method0(py, "item")
        .expect("Cannot call function `.item()`")
        .extract(py)
        .expect("Cannot turn `.item()` into f64")
    })
  }

  fn as_any(&self) -> &dyn Any {
    self
  }
}

impl PyExternalTensor for TorchTensor {
  fn from_py_value(p: &Bound<'_, PyAny>) -> Self {
    Self {
      internal: p.clone().unbind(),
    }
  }

  fn to_py_value(&self) -> Py<PyAny> {
    self.internal.clone()
  }
}
