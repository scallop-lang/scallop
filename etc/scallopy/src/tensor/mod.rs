use pyo3::prelude::*;

mod torch;

#[cfg(feature = "torch-tensor")]
pub use torch::{TorchTensor as Tensor, TorchTensorRegistry as TensorRegistry};

pub trait PyExternalTensor {
  /// Turn into tensor from a python value
  fn from_py_value(p: &PyAny) -> Self;

  /// Turn the tensor into a python value
  fn to_py_value(&self) -> Py<PyAny>;
}
