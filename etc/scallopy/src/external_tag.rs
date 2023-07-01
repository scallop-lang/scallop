use pyo3::prelude::*;

use scallop_core::common::tensors::*;

#[derive(Clone)]
pub struct ExtTag {
  pub tag: Py<PyAny>,
}

impl FromTensor for ExtTag {
  #[allow(unused)]
  #[cfg(not(feature = "torch-tensor"))]
  fn from_tensor(tensor: Tensor) -> Option<Self> {
    None
  }

  #[cfg(feature = "torch-tensor")]
  fn from_tensor(tensor: Tensor) -> Option<Self> {
    use super::torch::*;
    Python::with_gil(|py| {
      let py_tensor = PyTensor(tensor.tensor);
      let py_obj: Py<PyAny> = py_tensor.into_py(py);
      let ext_tag: ExtTag = py_obj.into();
      Some(ext_tag)
    })
  }
}

impl From<Py<PyAny>> for ExtTag {
  fn from(tag: Py<PyAny>) -> Self {
    Self { tag }
  }
}

impl From<&PyAny> for ExtTag {
  fn from(tag: &PyAny) -> Self {
    Self { tag: tag.into() }
  }
}

impl Into<Py<PyAny>> for ExtTag {
  fn into(self) -> Py<PyAny> {
    self.tag
  }
}

pub trait ExtTagVec {
  fn into_vec(self) -> Vec<Py<PyAny>>;
}

impl ExtTagVec for Vec<ExtTag> {
  fn into_vec(self) -> Vec<Py<PyAny>> {
    self.into_iter().map(|v| v.tag).collect()
  }
}

pub trait ExtTagOption {
  fn into_option(self) -> Option<Py<PyAny>>;
}

impl ExtTagOption for Option<ExtTag> {
  fn into_option(self) -> Option<Py<PyAny>> {
    self.map(|v| v.tag)
  }
}

impl ExtTagOption for Option<&ExtTag> {
  fn into_option(self) -> Option<Py<PyAny>> {
    self.map(|v| v.tag.clone())
  }
}
