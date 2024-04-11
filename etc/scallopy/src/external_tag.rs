use pyo3::prelude::*;

use scallop_core::common::foreign_tensor::*;

use super::tensor::*;

#[derive(Clone)]
pub struct ExtTag {
  pub tag: Py<PyAny>,
}

impl FromTensor for ExtTag {
  fn from_tensor(tensor: DynamicExternalTensor) -> Option<Self> {
    tensor.cast::<Tensor>().map(|t| t.to_py_value().into())
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

  fn into_none_prepended_vec(self) -> Vec<Py<PyAny>>;
}

impl ExtTagVec for Vec<ExtTag> {
  fn into_vec(self) -> Vec<Py<PyAny>> {
    self.into_iter().map(|v| v.tag).collect()
  }

  fn into_none_prepended_vec(self) -> Vec<Py<PyAny>> {
    let none: Option<Py<PyAny>> = None;
    std::iter::once(Python::with_gil(|py| none.to_object(py))).chain(self.into_iter().map(|v| v.tag)).collect()
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
