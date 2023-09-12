use pyo3::{exceptions::PyException, prelude::*};

use scallop_core::integrate::IntegrateError;
use scallop_core::runtime::error::RuntimeError;

#[derive(Debug)]
pub enum BindingError {
  CompileError(String),
  RuntimeError(RuntimeError),
  InvalidCustomProvenance,
  CustomProvenanceUnsupported,
  UnknownProvenance(String),
  UnknownRelation(String),
  RelationNotComputed(String),
  InvalidLoadCSVArg,
  InvalidBatchSize,
  EmptyBatchInput,
  InvalidInputTag,
  CannotRegisterTensor,
  PyErr(PyErr),
}

impl std::error::Error for BindingError {}

impl std::fmt::Display for BindingError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::CompileError(e) => std::fmt::Display::fmt(e, f),
      Self::RuntimeError(e) => std::fmt::Display::fmt(e, f),
      Self::InvalidCustomProvenance => f.write_str("Invalid custom provenance context"),
      Self::CustomProvenanceUnsupported => f.write_str("Custom provenance is not supported for this operation"),
      Self::UnknownProvenance(p) => f.write_fmt(format_args!("Unknown provenance `{}`", p)),
      Self::UnknownRelation(r) => f.write_fmt(format_args!("Unknown relation `{}`", r)),
      Self::RelationNotComputed(r) => f.write_fmt(format_args!("Relation `{}` has not been computed", r)),
      Self::InvalidLoadCSVArg => f.write_str("Invalid argument for `load_csv`"),
      Self::InvalidBatchSize => f.write_str("Invalid batch size"),
      Self::EmptyBatchInput => f.write_str("Empty batched input"),
      Self::InvalidInputTag => f.write_str("Invalid input tag"),
      Self::CannotRegisterTensor => f.write_str("Cannot register tensor"),
      Self::PyErr(e) => std::fmt::Display::fmt(e, f),
    }
  }
}

impl std::convert::From<IntegrateError> for BindingError {
  fn from(err: IntegrateError) -> Self {
    match err {
      IntegrateError::Compile(cs) => {
        Self::CompileError(cs.into_iter().map(|c| format!("{}", c)).collect::<Vec<_>>().join("\n"))
      }
      IntegrateError::Runtime(e) => Self::RuntimeError(e),
    }
  }
}

impl std::convert::From<PyErr> for BindingError {
  fn from(err: PyErr) -> Self {
    Self::PyErr(err)
  }
}

impl std::convert::From<BindingError> for PyErr {
  fn from(err: BindingError) -> PyErr {
    match err {
      BindingError::PyErr(e) => e,
      err => {
        let err_str = format!("{}", err.to_string());
        let py_err_str: Py<PyAny> = Python::with_gil(|py| err_str.to_object(py));
        PyException::new_err(py_err_str)
      }
    }
  }
}
