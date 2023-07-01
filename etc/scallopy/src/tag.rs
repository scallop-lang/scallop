use pyo3::types::*;
use pyo3::*;

use scallop_core::common::input_tag::*;

use super::error::*;

#[derive(FromPyObject)]
enum PythonInputTag<'a> {
  /// Boolean tag
  Bool(bool),

  /// Exclusion id tag
  Exclusive(usize),

  /// Float tag
  Float(f64),

  /// Tuple of (f64, usize) where `usize` is the exclusion id
  ExclusiveFloat(f64, usize),

  /// Catch all tag
  #[pyo3(transparent)]
  CatchAll(&'a PyAny),
}

pub fn from_python_input_tag(tag: &PyAny) -> Result<DynamicInputTag, BindingError> {
  let py_input_tag: Option<PythonInputTag> = tag.extract()?;
  if let Some(py_input_tag) = py_input_tag {
    match py_input_tag {
      PythonInputTag::Bool(b) => Ok(DynamicInputTag::Bool(b)),
      PythonInputTag::Exclusive(e) => Ok(DynamicInputTag::Exclusive(e)),
      PythonInputTag::Float(f) => Ok(DynamicInputTag::Float(f)),
      PythonInputTag::ExclusiveFloat(f, e) => Ok(DynamicInputTag::ExclusiveFloat(f, e)),
      PythonInputTag::CatchAll(_) => Err(BindingError::InvalidInputTag),
    }
  } else {
    Ok(DynamicInputTag::None)
  }
}
