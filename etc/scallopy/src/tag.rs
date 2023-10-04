use pyo3::types::*;

use scallop_core::common::foreign_tensor::*;
use scallop_core::common::input_tag::*;

use super::error::*;
use super::tensor::*;

pub fn from_python_input_tag(ty: &str, tag: &PyAny) -> Result<DynamicInputTag, BindingError> {
  match ty {
    "none" => Ok(DynamicInputTag::None),
    "natural" => Ok(DynamicInputTag::Natural(tag.extract()?)),
    "prob" => Ok(DynamicInputTag::Float(tag.extract()?)),
    "exclusion" => Ok(DynamicInputTag::Exclusive(tag.extract()?)),
    "boolean" => Ok(DynamicInputTag::Bool(tag.extract()?)),
    "exclusive-prob" => {
      let (prob, exc_id): (f64, usize) = tag.extract()?;
      Ok(DynamicInputTag::ExclusiveFloat(prob, exc_id))
    }
    "diff-prob" => Ok(DynamicInputTag::Tensor(DynamicExternalTensor::new(
      Tensor::from_py_value(tag.into()),
    ))),
    _ => Err(BindingError::InvalidInputTag),
  }
}
