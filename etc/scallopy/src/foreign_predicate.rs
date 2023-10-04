use pyo3::types::*;
use pyo3::*;

use scallop_core::common::foreign_predicate::*;
use scallop_core::common::input_tag::*;
use scallop_core::common::tuple_type::*;
use scallop_core::common::value::*;
use scallop_core::common::value_type::*;
use scallop_core::runtime::env::*;

use super::tag::*;
use super::tuple::*;

#[derive(Clone)]
pub struct PythonForeignPredicate {
  fp: PyObject,
  name: String,
  type_params: Vec<ValueType>,
  types: Vec<ValueType>,
  tag_type: String,
  num_bounded: usize,
  suppress_warning: bool,
}

impl PythonForeignPredicate {
  pub fn new(fp: PyObject) -> Self {
    Python::with_gil(|py| {
      let name = fp
        .getattr(py, "name")
        .expect("Cannot get foreign predicate name")
        .extract(py)
        .expect("Foreign predicate name cannot be extracted into String");

      let suppress_warning = fp
        .getattr(py, "suppress_warning")
        .expect("Cannot get foreign predicate `suppress_warning`")
        .extract(py)
        .expect("Foreign predicate `suppress_warning` cannot be extracted into bool");

      let type_params = {
        let type_param_pyobjs: Vec<PyObject> = fp
          .getattr(py, "type_params")
          .expect("Cannot get all_argument_types function")
          .extract(py)
          .expect("Cannot extract function into PyObject");

        // Convert the Python types into Scallop types
        type_param_pyobjs
          .into_iter()
          .map(|py_type| py_param_type_to_fp_param_type(py_type, py))
          .collect()
      };

      let types = {
        // Call `all_argument_types` function of the Python object
        let func: PyObject = fp
          .getattr(py, "all_argument_types")
          .expect("Cannot get all_argument_types function")
          .extract(py)
          .expect("Cannot extract function into PyObject");

        // Invoke the function
        let py_types: Vec<PyObject> = func
          .call0(py)
          .expect("Cannot call function")
          .extract(py)
          .expect("Cannot extract into PyList");

        // Convert the Python types into Scallop types
        py_types
          .into_iter()
          .map(|py_type| py_param_type_to_fp_param_type(py_type, py))
          .collect()
      };

      let tag_type: String = fp
        .getattr(py, "tag_type")
        .expect("Cannot get tag_type")
        .extract(py)
        .expect("tag_type is not a string");

      let num_bounded: usize = {
        let func: PyObject = fp
          .getattr(py, "num_bounded")
          .expect("Cannot get num_bounded function")
          .extract(py)
          .expect("Cannot extract function into PyObject");

        // Invoke the function
        func
          .call0(py)
          .expect("Cannot call function")
          .extract(py)
          .expect("Cannot extract into usize")
      };

      Self {
        fp,
        name,
        type_params,
        types,
        tag_type,
        num_bounded,
        suppress_warning,
      }
    })
  }

  fn output_tuple_type(&self) -> TupleType {
    self.types.iter().skip(self.num_bounded).cloned().collect()
  }
}

impl ForeignPredicate for PythonForeignPredicate {
  fn name(&self) -> String {
    self.name.clone()
  }

  fn generic_type_parameters(&self) -> Vec<ValueType> {
    self.type_params.clone()
  }

  fn arity(&self) -> usize {
    self.types.len()
  }

  fn argument_type(&self, i: usize) -> ValueType {
    self.types[i].clone()
  }

  fn num_bounded(&self) -> usize {
    self.num_bounded
  }

  fn evaluate_with_env(&self, env: &RuntimeEnvironment, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    Python::with_gil(|py| {
      // Construct the arguments
      let args: Vec<Py<PyAny>> = bounded.iter().filter_map(|v| to_python_value(v, &env.into())).collect();
      let args_tuple = PyTuple::new(py, args);

      // Invoke the function
      let maybe_result = match self.fp.call1(py, args_tuple) {
        Ok(result) => Some(result),
        Err(err) => {
          if !self.suppress_warning {
            eprintln!("[Foreign Predicate Error] {}", err);
            err.print(py);
          }
          None
        }
      };

      // Turn the result back to Scallop values
      if let Some(result) = maybe_result {
        let output_tuple_type = self.output_tuple_type();
        let elements: Vec<(&PyAny, &PyAny)> = result.extract(py).expect("Cannot extract into list of elements");
        let internal: Vec<_> = elements
          .into_iter()
          .filter_map(|(py_tag, py_tup)| {
            let tag = match from_python_input_tag(&self.tag_type, py_tag) {
              Ok(tag) => tag,
              Err(err) => {
                if !self.suppress_warning {
                  eprintln!("Error when parsing tag: {}", err);
                }
                return None;
              }
            };
            let tuple = match from_python_tuple(py_tup, &output_tuple_type, &env.into()) {
              Ok(tuple) => tuple,
              Err(err) => {
                if !self.suppress_warning {
                  eprintln!("Error when parsing tuple: {}", err);
                }
                return None;
              }
            };
            Some((tag, tuple.as_values()))
          })
          .collect();
        internal
      } else {
        vec![]
      }
    })
  }
}

fn py_param_type_to_fp_param_type(obj: PyObject, py: Python<'_>) -> ValueType {
  let param_type: String = obj
    .getattr(py, "type")
    .expect("Cannot get param type")
    .extract(py)
    .expect("Cannot extract into String");
  match param_type.as_str() {
    "i8" => ValueType::I8,
    "i16" => ValueType::I16,
    "i32" => ValueType::I32,
    "i64" => ValueType::I64,
    "i128" => ValueType::I128,
    "isize" => ValueType::ISize,
    "u8" => ValueType::U8,
    "u16" => ValueType::U16,
    "u32" => ValueType::U32,
    "u64" => ValueType::U64,
    "u128" => ValueType::U128,
    "usize" => ValueType::USize,
    "f32" => ValueType::F32,
    "f64" => ValueType::F64,
    "bool" => ValueType::Bool,
    "char" => ValueType::Char,
    "String" => ValueType::String,
    "DateTime" => ValueType::DateTime,
    "Duration" => ValueType::Duration,
    "Entity" => ValueType::Entity,
    "Tensor" => ValueType::Tensor,
    _ => panic!("Unknown type {}", param_type),
  }
}
