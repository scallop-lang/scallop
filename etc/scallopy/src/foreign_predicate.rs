use pyo3::types::*;
use pyo3::*;

use scallop_core::common::tuple_type::*;
use scallop_core::common::value_type::*;
use scallop_core::common::value::*;
use scallop_core::common::input_tag::*;
use scallop_core::common::foreign_predicate::*;

use super::tuple::*;
use super::tag::*;

#[derive(Clone)]
pub struct PythonForeignPredicate {
  fp: PyObject,
  name: String,
  types: Vec<ValueType>,
  num_bounded: usize,
}

impl PythonForeignPredicate {
  pub fn new(fp: PyObject) -> Self {
    let name = Python::with_gil(|py| {
      fp
        .getattr(py, "name")
        .expect("Cannot get foreign predicate name")
        .extract(py)
        .expect("Foreign predicate name cannot be extracted into String")
    });

    let types = Python::with_gil(|py| {
      // Call `all_argument_types` function of the Python object
      let func: PyObject = fp
        .getattr(py, "all_argument_types")
        .expect("Cannot get all_argument_types function")
        .extract(py)
        .expect("Cannot extract function into PyObject");

      // Invoke the function
      let py_types: Vec<PyObject> = func.call0(py).expect("Cannot call function").extract(py).expect("Cannot extract into PyList");

      // Convert the Python types into Scallop types
      py_types.into_iter().map(|py_type| py_param_type_to_fp_param_type(py_type, py)).collect()
    });

    let num_bounded: usize = Python::with_gil(|py| {
      let func: PyObject = fp
        .getattr(py, "num_bounded")
        .expect("Cannot get num_bounded function")
        .extract(py)
        .expect("Cannot extract function into PyObject");

      // Invoke the function
      func.call0(py).expect("Cannot call function").extract(py).expect("Cannot extract into usize")
    });

    Self {
      fp,
      name,
      types,
      num_bounded,
    }
  }

  fn output_tuple_type(&self) -> TupleType {
    self.types.iter().skip(self.num_bounded).cloned().collect()
  }
}

impl ForeignPredicate for PythonForeignPredicate {
    fn name(&self) -> String {
      self.name.clone()
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

    fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
      Python::with_gil(|py| {
        // Construct the arguments
        let args: Vec<Py<PyAny>> = bounded.iter().map(to_python_value).collect();
        let args_tuple = PyTuple::new(py, args);

        // Invoke the function
        let maybe_result = self.fp.call1(py, args_tuple).ok();

        // Turn the result back to Scallop values
        if let Some(result) = maybe_result {
          let output_tuple_type = self.output_tuple_type();
          let elements: Vec<(&PyAny, &PyAny)> = result.extract(py).expect("Cannot extract into list of elements");
          let internal: Option<Vec<_>> = elements
            .into_iter()
            .map(|(py_tag, py_tup)| {
              let tag = from_python_input_tag(py_tag).ok()?;
              let tuple = from_python_tuple(py_tup, &output_tuple_type).ok()?;
              Some((tag, tuple.as_values()))
            })
            .collect();
          if let Some(e) = internal {
            e
          } else {
            vec![]
          }
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
    _ => panic!("Unknown type {}", param_type),
  }
}
