use pyo3::types::*;
use pyo3::*;

use scallop_core::common::foreign_function::*;
use scallop_core::common::type_family::*;
use scallop_core::common::value::*;
use scallop_core::common::value_type::*;
use scallop_core::runtime::env::*;

use super::tuple::*;

/// A Python foreign function
///
/// TODO: Memoization
#[derive(Clone)]
pub struct PythonForeignFunction {
  ff: PyObject,
}

impl PythonForeignFunction {
  /// Create a new PythonForeignFunction
  pub fn new(ff: PyObject) -> Self {
    Self { ff }
  }
}

impl ForeignFunction for PythonForeignFunction {
  fn name(&self) -> String {
    Python::with_gil(|py| {
      self
        .ff
        .getattr(py, "name")
        .expect("Cannot get foreign function name")
        .extract(py)
        .expect("Foreign function name cannot be extracted into String")
    })
  }

  fn num_generic_types(&self) -> usize {
    Python::with_gil(|py| {
      let generic_type_params: PyObject = self
        .ff
        .getattr(py, "generic_type_params")
        .expect("Cannot get foreign function generic type parameters");
      let generic_type_params: &PyList = generic_type_params
        .downcast::<PyList>(py)
        .expect("Cannot cast into PyList");
      generic_type_params.len()
    })
  }

  fn generic_type_family(&self, i: usize) -> TypeFamily {
    Python::with_gil(|py| {
      let generic_type_params: PyObject = self
        .ff
        .getattr(py, "generic_type_params")
        .expect("Cannot get foreign function generic type parameters");
      let generic_type_params: &PyList = generic_type_params
        .downcast::<PyList>(py)
        .expect("Cannot cast into PyList");
      let param: String = generic_type_params
        .get_item(i)
        .expect("Cannot get i-th param")
        .extract()
        .expect("Cannot extract param into String");
      match param.as_str() {
        "Any" => TypeFamily::Any,
        "Number" => TypeFamily::Number,
        "Integer" => TypeFamily::Integer,
        "SignedInteger" => TypeFamily::SignedInteger,
        "UnsignedInteger" => TypeFamily::UnsignedInteger,
        "Float" => TypeFamily::Float,
        _ => panic!("Unknown type family {}", param),
      }
    })
  }

  fn num_static_arguments(&self) -> usize {
    Python::with_gil(|py| {
      let static_arg_types: PyObject = self
        .ff
        .getattr(py, "static_arg_types")
        .expect("Cannot get foreign function static arg types");
      let static_arg_types: &PyList = static_arg_types
        .downcast::<PyList>(py)
        .expect("Cannot cast into PyList");
      static_arg_types.len()
    })
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    Python::with_gil(|py| {
      let static_arg_types: PyObject = self
        .ff
        .getattr(py, "static_arg_types")
        .expect("Cannot get foreign function static arg types");
      let static_arg_types: &PyList = static_arg_types
        .downcast::<PyList>(py)
        .expect("Cannot cast into PyList");
      let param_type: PyObject = static_arg_types
        .get_item(i)
        .expect("Cannot get i-th param")
        .extract()
        .expect("Cannot extract param into Object");
      py_param_type_to_ff_param_type(param_type, py)
    })
  }

  fn num_optional_arguments(&self) -> usize {
    Python::with_gil(|py| {
      let optional_arg_types: PyObject = self
        .ff
        .getattr(py, "optional_arg_types")
        .expect("Cannot get foreign function optional arg types");
      let optional_arg_types: &PyList = optional_arg_types
        .downcast::<PyList>(py)
        .expect("Cannot cast into PyList");
      optional_arg_types.len()
    })
  }

  fn optional_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    Python::with_gil(|py| {
      let optional_arg_types: PyObject = self
        .ff
        .getattr(py, "optional_arg_types")
        .expect("Cannot get foreign function optional arg types");
      let optional_arg_types: &PyList = optional_arg_types
        .downcast::<PyList>(py)
        .expect("Cannot cast into PyList");
      let param_type: PyObject = optional_arg_types
        .get_item(i)
        .expect("Cannot get i-th param")
        .extract()
        .expect("Cannot extract param into Object");
      py_param_type_to_ff_param_type(param_type, py)
    })
  }

  fn has_variable_arguments(&self) -> bool {
    Python::with_gil(|py| {
      !self
        .ff
        .getattr(py, "var_arg_types")
        .expect("Cannot get foreign function variable arg types")
        .is_none(py)
    })
  }

  fn variable_argument_type(&self) -> ForeignFunctionParameterType {
    Python::with_gil(|py| {
      let var_arg_type: PyObject = self
        .ff
        .getattr(py, "var_arg_types")
        .expect("Cannot get foreign function variable arg types");
      py_param_type_to_ff_param_type(var_arg_type, py)
    })
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    Python::with_gil(|py| {
      let var_arg_type: PyObject = self
        .ff
        .getattr(py, "return_type")
        .expect("Cannot get foreign function return type");
      py_param_type_to_ff_param_type(var_arg_type, py)
    })
  }

  fn execute_with_env(&self, env: &RuntimeEnvironment, args: Vec<Value>) -> Option<Value> {
    let ty = self.infer_return_type(&args);

    // Actually run the function
    Python::with_gil(|py| {
      // First obtain the function
      let func: PyObject = self
        .ff
        .getattr(py, "func")
        .expect("Cannot get function")
        .extract(py)
        .expect("Cannot extract function");

      // Construct the arguments
      let args: Vec<Py<PyAny>> = args.iter().map(|a| to_python_value(a, &env.into())).collect();
      let args_tuple = PyTuple::new(py, args);

      // Invoke the function
      let maybe_result = match func.call1(py, args_tuple) {
        Ok(result) => Some(result),
        Err(err) => {
          eprintln!("{}", err);
          None
        }
      };

      // Turn the result back to Scallop value
      if let Some(result) = maybe_result {
        let result: &PyAny = result.extract(py).expect("");
        from_python_value(result, &ty, &env.into()).ok()
      } else {
        None
      }
    })
  }
}

fn py_param_type_to_ff_param_type(obj: PyObject, py: Python<'_>) -> ForeignFunctionParameterType {
  let param_kind: String = obj
    .getattr(py, "kind")
    .expect("Cannot get param kind")
    .extract(py)
    .expect("Cannot extract into String");
  match param_kind.as_str() {
    "generic" => {
      let generic_id: usize = obj
        .getattr(py, "id")
        .expect("Cannot get param id")
        .extract(py)
        .expect("Cannot extract into usize");
      ForeignFunctionParameterType::Generic(generic_id)
    }
    "family" => {
      let type_family_str: String = obj
        .getattr(py, "type_family")
        .expect("Cannot get param id")
        .extract(py)
        .expect("Cannot extract into usize");
      match type_family_str.as_str() {
        "Any" => ForeignFunctionParameterType::TypeFamily(TypeFamily::Any),
        "Number" => ForeignFunctionParameterType::TypeFamily(TypeFamily::Number),
        "Integer" => ForeignFunctionParameterType::TypeFamily(TypeFamily::Integer),
        "SignedInteger" => ForeignFunctionParameterType::TypeFamily(TypeFamily::SignedInteger),
        "UnsignedInteger" => ForeignFunctionParameterType::TypeFamily(TypeFamily::UnsignedInteger),
        "Float" => ForeignFunctionParameterType::TypeFamily(TypeFamily::Float),
        _ => panic!("Unknown type family {}", type_family_str),
      }
    }
    "base" => {
      let type_str: String = obj
        .getattr(py, "type")
        .expect("Cannot get param id")
        .extract(py)
        .expect("Cannot extract into usize");
      match type_str.as_str() {
        "i8" => ForeignFunctionParameterType::BaseType(ValueType::I8),
        "i16" => ForeignFunctionParameterType::BaseType(ValueType::I16),
        "i32" => ForeignFunctionParameterType::BaseType(ValueType::I32),
        "i64" => ForeignFunctionParameterType::BaseType(ValueType::I64),
        "i128" => ForeignFunctionParameterType::BaseType(ValueType::I128),
        "isize" => ForeignFunctionParameterType::BaseType(ValueType::ISize),
        "u8" => ForeignFunctionParameterType::BaseType(ValueType::U8),
        "u16" => ForeignFunctionParameterType::BaseType(ValueType::U16),
        "u32" => ForeignFunctionParameterType::BaseType(ValueType::U32),
        "u64" => ForeignFunctionParameterType::BaseType(ValueType::U64),
        "u128" => ForeignFunctionParameterType::BaseType(ValueType::U128),
        "usize" => ForeignFunctionParameterType::BaseType(ValueType::USize),
        "f32" => ForeignFunctionParameterType::BaseType(ValueType::F32),
        "f64" => ForeignFunctionParameterType::BaseType(ValueType::F64),
        "bool" => ForeignFunctionParameterType::BaseType(ValueType::Bool),
        "char" => ForeignFunctionParameterType::BaseType(ValueType::Char),
        "String" => ForeignFunctionParameterType::BaseType(ValueType::String),
        _ => panic!("Unknown base type {}", type_str),
      }
    }
    _ => panic!("Unknown parameter kind"),
  }
}
