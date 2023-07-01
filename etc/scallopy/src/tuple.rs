use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::types::*;

use scallop_core::common::tensors::*;
use scallop_core::common::tuple::Tuple;
use scallop_core::common::tuple_type::TupleType;
use scallop_core::common::value::Value;
use scallop_core::common::value_type::ValueType;
use scallop_core::utils;

use super::runtime::*;

pub fn to_python_tuple(tup: &Tuple, env: &PythonRuntimeEnvironment) -> Py<PyAny> {
  match tup {
    Tuple::Tuple(t) => Python::with_gil(|py| {
      let values = t.iter().map(|t| to_python_tuple(t, env)).collect::<Vec<_>>();
      PyTuple::new(py, values).into()
    }),
    Tuple::Value(v) => to_python_value(v, env),
  }
}

pub fn to_python_value(val: &Value, env: &PythonRuntimeEnvironment) -> Py<PyAny> {
  use Value::*;
  Python::with_gil(|py| match val {
    I8(i) => i.to_object(py),
    I16(i) => i.to_object(py),
    I32(i) => i.to_object(py),
    I64(i) => i.to_object(py),
    I128(i) => i.to_object(py),
    ISize(i) => i.to_object(py),
    U8(i) => i.to_object(py),
    U16(i) => i.to_object(py),
    U32(i) => i.to_object(py),
    U64(i) => i.to_object(py),
    U128(i) => i.to_object(py),
    USize(i) => i.to_object(py),
    F32(f) => f.to_object(py),
    F64(f) => f.to_object(py),
    Char(c) => c.to_object(py),
    Bool(b) => b.to_object(py),
    Str(s) => s.to_object(py),
    String(s) => s.to_object(py),
    Symbol(s) => env.symbol_registry.get_symbol(*s).to_object(py),
    SymbolString(s) => s.to_object(py),
    DateTime(d) => d.to_string().to_object(py),
    Duration(d) => d.to_string().to_object(py),
    Entity(e) => e.to_object(py),
    Tensor(t) => tensor_to_py_object(t.clone(), py),
    TensorValue(v) => tensor_to_py_object(env.tensor_registry.eval(v), py),
  })
}

pub fn from_python_tuple(v: &PyAny, ty: &TupleType, env: &PythonRuntimeEnvironment) -> PyResult<Tuple> {
  match ty {
    TupleType::Tuple(ts) => {
      let tup: &PyTuple = v.downcast()?;
      if tup.len() == ts.len() {
        let elems = ts
          .iter()
          .enumerate()
          .map(|(i, t)| {
            let e = tup.get_item(i)?;
            from_python_tuple(e, t, env)
          })
          .collect::<PyResult<Box<_>>>()?;
        Ok(Tuple::Tuple(elems))
      } else {
        Err(PyIndexError::new_err("Invalid tuple size"))
      }
    }
    TupleType::Value(t) => from_python_value(v, t, env).map(Tuple::Value),
  }
}

pub fn from_python_value(v: &PyAny, ty: &ValueType, env: &PythonRuntimeEnvironment) -> PyResult<Value> {
  match ty {
    ValueType::I8 => Ok(Value::I8(v.extract()?)),
    ValueType::I16 => Ok(Value::I16(v.extract()?)),
    ValueType::I32 => Ok(Value::I32(v.extract()?)),
    ValueType::I64 => Ok(Value::I64(v.extract()?)),
    ValueType::I128 => Ok(Value::I128(v.extract()?)),
    ValueType::ISize => Ok(Value::ISize(v.extract()?)),
    ValueType::U8 => Ok(Value::U8(v.extract()?)),
    ValueType::U16 => Ok(Value::U16(v.extract()?)),
    ValueType::U32 => Ok(Value::U32(v.extract()?)),
    ValueType::U64 => Ok(Value::U64(v.extract()?)),
    ValueType::U128 => Ok(Value::U128(v.extract()?)),
    ValueType::USize => Ok(Value::USize(v.extract()?)),
    ValueType::F32 => Ok(Value::F32(v.extract()?)),
    ValueType::F64 => Ok(Value::F64(v.extract()?)),
    ValueType::Char => Ok(Value::Char(v.extract()?)),
    ValueType::Bool => Ok(Value::Bool(v.extract()?)),
    ValueType::Str => panic!("[Internal Error] Cannot convert python value into static string"),
    ValueType::String => Ok(Value::String(v.extract()?)),
    ValueType::Symbol => {
      let symbol_str: String = v.extract()?;
      let id = env.symbol_registry.register(symbol_str);
      Ok(Value::Symbol(id))
    }
    ValueType::DateTime => {
      let dt = utils::parse_date_time_string(v.extract()?).ok_or(PyTypeError::new_err("Cannot parse into DateTime"))?;
      Ok(Value::DateTime(dt))
    }
    ValueType::Duration => {
      let dt = utils::parse_duration_string(v.extract()?).ok_or(PyTypeError::new_err("Cannot parse into Duration"))?;
      Ok(Value::Duration(dt))
    }
    ValueType::Entity => Ok(Value::Entity(v.extract()?)),
    ValueType::Tensor => tensor_from_py_object(v, env),
  }
}

#[cfg(feature = "torch-tensor")]
fn tensor_from_py_object(pyobj: &PyAny, env: &PythonRuntimeEnvironment) -> PyResult<Value> {
  use super::torch::PyTensor;
  let py_tensor: PyTensor = pyobj.extract()?;
  let scl_tensor: Tensor = Tensor::new(py_tensor.0);
  let symbol: TensorSymbol = env.tensor_registry.register(scl_tensor);
  Ok(Value::TensorValue(symbol.into()))
}

#[cfg(not(feature = "torch-tensor"))]
#[allow(unused)]
fn tensor_from_py_object(pyobj: &PyAny, env: &PythonRuntimeEnvironment) -> PyResult<Value> {
  panic!(
    "This `scallopy` version is not compiled with tensor support; consider adding `torch-tensor` flag when compiling"
  )
}

#[cfg(feature = "torch-tensor")]
fn tensor_to_py_object(tensor: Tensor, py: Python<'_>) -> PyObject {
  use super::torch::PyTensor;
  let py_tensor = PyTensor(tensor.tensor);
  py_tensor.into_py(py)
}

#[cfg(not(feature = "torch-tensor"))]
#[allow(unused)]
fn tensor_to_py_object(tensor: Tensor, py: Python<'_>) -> PyObject {
  panic!(
    "This `scallopy` version is not compiled with tensor support; consider adding `torch-tensor` flag when compiling"
  )
}
