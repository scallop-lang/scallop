use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::types::*;

use scallop_core::common::foreign_tensor::*;
use scallop_core::common::tuple::Tuple;
use scallop_core::common::tuple_type::TupleType;
use scallop_core::common::value::Value;
use scallop_core::common::value_type::ValueType;
use scallop_core::utils;

use super::error::*;
use super::runtime::*;
use super::tensor::*;

pub fn to_python_tuple(tup: &Tuple, env: &PythonRuntimeEnvironment) -> Option<Py<PyAny>> {
  match tup {
    Tuple::Tuple(t) => Python::with_gil(|py| {
      let values = t.iter().map(|t| to_python_tuple(t, env)).collect::<Option<Vec<_>>>()?;
      Some(PyTuple::new(py, values).into())
    }),
    Tuple::Value(v) => to_python_value(v, env),
  }
}

pub fn to_python_value(val: &Value, env: &PythonRuntimeEnvironment) -> Option<Py<PyAny>> {
  use Value::*;
  Python::with_gil(|py| match val {
    I8(i) => Some(i.to_object(py)),
    I16(i) => Some(i.to_object(py)),
    I32(i) => Some(i.to_object(py)),
    I64(i) => Some(i.to_object(py)),
    I128(i) => Some(i.to_object(py)),
    ISize(i) => Some(i.to_object(py)),
    U8(i) => Some(i.to_object(py)),
    U16(i) => Some(i.to_object(py)),
    U32(i) => Some(i.to_object(py)),
    U64(i) => Some(i.to_object(py)),
    U128(i) => Some(i.to_object(py)),
    USize(i) => Some(i.to_object(py)),
    F32(f) => Some(f.to_object(py)),
    F64(f) => Some(f.to_object(py)),
    Char(c) => Some(c.to_object(py)),
    Bool(b) => Some(b.to_object(py)),
    Str(s) => Some(s.to_object(py)),
    String(s) => Some(s.to_object(py)),
    Symbol(s) => Some(env.symbol_registry.get_symbol(*s).to_object(py)),
    SymbolString(s) => Some(s.to_object(py)),
    DateTime(d) => Some(d.to_string().to_object(py)),
    Duration(d) => Some(d.to_string().to_object(py)),
    Entity(e) => Some(e.to_object(py)),
    EntityString(_) => panic!("[Internal Error] Entity string could not be turned to python value"),
    Tensor(t) => tensor_to_py_object(t.clone()),
    TensorValue(v) => tensor_to_py_object(env.tensor_registry.eval(v)?),
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
      let string = v.extract()?;
      let dt = utils::parse_date_time_string(string)
        .ok_or(PyTypeError::new_err(format!("Cannot parse into DateTime: {}", string)))?;
      Ok(Value::DateTime(dt))
    }
    ValueType::Duration => {
      let string = v.extract()?;
      let dt = utils::parse_duration_string(string)
        .ok_or(PyTypeError::new_err(format!("Cannot parse into Duration: {}", string)))?;
      Ok(Value::Duration(dt))
    }
    ValueType::Entity => {
      let entity_string: String = v.extract()?;
      Ok(Value::EntityString(entity_string))
    }
    ValueType::Tensor => tensor_from_py_object(v, env),
  }
}

fn tensor_from_py_object(pyobj: &PyAny, env: &PythonRuntimeEnvironment) -> PyResult<Value> {
  let py_tensor = Tensor::from_py_value(pyobj);
  let symbol = env
    .tensor_registry
    .register(DynamicExternalTensor::new(py_tensor))
    .ok_or(BindingError::CannotRegisterTensor)?;
  Ok(Value::TensorValue(symbol.into()))
}

fn tensor_to_py_object(tensor: DynamicExternalTensor) -> Option<PyObject> {
  tensor.cast::<Tensor>().map(|t| t.to_py_value())
}
