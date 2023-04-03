// use std::rc::Rc;

use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::{prelude::*, types::PyTuple};

use scallop_core::common::tuple::Tuple;
use scallop_core::common::tuple_type::TupleType;
use scallop_core::common::value::Value;
use scallop_core::common::value_type::ValueType;
use scallop_core::utils;

pub fn from_python_tuple(v: &PyAny, ty: &TupleType) -> PyResult<Tuple> {
  match ty {
    TupleType::Tuple(ts) => {
      let tup: &PyTuple = v.cast_as()?;
      if tup.len() == ts.len() {
        let elems = ts
          .iter()
          .enumerate()
          .map(|(i, t)| {
            let e = tup.get_item(i)?;
            from_python_tuple(e, t)
          })
          .collect::<PyResult<Box<_>>>()?;
        Ok(Tuple::Tuple(elems))
      } else {
        Err(PyIndexError::new_err("Invalid tuple size"))
      }
    }
    TupleType::Value(t) => from_python_value(v, t).map(Tuple::Value),
  }
}

pub fn to_python_tuple(tup: &Tuple) -> Py<PyAny> {
  match tup {
    Tuple::Tuple(t) => {
      Python::with_gil(|py| PyTuple::new(py, t.iter().map(to_python_tuple).collect::<Vec<_>>()).into())
    }
    Tuple::Value(v) => to_python_value(v),
  }
}

pub fn to_python_value(val: &Value) -> Py<PyAny> {
  use Value::*;
  match val {
    I8(i) => Python::with_gil(|py| i.to_object(py)),
    I16(i) => Python::with_gil(|py| i.to_object(py)),
    I32(i) => Python::with_gil(|py| i.to_object(py)),
    I64(i) => Python::with_gil(|py| i.to_object(py)),
    I128(i) => Python::with_gil(|py| i.to_object(py)),
    ISize(i) => Python::with_gil(|py| i.to_object(py)),
    U8(i) => Python::with_gil(|py| i.to_object(py)),
    U16(i) => Python::with_gil(|py| i.to_object(py)),
    U32(i) => Python::with_gil(|py| i.to_object(py)),
    U64(i) => Python::with_gil(|py| i.to_object(py)),
    U128(i) => Python::with_gil(|py| i.to_object(py)),
    USize(i) => Python::with_gil(|py| i.to_object(py)),
    F32(f) => Python::with_gil(|py| f.to_object(py)),
    F64(f) => Python::with_gil(|py| f.to_object(py)),
    Char(c) => Python::with_gil(|py| c.to_object(py)),
    Bool(b) => Python::with_gil(|py| b.to_object(py)),
    Str(s) => Python::with_gil(|py| s.to_object(py)),
    String(s) => Python::with_gil(|py| s.to_object(py)),
    // RcString(s) => Python::with_gil(|py| s.to_object(py)),
    DateTime(d) => Python::with_gil(|py| d.to_string().to_object(py)),
    Duration(d) => Python::with_gil(|py| d.to_string().to_object(py)),
  }
}

pub fn from_python_value(v: &PyAny, ty: &ValueType) -> PyResult<Value> {
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
    ValueType::Str => panic!(""),
    ValueType::String => Ok(Value::String(v.extract()?)),
    // ValueType::RcString => Ok(Tuple::Value(Value::RcString(Rc::new(
    //   v.extract::<String>()?,
    // )))),
    ValueType::DateTime => {
      let dt = utils::parse_date_time_string(v.extract()?).ok_or(PyTypeError::new_err("Cannot parse into DateTime"))?;
      Ok(Value::DateTime(dt))
    }
    ValueType::Duration => {
      let dt = utils::parse_duration_string(v.extract()?).ok_or(PyTypeError::new_err("Cannot parse into Duration"))?;
      Ok(Value::Duration(dt))
    }
  }
}
