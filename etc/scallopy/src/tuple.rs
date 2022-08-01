// use std::rc::Rc;

use pyo3::exceptions::PyIndexError;
use pyo3::{prelude::*, types::PyTuple};

use scallop_core::common::tuple::Tuple;
use scallop_core::common::tuple_type::TupleType;
use scallop_core::common::value::Value;
use scallop_core::common::value_type::ValueType;

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
    TupleType::Value(t) => match t {
      ValueType::I8 => Ok(Tuple::Value(Value::I8(v.extract()?))),
      ValueType::I16 => Ok(Tuple::Value(Value::I16(v.extract()?))),
      ValueType::I32 => Ok(Tuple::Value(Value::I32(v.extract()?))),
      ValueType::I64 => Ok(Tuple::Value(Value::I64(v.extract()?))),
      ValueType::I128 => Ok(Tuple::Value(Value::I128(v.extract()?))),
      ValueType::ISize => Ok(Tuple::Value(Value::ISize(v.extract()?))),
      ValueType::U8 => Ok(Tuple::Value(Value::U8(v.extract()?))),
      ValueType::U16 => Ok(Tuple::Value(Value::U16(v.extract()?))),
      ValueType::U32 => Ok(Tuple::Value(Value::U32(v.extract()?))),
      ValueType::U64 => Ok(Tuple::Value(Value::U64(v.extract()?))),
      ValueType::U128 => Ok(Tuple::Value(Value::U128(v.extract()?))),
      ValueType::USize => Ok(Tuple::Value(Value::USize(v.extract()?))),
      ValueType::F32 => Ok(Tuple::Value(Value::F32(v.extract()?))),
      ValueType::F64 => Ok(Tuple::Value(Value::F64(v.extract()?))),
      ValueType::Char => Ok(Tuple::Value(Value::Char(v.extract()?))),
      ValueType::Bool => Ok(Tuple::Value(Value::Bool(v.extract()?))),
      ValueType::Str => panic!(""),
      ValueType::String => Ok(Tuple::Value(Value::String(v.extract()?))),
      // ValueType::RcString => Ok(Tuple::Value(Value::RcString(Rc::new(
      //   v.extract::<String>()?,
      // )))),
    },
  }
}

pub fn to_python_tuple(tup: &Tuple) -> Py<PyAny> {
  match tup {
    Tuple::Tuple(t) => {
      Python::with_gil(|py| PyTuple::new(py, t.iter().map(to_python_tuple).collect::<Vec<_>>()).into())
    }
    Tuple::Value(v) => {
      use Value::*;
      match v {
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
      }
    }
  }
}
