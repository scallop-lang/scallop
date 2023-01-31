mod collection;
mod context;
mod custom_tag;
mod error;
mod foreign_function;
mod io;
mod provenance;
mod tuple;

use pyo3::prelude::*;

use collection::*;
use context::*;

#[pymodule]
fn scallopy(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_class::<Context>()?;
  m.add_class::<Collection>()?;
  m.add_class::<CollectionIterator>()?;
  Ok(())
}
