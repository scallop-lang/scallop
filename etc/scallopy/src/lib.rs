mod collection;
mod config;
mod context;
mod custom_tag;
mod error;
mod external_tag;
mod foreign_attribute;
mod foreign_function;
mod foreign_predicate;
mod io;
mod provenance;
mod runtime;
mod tag;
mod tuple;

#[cfg(feature = "torch-tensor")]
mod torch;

use pyo3::prelude::*;

use collection::*;
use context::*;

#[pymodule]
fn scallopy(_py: Python, m: &PyModule) -> PyResult<()> {
  // Configurations
  m.add_function(wrap_pyfunction!(config::torch_tensor_enabled, m).unwrap())?;

  // Add classes
  m.add_class::<Context>()?;
  m.add_class::<Collection>()?;
  m.add_class::<CollectionIterator>()?;

  // Ok
  Ok(())
}
