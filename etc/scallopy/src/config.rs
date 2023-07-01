use pyo3::prelude::*;

#[pyfunction]
pub fn torch_tensor_enabled() -> bool {
  if cfg!(feature = "torch-tensor") {
    true
  } else {
    false
  }
}
