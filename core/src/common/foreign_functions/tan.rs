use super::*;

/// Tan value foreign function
///
/// ``` scl
/// extern fn $tan<T: Float>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Tan;

impl UnaryFloatFunction for Tan {
  fn name(&self) -> String {
    "tan".to_string()
  }

  fn execute_f32(&self, arg: f32) -> f32 {
    arg.tan()
  }

  fn execute_f64(&self, arg: f64) -> f64 {
    arg.tan()
  }
}
