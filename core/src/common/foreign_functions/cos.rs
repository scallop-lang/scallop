use super::*;

/// Cos value foreign function
///
/// ``` scl
/// extern fn $cos<T: Float>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Cos;

impl UnaryFloatFunction for Cos {
  fn name(&self) -> String {
    "cos".to_string()
  }

  fn execute_f32(&self, arg: f32) -> f32 {
    arg.cos()
  }

  fn execute_f64(&self, arg: f64) -> f64 {
    arg.cos()
  }
}
