use super::*;

/// Exponential foreign function: e^x
///
/// ``` scl
/// extern fn $exp<T: Float>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Exp;

impl UnaryFloatFunction for Exp {
  fn name(&self) -> String {
    "exp".to_string()
  }

  fn execute_f32(&self, arg: f32) -> f32 {
    arg.exp()
  }

  fn execute_f64(&self, arg: f64) -> f64 {
    arg.exp()
  }
}
