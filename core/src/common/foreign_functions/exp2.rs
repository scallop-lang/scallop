use super::*;

/// Exponential foreign function: 2^x
///
/// ``` scl
/// extern fn $exp2<T: Float>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Exp2;

impl UnaryFloatFunction for Exp2 {
  fn name(&self) -> String {
    "exp2".to_string()
  }

  fn execute_f32(&self, arg: f32) -> f32 {
    arg.exp2()
  }

  fn execute_f64(&self, arg: f64) -> f64 {
    arg.exp2()
  }
}
