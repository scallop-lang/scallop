use super::*;

/// Arctangent foreign function
///
/// ``` scl
/// extern fn $atan<T: Float>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Atan;

impl UnaryFloatFunction for Atan {
  fn name(&self) -> String {
    "atan".to_string()
  }

  fn execute_f32(&self, arg: f32) -> f32 {
    arg.atan()
  }

  fn execute_f64(&self, arg: f64) -> f64 {
    arg.atan()
  }
}
