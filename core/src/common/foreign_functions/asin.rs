use super::*;

/// Arcsine foreign function
///
/// ``` scl
/// extern fn $asin<T: Float>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Asin;

impl UnaryFloatFunction for Asin {
  fn name(&self) -> String {
    "asin".to_string()
  }

  fn execute_f32_partial(&self, arg: f32) -> Option<f32> {
    if arg >= -1.0 && arg <= 1.0 {
      Some(arg.asin())
    } else {
      None
    }
  }

  fn execute_f64_partial(&self, arg: f64) -> Option<f64> {
    if arg >= -1.0 && arg <= 1.0 {
      Some(arg.asin())
    } else {
      None
    }
  }
}
