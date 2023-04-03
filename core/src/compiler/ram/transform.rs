use crate::common::output_option::OutputOption;

use super::*;

impl Stratum {
  /// Set the output option to `default` for all relations in the stratum
  pub fn output_all(&mut self) {
    self.relations.iter_mut().for_each(|(_, r)| {
      r.output = OutputOption::default();
    });
  }
}
