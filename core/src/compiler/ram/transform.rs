use crate::common::output_option::OutputOption;

use super::*;

impl Stratum {
  pub fn output_all(&mut self) {
    self.relations.iter_mut().for_each(|(_, r)| {
      r.output = OutputOption::default();
    });
  }
}
