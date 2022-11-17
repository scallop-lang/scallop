use super::*;

pub fn optimize_ram(ram: &mut Program) {
  let mut can_optimize = true;
  while can_optimize {
    can_optimize = false;
    can_optimize |= project_cascade(ram);
  }
}
