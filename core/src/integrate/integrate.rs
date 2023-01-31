use std::path::*;

use crate::runtime::database::intentional::*;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;

use super::*;

pub fn interpret_string(program_string: String) -> Result<IntentionalDatabase<unit::UnitProvenance>, IntegrateError> {
  let prov = unit::UnitProvenance::default();
  let mut interpret_ctx = InterpretContext::new(program_string, prov)?;
  interpret_ctx.run()?;
  Ok(interpret_ctx.idb())
}

pub fn interpret_string_with_ctx<Prov: Provenance>(
  program_string: String,
  prov: Prov,
) -> Result<IntentionalDatabase<Prov>, IntegrateError> {
  let mut interpret_ctx = InterpretContext::new(program_string, prov)?;
  interpret_ctx.run()?;
  Ok(interpret_ctx.idb())
}

pub fn interpret_string_with_ctx_and_monitor<Prov: Provenance, M: Monitor<Prov>>(
  program_string: String,
  prov: Prov,
  monitor: &M,
) -> Result<IntentionalDatabase<Prov>, IntegrateError> {
  let mut interpret_ctx = InterpretContext::new(program_string, prov)?;
  interpret_ctx.run_with_monitor(monitor)?;
  Ok(interpret_ctx.idb())
}

pub fn interpret_file(file_name: &PathBuf) -> Result<IntentionalDatabase<unit::UnitProvenance>, IntegrateError> {
  let prov = unit::UnitProvenance::default();
  let mut interpret_ctx = InterpretContext::new_from_file(file_name, prov)?;
  interpret_ctx.run()?;
  Ok(interpret_ctx.idb())
}

pub fn interpret_file_with_ctx<Prov: Provenance>(
  file_name: &PathBuf,
  prov: Prov,
) -> Result<IntentionalDatabase<Prov>, IntegrateError> {
  let mut interpret_ctx = InterpretContext::new_from_file(file_name, prov)?;
  interpret_ctx.run()?;
  Ok(interpret_ctx.idb())
}
