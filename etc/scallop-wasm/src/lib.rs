use scallop_core::utils::RcFamily;
use std::collections::*;
use wasm_bindgen::prelude::*;

use scallop_core::compiler;
use scallop_core::integrate::{self, IntegrateError};
use scallop_core::runtime::dynamic;
use scallop_core::runtime::provenance::*;

#[wasm_bindgen]
pub fn interpret(source: String) -> String {
  result_to_string(integrate::interpret_string(source))
}

#[wasm_bindgen]
pub fn interpret_with_minmaxprob(source: String) -> String {
  let ram = match compiler::compile_string_to_ram(source).map_err(IntegrateError::Compile) {
    Ok(ram) => ram,
    Err(err) => {
      return format!("{}", err);
    }
  };
  let mut ctx = min_max_prob::MinMaxProbContext::default();
  let result = match dynamic::interpret(ram, &mut ctx).map_err(IntegrateError::Runtime) {
    Ok(result) => result,
    Err(err) => {
      return format!("{}", err);
    }
  };
  result
    .into_iter()
    .map(|(r, c)| format!("{}: {}", r, c))
    .collect::<Vec<_>>()
    .join("\n")
}

#[wasm_bindgen]
pub fn interpret_with_topkproofs(source: String, top_k: usize) -> String {
  let mut ctx = top_k_proofs::TopKProofsContext::<RcFamily>::new(top_k);
  let ram = match compiler::compile_string_to_ram(source).map_err(IntegrateError::Compile) {
    Ok(ram) => ram,
    Err(err) => {
      return format!("{}", err);
    }
  };
  let result = match dynamic::interpret(ram, &mut ctx).map_err(IntegrateError::Runtime) {
    Ok(result) => result,
    Err(err) => {
      return format!("{}", err);
    }
  };
  result
    .into_iter()
    .map(|(r, c)| format!("{}: {}", r, c))
    .collect::<Vec<_>>()
    .join("\n")
}

#[wasm_bindgen]
pub fn interpret_with_topbottomkclauses(source: String, k: usize) -> String {
  let mut ctx = top_bottom_k_clauses::TopBottomKClausesContext::<RcFamily>::new(k);
  let ram = match compiler::compile_string_to_ram(source).map_err(IntegrateError::Compile) {
    Ok(ram) => ram,
    Err(err) => {
      return format!("{}", err);
    }
  };
  let result = match dynamic::interpret(ram, &mut ctx).map_err(IntegrateError::Runtime) {
    Ok(result) => result,
    Err(err) => {
      return format!("{}", err);
    }
  };
  result
    .into_iter()
    .map(|(r, c)| format!("{}: {}", r, c))
    .collect::<Vec<_>>()
    .join("\n")
}

fn result_to_string<T: Tag>(
  result: Result<BTreeMap<String, dynamic::DynamicOutputCollection<T>>, IntegrateError>,
) -> String {
  match result {
    Ok(result) => result
      .into_iter()
      .map(|(k, v)| format!("{}: {}", k, v))
      .collect::<Vec<_>>()
      .join("\n"),
    Err(err) => match err {
      IntegrateError::Compile(errors) => errors
        .into_iter()
        .map(|e| format!("{}", e))
        .collect::<Vec<_>>()
        .join("\n"),
      IntegrateError::Runtime(error) => {
        format!("{}", error)
      }
    },
  }
}
