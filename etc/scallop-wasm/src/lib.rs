use wasm_bindgen::prelude::*;

use scallop_core::integrate::*;
use scallop_core::runtime::database::intentional::*;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::*;

#[wasm_bindgen]
pub fn interpret(source: String) -> String {
  result_to_string(interpret_string(source))
}

#[wasm_bindgen]
pub fn interpret_with_minmaxprob(source: String) -> String {
  let ctx = min_max_prob::MinMaxProbProvenance::default();
  match interpret_string_with_ctx(source, ctx) {
    Ok(result) => result
      .into_iter()
      .map(|(r, c)| format!("{}: {}", r, c))
      .collect::<Vec<_>>()
      .join("\n"),
    Err(err) => {
      format!("{}", err)
    }
  }
}

#[wasm_bindgen]
pub fn interpret_with_topkproofs(source: String, top_k: usize) -> String {
  let ctx = top_k_proofs::TopKProofsProvenance::<RcFamily>::new(top_k, false);
  match interpret_string_with_ctx(source, ctx) {
    Ok(result) => result
      .into_iter()
      .map(|(r, c)| format!("{}: {}", r, c))
      .collect::<Vec<_>>()
      .join("\n"),
    Err(err) => {
      format!("{}", err)
    }
  }
}

#[wasm_bindgen]
pub fn interpret_with_topbottomkclauses(source: String, k: usize) -> String {
  let ctx = top_bottom_k_clauses::TopBottomKClausesProvenance::<RcFamily>::new(k, false);
  match interpret_string_with_ctx(source, ctx) {
    Ok(result) => result
      .into_iter()
      .map(|(r, c)| format!("{}: {}", r, c))
      .collect::<Vec<_>>()
      .join("\n"),
    Err(err) => {
      format!("{}", err)
    }
  }
}

fn result_to_string<Prov: Provenance>(result: Result<IntentionalDatabase<Prov>, IntegrateError>) -> String {
  match result {
    Ok(result) => result
      .into_iter()
      .map(|(k, v)| format!("{}: {}", k, v.recovered_facts))
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
