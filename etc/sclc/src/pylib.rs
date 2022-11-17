use proc_macro2::TokenStream;
use quote::*;
use scallop_core::common::output_option::OutputOption;
use std::env;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::process::Command;

use scallop_core::compiler;

use super::error::*;
use super::options::*;

pub fn create_pylib(
  opt: &Options,
  compile_opt: compiler::CompileOptions,
  ram: &compiler::ram::Program,
) -> Result<(), SclcError> {
  // Get program name
  let program_name = opt.input.file_prefix().unwrap().to_str().unwrap();

  // Rust source
  let file_content = generate_pylib_code(compile_opt, program_name, ram).to_string();

  // Create a folder
  let parent_dir = opt.input.parent().unwrap();
  let tmp_dir = parent_dir.join(format!(".{}.pylib.sclcmpl", program_name));
  let scallop_source_dir = env::var("SCALLOPDIR").expect(
    "Please set envrionment variable `SCALLOPDIR` to be the root of Scallop source directory before using `sclc`.",
  );

  // Create a temporary directory holding the cargo project
  fs::create_dir_all(&tmp_dir).unwrap();
  fs::create_dir_all(&tmp_dir.join("src")).unwrap();
  fs::create_dir_all(&tmp_dir.join("target")).unwrap();
  fs::create_dir_all(&tmp_dir.join("target").join("wheels")).unwrap();
  fs::create_dir_all(&tmp_dir.join("target").join("wheels").join("current")).unwrap();

  // Create a Cargo.toml file
  let mut cargo_toml_file = File::create(tmp_dir.join("Cargo.toml")).unwrap();
  cargo_toml_file
    .write_all(
      format!(
        r#"[package]
name = "{}"
version = "1.0.0"
edition = "2018"
[lib]
name = "{}"
crate-type = ["cdylib"]
[dependencies]
scallop-core = {{ path = "{}/core" }}
rayon = "1.5"
[dependencies.pyo3]
version = "0.16.5"
features = ["extension-module"]
[workspace]
"#,
        program_name, program_name, scallop_source_dir,
      )
      .as_bytes(),
    )
    .unwrap();

  // Create a lib.rs file
  let mut lib_rs_file = File::create(tmp_dir.join("src/lib.rs")).unwrap();
  lib_rs_file.write_all(file_content.as_bytes()).unwrap();

  // Compile the file: create command
  let mut cmd = Command::new("maturin");

  // Add arguments
  cmd
    .current_dir(&tmp_dir)
    .arg("build")
    .arg("--release")
    .arg("--out")
    .arg(PathBuf::new().join("target").join("wheels").join("current"));

  // Run the command
  let output = cmd.output().unwrap();
  if output.status.success() {
    let whls = fs::read_dir(&tmp_dir.join("target").join("wheels").join("current"))
      .unwrap()
      .collect::<Vec<_>>();
    assert_eq!(whls.len(), 1, "Only one wheel should be generated for compilation");
    let whl = whls.into_iter().next().unwrap().unwrap().path();
    let whl_name = whl.file_name().unwrap();

    // Copy the wheel to the wheels directory
    fs::copy(whl.clone(), &tmp_dir.join("target").join("wheels").join(whl_name)).unwrap();

    // If we want to copy the wheel to the parent directory
    if !opt.do_not_copy_artifact {
      fs::copy(whl.clone(), parent_dir.join(whl_name)).unwrap();
    }

    // If we want to remove the temporary directory
    if opt.do_not_keep_temporary_directory {
      fs::remove_dir_all(tmp_dir).unwrap();
    }

    Ok(())
  } else {
    println!("[Compile Error]");
    println!("stdout: {}", std::str::from_utf8(&output.stdout).unwrap());
    println!("{}", std::str::from_utf8(&output.stderr).unwrap());
    Ok(())
  }
}

fn generate_pylib_code(
  compile_opt: compiler::CompileOptions,
  program_name: &str,
  ram: &compiler::ram::Program,
) -> TokenStream {
  let program_name_ident = format_ident!("{}", program_name);

  // Turn the ram module into a sequence of rust tokens
  let module = ram.to_rs_module(&compile_opt);

  // Generate code
  let create_edb_code = ram.to_rs_create_edb_fn();
  let static_context_code = generate_static_context_code(ram);
  let binding_error_code = generate_binding_error_code();
  let context_code = generate_context_code();
  let helper_functions = generate_helper_functions();

  // Generate the code
  quote! {
    use pyo3::prelude::*;
    use pyo3::exceptions::*;
    use pyo3::types::*;
    use rayon::prelude::*;
    use scallop_core::common::tuple::*;
    use scallop_core::common::tuple_type::*;
    use scallop_core::common::value::*;
    use scallop_core::common::value_type::*;
    use scallop_core::runtime::provenance::*;
    use scallop_core::runtime::edb::*;
    use scallop_core::utils::*;
    use std::collections::*;
    mod scallop_module { #module #create_edb_code }
    #static_context_code
    #binding_error_code
    #context_code
    #helper_functions
    #[pymodule]
    fn #program_name_ident(_py: Python, m: &PyModule) -> PyResult<()> {
      m.add_class::<Context>()?;
      Ok(())
    }
  }
}

fn generate_static_context_code(ram: &compiler::ram::Program) -> TokenStream {
  let get_relation_match_cases = ram
    .relations()
    .filter_map(|r| {
      let relation_name = r.predicate.clone();
      let field_name = r.to_rs_field_name();
      match &r.output {
        OutputOption::Default => Some(quote! {
          #relation_name => Ok(output.#field_name.to_dynamic_vec()),
        }),
        _ => None,
      }
    })
    .collect::<Vec<_>>();

  quote! {
    #[derive(Clone)]
    struct StaticContext<C: Provenance> {
      prov_ctx: C,
      edb: EDB<C>,
      output: Option<scallop_module::OutputRelations<C>>,
    }

    impl<C: Provenance> StaticContext<C> {
      fn new(prov_ctx: C) -> Self {
        let edb = scallop_module::create_edb();
        Self { prov_ctx, edb, output: None }
      }

      fn add_py_facts<F>(&mut self, r: &str, py_facts: Vec<&PyAny>, f: F) -> Result<(), BindingError> where F: Fn(&PyAny) -> Result<(Option<C::InputTag>, &PyAny), BindingError> {
        if let Some(ty) = self.edb.type_of(r) {
          let facts = py_facts.into_iter().map(|py_fact| {
            let (input_tag, py_tup) = f(py_fact)?;
            let tup = from_python_tuple(py_tup, &ty).map_err(|_| BindingError(format!("Cannot parse tuple `{}` with type `{}`", py_tup, ty)))?;
            Ok(EDBFact::new(input_tag, tup))
          }).collect::<Result<Vec<_>, _>>()?;
          // Add facts and disjunctions to relation
          let rel = self.edb.get_relation_mut(r);
          rel.extend_facts(facts);
          // Return Ok
          Ok(())
        } else {
          Err(BindingError(format!("Unknown relation `{}`", r)))
        }
      }

      fn add_facts_with_disjunction(&mut self, r: &str, facts: Vec<(Option<C::InputTag>, Tuple)>, disjunctions: Option<Vec<Vec<usize>>>) -> Result<(), BindingError> {
        let rel = self.edb.get_relation_mut(r);
        rel.extend_facts(facts.into_iter().map(|(tag, tup)| EDBFact::new(tag, tup)).collect());
        if let Some(ds) = disjunctions {
          rel.extend_disjunctions(ds.into_iter());
        }
        Ok(())
      }

      fn add_py_facts_with_disjunction<F>(&mut self, r: &str, py_facts: Vec<&PyAny>, disjunctions: Option<Vec<Vec<usize>>>, f: F) -> Result<(), BindingError> where F: Fn(&PyAny) -> Result<(Option<C::InputTag>, &PyAny), BindingError> {
        if let Some(ty) = self.edb.type_of(r) {
          let facts = py_facts.into_iter().map(|py_fact| {
            let (input_tag, py_tup) = f(py_fact)?;
            let tup = from_python_tuple(py_tup, &ty).map_err(|_| BindingError(format!("Cannot parse tuple `{}` with type `{}`", py_tup, ty)))?;
            Ok(EDBFact::new(input_tag, tup))
          }).collect::<Result<Vec<_>, _>>()?;

          // Add facts and disjunctions to relation
          let rel = self.edb.get_relation_mut(r);
          rel.extend_facts(facts);
          if let Some(ds) = disjunctions {
            rel.extend_disjunctions(ds.into_iter());
          }

          // Return ok
          Ok(())
        } else {
          Err(BindingError(format!("Unknown relation `{}`", r)))
        }
      }

      fn run(&mut self) {
        let edb_clone = self.edb.clone();
        let res = scallop_module::run_with_edb(&mut self.prov_ctx, edb_clone);
        self.output = Some(res);
      }

      fn relation(&self, r: &str) -> Result<Vec<(C::OutputTag, Tuple)>, BindingError> {
        if let Some(output) = &self.output {
          match r {
            #(#get_relation_match_cases)*
            r => Err(BindingError(format!("Unknown relation `{}`", r))),
          }
        } else {
          Err(BindingError(format!("There is no execution result yet; please call `.run()` first.")))
        }
      }

      fn py_relation<F>(&self, r: &str, f: F) -> Result<Vec<Py<PyAny>>, BindingError> where F: Fn((C::OutputTag, Tuple)) -> Py<PyAny> {
        self.relation(r).map(|c| c.into_iter().map(f).collect())
      }
    }
  }
}

fn generate_binding_error_code() -> TokenStream {
  quote! {
    #[derive(Debug)]
    struct BindingError(String);

    impl std::error::Error for BindingError {}

    impl std::fmt::Display for BindingError {
      fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
      }
    }

    impl std::convert::From<BindingError> for PyErr {
      fn from(err: BindingError) -> Self {
        let err_str = format!("Scallop Error: {}", err.0);
        let py_err_str: Py<PyAny> = Python::with_gil(|py| err_str.to_object(py));
        PyException::new_err(py_err_str)
      }
    }
  }
}

fn generate_context_code() -> TokenStream {
  quote! {
    #[derive(Clone)]
    enum ContextEnum {
      Unit(StaticContext<unit::UnitProvenance>),
      MinMaxProb(StaticContext<min_max_prob::MinMaxProbContext>),
      AddMultProb(StaticContext<add_mult_prob::AddMultProbContext>),
      DiffMinMaxProb(StaticContext<diff_min_max_prob::DiffMinMaxProbContext<Py<PyAny>, ArcFamily>>),
      DiffTopKProofs(StaticContext<diff_top_k_proofs::DiffTopKProofsContext<Py<PyAny>, ArcFamily>>),
      DiffTopBottomKClauses(StaticContext<diff_top_bottom_k_clauses::DiffTopBottomKClausesContext<Py<PyAny>, ArcFamily>>),
    }

    #[pyclass(unsendable, name = "StaticContext")]
    struct Context { ctx: ContextEnum }

    #[pymethods]
    impl Context {
      #[new]
      #[args(provenance = "\"unit\"", top_k = "None")]
      fn new(provenance: &str, top_k: Option<usize>) -> Result<Self, BindingError> {
        let top_k = top_k.unwrap_or(3);
        match provenance {
          "unit" => Ok(Self { ctx: ContextEnum::Unit(StaticContext::new(unit::UnitProvenance::default())) }),
          "minmaxprob" => Ok(Self { ctx: ContextEnum::MinMaxProb(StaticContext::new(min_max_prob::MinMaxProbContext::default())) }),
          "addmultprob" => Ok(Self { ctx: ContextEnum::AddMultProb(StaticContext::new(add_mult_prob::AddMultProbContext::default())) }),
          "diffminmaxprob" => Ok(Self { ctx: ContextEnum::DiffMinMaxProb(StaticContext::new(diff_min_max_prob::DiffMinMaxProbContext::default())) }),
          "difftopkproofs" => Ok(Self { ctx: ContextEnum::DiffTopKProofs(StaticContext::new(diff_top_k_proofs::DiffTopKProofsContext::new(top_k))) }),
          "difftopbottomkclauses" => Ok(Self { ctx: ContextEnum::DiffTopBottomKClauses(StaticContext::new(diff_top_bottom_k_clauses::DiffTopBottomKClausesContext::new(top_k))) }),
          p => Err(BindingError(format!("Unknown provenance `{}`", p.to_string()))),
        }
      }

      fn input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        match &self.ctx {
          ContextEnum::Unit(_) => None,
          ContextEnum::MinMaxProb(_) => None,
          ContextEnum::AddMultProb(_) => None,
          ContextEnum::DiffMinMaxProb(_) => None,
          ContextEnum::DiffTopKProofs(c) => Some(c.prov_ctx.input_tags()),
          ContextEnum::DiffTopBottomKClauses(c) => Some(c.prov_ctx.input_tags()),
        }
      }

      fn add_facts(&mut self, r: &str, facts: Vec<&PyAny>) -> Result<(), BindingError> {
        match &mut self.ctx {
          ContextEnum::Unit(c) => c.add_py_facts(r, facts, to_unit_fact),
          ContextEnum::MinMaxProb(c) => c.add_py_facts(r, facts, to_f64_fact),
          ContextEnum::AddMultProb(c) => c.add_py_facts(r, facts, to_f64_fact),
          ContextEnum::DiffMinMaxProb(c) => c.add_py_facts(r, facts, to_diff_prob_fact),
          ContextEnum::DiffTopKProofs(c) => c.add_py_facts(r, facts, to_diff_prob_fact),
          ContextEnum::DiffTopBottomKClauses(c) => c.add_py_facts(r, facts, to_diff_prob_fact),
        }
      }

      #[args(disjunctions = "None")]
      fn add_facts_with_disjunction(&mut self, r: &str, facts: Vec<&PyAny>, disjunctions: Option<Vec<Vec<usize>>>) -> Result<(), BindingError> {
        match &mut self.ctx {
          ContextEnum::Unit(c) => c.add_py_facts_with_disjunction(r, facts, disjunctions, to_unit_fact),
          ContextEnum::MinMaxProb(c) => c.add_py_facts_with_disjunction(r, facts, disjunctions, to_f64_fact),
          ContextEnum::AddMultProb(c) => c.add_py_facts_with_disjunction(r, facts, disjunctions, to_f64_fact),
          ContextEnum::DiffMinMaxProb(c) => c.add_py_facts_with_disjunction(r, facts, disjunctions, to_diff_prob_fact),
          ContextEnum::DiffTopKProofs(c) => c.add_py_facts_with_disjunction(r, facts, disjunctions, to_diff_prob_fact),
          ContextEnum::DiffTopBottomKClauses(c) => c.add_py_facts_with_disjunction(r, facts, disjunctions, to_diff_prob_fact),
        }
      }

      fn run(&mut self) {
        match &mut self.ctx {
          ContextEnum::Unit(c) => c.run(),
          ContextEnum::MinMaxProb(c) => c.run(),
          ContextEnum::AddMultProb(c) => c.run(),
          ContextEnum::DiffMinMaxProb(c) => c.run(),
          ContextEnum::DiffTopKProofs(c) => c.run(),
          ContextEnum::DiffTopBottomKClauses(c) => c.run(),
        }
      }

      fn run_batch_parallel(&self, inputs: HashMap<String, Vec<(Vec<&PyAny>, Option<Vec<Vec<usize>>>)>>, outputs: Vec<&str>) -> Result<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>, BindingError> {
        match &self.ctx {
          ContextEnum::Unit(c) => run_batch_parallel(c, inputs, outputs, to_unit_fact, from_unit_fact, get_none),
          ContextEnum::MinMaxProb(c) => run_batch_parallel(c, inputs, outputs, to_f64_fact, from_f64_fact, get_none),
          ContextEnum::AddMultProb(c) => run_batch_parallel(c, inputs, outputs, to_f64_fact, from_f64_fact, get_none),
          ContextEnum::DiffMinMaxProb(c) => run_batch_parallel(c, inputs, outputs, to_diff_prob_fact, from_diff_min_max_prob_fact, get_none),
          ContextEnum::DiffTopKProofs(c) => run_batch_parallel(c, inputs, outputs, to_diff_prob_fact, from_diff_prob_fact, |c| Some(c.input_tags())),
          ContextEnum::DiffTopBottomKClauses(c) => run_batch_parallel(c, inputs, outputs, to_diff_prob_fact, from_diff_prob_fact, |c| Some(c.input_tags())),
        }
      }

      fn run_batch_non_parallel(&self, inputs: HashMap<String, Vec<(Vec<&PyAny>, Option<Vec<Vec<usize>>>)>>, outputs: Vec<&str>) -> Result<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>, BindingError> {
        match &self.ctx {
          ContextEnum::Unit(c) => run_batch_non_parallel(c, inputs, outputs, to_unit_fact, from_unit_fact, get_none),
          ContextEnum::MinMaxProb(c) => run_batch_non_parallel(c, inputs, outputs, to_f64_fact, from_f64_fact, get_none),
          ContextEnum::AddMultProb(c) => run_batch_non_parallel(c, inputs, outputs, to_f64_fact, from_f64_fact, get_none),
          ContextEnum::DiffMinMaxProb(c) => run_batch_non_parallel(c, inputs, outputs, to_diff_prob_fact, from_diff_min_max_prob_fact, get_none),
          ContextEnum::DiffTopKProofs(c) => run_batch_non_parallel(c, inputs, outputs, to_diff_prob_fact, from_diff_prob_fact, |c| Some(c.input_tags())),
          ContextEnum::DiffTopBottomKClauses(c) => run_batch_non_parallel(c, inputs, outputs, to_diff_prob_fact, from_diff_prob_fact, |c| Some(c.input_tags())),
        }
      }

      fn relation(&self, r: &str) -> Result<Vec<Py<PyAny>>, BindingError> {
        match &self.ctx {
          ContextEnum::Unit(c) => c.py_relation(r, from_unit_fact),
          ContextEnum::MinMaxProb(c) => c.py_relation(r, from_f64_fact),
          ContextEnum::AddMultProb(c) => c.py_relation(r, from_f64_fact),
          ContextEnum::DiffMinMaxProb(c) => c.py_relation(r, from_diff_min_max_prob_fact),
          ContextEnum::DiffTopKProofs(c) => c.py_relation(r, from_diff_prob_fact),
          ContextEnum::DiffTopBottomKClauses(c) => c.py_relation(r, from_diff_prob_fact),
        }
      }
    }
  }
}

fn generate_helper_functions() -> TokenStream {
  quote! {
    fn run_batch_parallel<C, F, G, H>(
      c: &StaticContext<C>,
      inputs: HashMap<String, Vec<(Vec<&PyAny>, Option<Vec<Vec<usize>>>)>>,
      outputs: Vec<&str>,
      f: F,
      g: G,
      h: H,
    ) -> Result<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>, BindingError>
    where
      C: Provenance + std::marker::Sync,
      F: Fn(&PyAny) -> Result<(Option<C::InputTag>, &PyAny), BindingError> + std::marker::Sync, // From python obj to input fact
      G: Fn((C::OutputTag, Tuple)) -> Py<PyAny> + Copy + std::marker::Send + std::marker::Sync, // From output fact to python obj
      H: Fn(&C) -> Option<Vec<Py<PyAny>>> + std::marker::Sync, // From provenance context to input tags
      <C as Provenance>::InputTag: std::marker::Sync,
      <C as Provenance>::OutputTag: std::marker::Sync + std::marker::Send,
      <C as Provenance>::Tag: std::marker::Sync + std::marker::Send,
    {
      let batch_size = inputs.iter().next().unwrap().1.len();

      // Create contexts for parallel computing
      let prep_inputs = inputs
        .into_iter()
        .map(|(r, batch)| {
          let prep_batch = batch
            .into_iter()
            .map(|(py_facts, disjunctions)| {
              if let Some(ty) = c.edb.type_of(&r) {
                let facts = py_facts.into_iter().map(|py_fact| {
                  let (input_tag, py_tup) = f(py_fact)?;
                  let tup = from_python_tuple(py_tup, &ty).map_err(|_| BindingError(format!("Cannot parse tuple `{}` with type `{}`", py_tup, ty)))?;
                  Ok((input_tag, tup))
                }).collect::<Result<Vec<_>, _>>()?;
                Ok((facts, disjunctions))
              } else {
                Err(BindingError(format!("Unknown relation `{}`", r)))
              }
            })
            .collect::<Result<Vec<(Vec<(Option<C::InputTag>, Tuple)>, Option<Vec<Vec<usize>>>)>, BindingError>>()?;
          Ok((r, prep_batch))
        })
        .collect::<Result<HashMap<String, Vec<(Vec<(Option<C::InputTag>, Tuple)>, Option<Vec<Vec<usize>>>)>>, BindingError>>()?;

      // Actually execute the program in parallel
      let base_results = (0..batch_size)
        .into_par_iter()
        .map(|i| {
          let mut temp_ctx = c.clone();
          for (r, batch) in prep_inputs.iter() {
            temp_ctx.add_facts_with_disjunction(r, batch[i].0.clone(), batch[i].1.clone())?;
          }
          temp_ctx.run();
          let input_tags = h(&temp_ctx.prov_ctx);
          let relations = outputs.iter().map(|o| temp_ctx.relation(o)).collect::<Result<Vec<_>, _>>()?;
          Ok((input_tags, relations))
        })
        .collect::<Result<Vec<_>, BindingError>>()?;

      // Recover python collection
      let results = base_results
        .into_iter()
        .map(|(input_tags, relations)| {
          (input_tags, relations.into_iter().map(|rs| rs.into_iter().map(g).collect()).collect())
        })
        .collect();

      // Return
      Ok(results)
    }

    fn run_batch_non_parallel<C, F, G, H>(
      c: &StaticContext<C>,
      inputs: HashMap<String, Vec<(Vec<&PyAny>, Option<Vec<Vec<usize>>>)>>,
      outputs: Vec<&str>,
      f: F,
      g: G,
      h: H,
    ) -> Result<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>, BindingError>
    where
      C: Provenance,
      F: Fn(&PyAny) -> Result<(Option<C::InputTag>, &PyAny), BindingError>, // From python obj to input fact
      G: Fn((C::OutputTag, Tuple)) -> Py<PyAny> + Copy, // From output fact to python obj
      H: Fn(&C) -> Option<Vec<Py<PyAny>>>, // From provenance context to input tags
    {
      let batch_size = inputs.iter().next().unwrap().1.len();

      // Create contexts for parallel computing
      let prep_inputs = inputs
        .into_iter()
        .map(|(r, batch)| {
          let prep_batch = batch
            .into_iter()
            .map(|(py_facts, disjunctions)| {
              if let Some(ty) = c.edb.type_of(&r) {
                let facts = py_facts.into_iter().map(|py_fact| {
                  let (input_tag, py_tup) = f(py_fact)?;
                  let tup = from_python_tuple(py_tup, &ty).map_err(|_| BindingError(format!("Cannot parse tuple `{}` with type `{}`", py_tup, ty)))?;
                  Ok((input_tag, tup))
                }).collect::<Result<Vec<_>, _>>()?;
                Ok((facts, disjunctions))
              } else {
                Err(BindingError(format!("Unknown relation `{}`", r)))
              }
            })
            .collect::<Result<Vec<(Vec<(Option<C::InputTag>, Tuple)>, Option<Vec<Vec<usize>>>)>, BindingError>>()?;
          Ok((r, prep_batch))
        })
        .collect::<Result<HashMap<String, Vec<(Vec<(Option<C::InputTag>, Tuple)>, Option<Vec<Vec<usize>>>)>>, BindingError>>()?;

      // Actually execute the program in parallel
      let base_results = (0..batch_size)
        .into_iter()
        .map(|i| {
          let mut temp_ctx = c.clone();
          for (r, batch) in prep_inputs.iter() {
            temp_ctx.add_facts_with_disjunction(r, batch[i].0.clone(), batch[i].1.clone())?;
          }
          temp_ctx.run();
          let input_tags = h(&temp_ctx.prov_ctx);
          let relations = outputs.iter().map(|o| temp_ctx.relation(o)).collect::<Result<Vec<_>, _>>()?;
          Ok((input_tags, relations))
        })
        .collect::<Result<Vec<_>, BindingError>>()?;

      // Recover python collection
      let results = base_results
        .into_iter()
        .map(|(input_tags, relations)| {
          (input_tags, relations.into_iter().map(|rs| rs.into_iter().map(g).collect()).collect())
        })
        .collect();

      // Return
      Ok(results)
    }

    fn to_unit_fact(t: &PyAny) -> Result<(Option<()>, &PyAny), BindingError> {
      Ok((None, t))
    }

    fn from_unit_fact((_, tup): (unit::Unit, Tuple)) -> Py<PyAny> {
      to_python_tuple(&tup)
    }

    fn to_f64_fact(f: &PyAny) -> Result<(Option<f64>, &PyAny), BindingError> {
      let (f_tag, py_tuple): (Option<f64>, &PyAny) = f.extract().map_err(|_| BindingError(format!("Cannot extract tag and tuple from {}", f)))?;
      Ok((f_tag, py_tuple))
    }

    fn from_f64_fact((tag, tup): (f64, Tuple)) -> Py<PyAny> {
      Python::with_gil(|py| (tag, to_python_tuple(&tup)).to_object(py))
    }

    fn to_diff_prob_fact(f: &PyAny) -> Result<(Option<InputDiffProb<Py<PyAny>>>, &PyAny), BindingError> {
      let (tag, py_tuple): (Option<&PyAny>, &PyAny) = f.extract().map_err(|_| BindingError(format!("Cannot extract tag and tuple from {}", f)))?;
      if let Some(t) = tag {
        let f_tag: f64 = t.extract().map_err(|_| BindingError(format!("Cannot extract float from {}", t)))?;
        let base_tag: Py<PyAny> = t.into();
        Ok((Some(InputDiffProb(f_tag, base_tag)), py_tuple))
      } else {
        Ok((None, py_tuple))
      }
    }

    fn from_diff_prob_fact((tag, tup): (OutputDiffProb<Py<PyAny>>, Tuple)) -> Py<PyAny> {
      Python::with_gil(|py| {
        let py_tag = (tag.0, tag.1.clone()).to_object(py);
        let py_tup = to_python_tuple(&tup);
        (py_tag, py_tup).to_object(py)
      })
    }

    fn from_diff_min_max_prob_fact((tag, tup): (diff_min_max_prob::OutputDiffProb<Py<PyAny>>, Tuple)) -> Py<PyAny> {
      Python::with_gil(|py| (diff_min_max_prob_output_tag_to_py(&tag), to_python_tuple(&tup)).to_object(py))
    }

    fn diff_min_max_prob_output_tag_to_py(t: &diff_min_max_prob::OutputDiffProb<Py<PyAny>>) -> Py<PyAny> {
      match t.2 {
        1 => Python::with_gil(|py| (1, t.3.clone().unwrap()).to_object(py)),
        0 => Python::with_gil(|py| (0, t.0).to_object(py)),
        _ => Python::with_gil(|py| (-1, t.3.clone().unwrap()).to_object(py)),
      }
    }

    fn get_none<C: Provenance>(ctx: &C) -> Option<Vec<Py<PyAny>>> {
      None
    }

    fn to_python_tuple(tup: &Tuple) -> Py<PyAny> {
      match tup {
        Tuple::Tuple(t) => Python::with_gil(|py| PyTuple::new(py, t.iter().map(to_python_tuple).collect::<Vec<_>>()).into()),
        Tuple::Value(v) => {
          use Value::*;
          match v {
            I8(i) => Python::with_gil(|py| i.to_object(py)),
            I16(i) => Python::with_gil(|py| i.to_object(py)),
            I32(i) => Python::with_gil(|py| i.to_object(py)),
            I64(i) => Python::with_gil(|py| i.to_object(py)),
            I128(i) => Python::with_gil(|py| i.to_object(py)),
            ISize(i) => Python::with_gil(|py| i.to_object(py)),
            U8(i) => Python::with_gil(|py| i.to_object(py)),
            U16(i) => Python::with_gil(|py| i.to_object(py)),
            U32(i) => Python::with_gil(|py| i.to_object(py)),
            U64(i) => Python::with_gil(|py| i.to_object(py)),
            U128(i) => Python::with_gil(|py| i.to_object(py)),
            USize(i) => Python::with_gil(|py| i.to_object(py)),
            F32(f) => Python::with_gil(|py| f.to_object(py)),
            F64(f) => Python::with_gil(|py| f.to_object(py)),
            Char(c) => Python::with_gil(|py| c.to_object(py)),
            Bool(b) => Python::with_gil(|py| b.to_object(py)),
            Str(s) => Python::with_gil(|py| s.to_object(py)),
            String(s) => Python::with_gil(|py| s.to_object(py)),
            // RcString(s) => Python::with_gil(|py| s.to_object(py)),
          }
        }
      }
    }

    pub fn from_python_tuple(v: &PyAny, ty: &TupleType) -> PyResult<Tuple> {
      match ty {
        TupleType::Tuple(ts) => {
          let tup: &PyTuple = v.cast_as()?;
          if tup.len() == ts.len() {
            let elems = ts
              .iter()
              .enumerate()
              .map(|(i, t)| {
                let e = tup.get_item(i)?;
                from_python_tuple(e, t)
              })
              .collect::<PyResult<Box<_>>>()?;
            Ok(Tuple::Tuple(elems))
          } else {
            Err(PyIndexError::new_err("Invalid tuple size"))
          }
        }
        TupleType::Value(t) => match t {
          ValueType::I8 => Ok(Tuple::Value(Value::I8(v.extract()?))),
          ValueType::I16 => Ok(Tuple::Value(Value::I16(v.extract()?))),
          ValueType::I32 => Ok(Tuple::Value(Value::I32(v.extract()?))),
          ValueType::I64 => Ok(Tuple::Value(Value::I64(v.extract()?))),
          ValueType::I128 => Ok(Tuple::Value(Value::I128(v.extract()?))),
          ValueType::ISize => Ok(Tuple::Value(Value::ISize(v.extract()?))),
          ValueType::U8 => Ok(Tuple::Value(Value::U8(v.extract()?))),
          ValueType::U16 => Ok(Tuple::Value(Value::U16(v.extract()?))),
          ValueType::U32 => Ok(Tuple::Value(Value::U32(v.extract()?))),
          ValueType::U64 => Ok(Tuple::Value(Value::U64(v.extract()?))),
          ValueType::U128 => Ok(Tuple::Value(Value::U128(v.extract()?))),
          ValueType::USize => Ok(Tuple::Value(Value::USize(v.extract()?))),
          ValueType::F32 => Ok(Tuple::Value(Value::F32(v.extract()?))),
          ValueType::F64 => Ok(Tuple::Value(Value::F64(v.extract()?))),
          ValueType::Char => Ok(Tuple::Value(Value::Char(v.extract()?))),
          ValueType::Bool => Ok(Tuple::Value(Value::Bool(v.extract()?))),
          ValueType::Str => panic!("Static reference string cannot be used for Python binding"),
          ValueType::String => Ok(Tuple::Value(Value::String(v.extract()?))),
          // ValueType::RcString => Ok(Tuple::Value(Value::RcString(Rc::new(
          //   v.extract::<String>()?,
          // )))),
        },
      }
    }
  }
}
