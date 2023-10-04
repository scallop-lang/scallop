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
  let (parent_dir, tmp_dir) = generate_pylib_rust_project(&opt.input, compile_opt, ram);

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
    eprintln!("[Compile Error]");
    eprintln!("stdout: {}", std::str::from_utf8(&output.stdout).unwrap());
    eprintln!("{}", std::str::from_utf8(&output.stderr).unwrap());
    Ok(())
  }
}

pub fn generate_pylib_rust_project(
  input: &PathBuf,
  compile_opt: compiler::CompileOptions,
  ram: &compiler::ram::Program,
) -> (PathBuf, PathBuf) {
  // Get program name
  let program_name = input.file_prefix().unwrap().to_str().unwrap();

  // Rust source
  let file_content = generate_pylib_code(compile_opt, program_name, ram).to_string();

  // Create a folder
  let parent_dir = input.parent().unwrap();
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

  // Return
  (parent_dir.to_path_buf(), tmp_dir)
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
    use scallop_core::common::tensors::*;
    use scallop_core::common::tuple::*;
    use scallop_core::common::tuple_type::*;
    use scallop_core::common::value::*;
    use scallop_core::common::value_type::*;
    use scallop_core::runtime::provenance::*;
    use scallop_core::runtime::database::extensional::*;
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
    struct StaticContext<C: PythonProvenance> {
      prov_ctx: C,
      edb: ExtensionalDatabase<C>,
      output: Option<scallop_module::OutputRelations<C>>,
    }

    impl<C: PythonProvenance> StaticContext<C> {
      fn new(prov_ctx: C) -> Self {
        let edb = scallop_module::create_edb();
        Self { prov_ctx, edb, output: None }
      }

      fn add_py_facts(&mut self, r: &str, py_facts: &PyList) -> Result<(), BindingError> {
        if let Some(ty) = self.edb.type_of(r) {
          let facts = C::process_typed_py_facts(py_facts, &ty)?;

          // Add facts and disjunctions to relation
          self.edb.add_static_input_facts_without_type_check(r, facts).unwrap();

          // Return Ok
          Ok(())
        } else {
          Err(BindingError(format!("Unknown relation `{}`", r)))
        }
      }

      fn add_facts(&mut self, r: &str, facts: Vec<(Option<C::InputTag>, Tuple)>) -> Result<(), BindingError> {
        if let Some(ty) = self.edb.type_of(r) {
          // Add facts and disjunctions to relation
          self.edb.add_static_input_facts(r, facts).unwrap();

          // Return Ok
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

      fn py_relation(&self, r: &str) -> Result<Vec<Py<PyAny>>, BindingError> {
        self.relation(r).map(|c| c.into_iter().map(C::to_output_py_fact).collect())
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
        let err_str = format!("{}", err.0);
        let py_err_str: Py<PyAny> = Python::with_gil(|py| err_str.to_object(py));
        PyException::new_err(py_err_str)
      }
    }

    impl std::convert::From<PyErr> for BindingError {
      fn from(err: PyErr) -> Self {
        Self(format!("{}", err))
      }
    }
  }
}

fn generate_context_code() -> TokenStream {
  quote! {
    #[derive(Clone)]
    enum ContextEnum {
      Unit(StaticContext<unit::UnitProvenance>),
      MinMaxProb(StaticContext<min_max_prob::MinMaxProbProvenance>),
      AddMultProb(StaticContext<add_mult_prob::AddMultProbProvenance>),
      DiffMinMaxProb(StaticContext<diff_min_max_prob::DiffMinMaxProbProvenance<ExtTag, ArcFamily>>),
      DiffTopKProofs(StaticContext<diff_top_k_proofs::DiffTopKProofsProvenance<ExtTag, ArcFamily>>),
      DiffTopBottomKClauses(StaticContext<diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance<ExtTag, ArcFamily>>),
    }

    #[pyclass(unsendable, name = "StaticContext")]
    struct Context { ctx: ContextEnum }

    #[pymethods]
    impl Context {
      #[new]
      #[args(provenance = "\"unit\"", top_k = "None", wmc_with_disjunctions = "False")]
      fn new(provenance: &str, top_k: Option<usize>, wmc_with_disjunctions: bool) -> PyResult<Self> {
        let top_k = top_k.unwrap_or(3);
        match provenance {
          "unit" => Ok(Self { ctx: ContextEnum::Unit(StaticContext::new(unit::UnitProvenance::default())) }),
          "minmaxprob" => Ok(Self { ctx: ContextEnum::MinMaxProb(StaticContext::new(min_max_prob::MinMaxProbProvenance::default())) }),
          "addmultprob" => Ok(Self { ctx: ContextEnum::AddMultProb(StaticContext::new(add_mult_prob::AddMultProbProvenance::default())) }),
          "diffminmaxprob" => Ok(Self { ctx: ContextEnum::DiffMinMaxProb(StaticContext::new(diff_min_max_prob::DiffMinMaxProbProvenance::default())) }),
          "difftopkproofs" => Ok(Self { ctx: ContextEnum::DiffTopKProofs(StaticContext::new(diff_top_k_proofs::DiffTopKProofsProvenance::new(top_k, wmc_with_disjunctions))) }),
          "difftopbottomkclauses" => Ok(Self { ctx: ContextEnum::DiffTopBottomKClauses(StaticContext::new(diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance::new(top_k))) }),
          p => Err(PyErr::from(BindingError(format!("Unknown provenance `{}`", p.to_string())))),
        }
      }

      fn input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        match &self.ctx {
          ContextEnum::Unit(_) => None,
          ContextEnum::MinMaxProb(_) => None,
          ContextEnum::AddMultProb(_) => None,
          ContextEnum::DiffMinMaxProb(_) => None,
          ContextEnum::DiffTopKProofs(c) => Some(c.prov_ctx.input_tags().into_vec()),
          ContextEnum::DiffTopBottomKClauses(c) => Some(c.prov_ctx.input_tags().into_vec()),
        }
      }

      fn add_facts(&mut self, r: &str, facts: &PyList) -> PyResult<()> {
        let res = match &mut self.ctx {
          ContextEnum::Unit(c) => c.add_py_facts(r, facts),
          ContextEnum::MinMaxProb(c) => c.add_py_facts(r, facts),
          ContextEnum::AddMultProb(c) => c.add_py_facts(r, facts),
          ContextEnum::DiffMinMaxProb(c) => c.add_py_facts(r, facts),
          ContextEnum::DiffTopKProofs(c) => c.add_py_facts(r, facts),
          ContextEnum::DiffTopBottomKClauses(c) => c.add_py_facts(r, facts),
        };
        res.map_err(PyErr::from)
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

      fn run_batch_parallel(&self, inputs: HashMap<String, Vec<&PyList>>, outputs: Vec<&str>) -> PyResult<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>> {
        let res = match &self.ctx {
          ContextEnum::Unit(c) => run_batch_parallel(c, inputs, outputs),
          ContextEnum::MinMaxProb(c) => run_batch_parallel(c, inputs, outputs),
          ContextEnum::AddMultProb(c) => run_batch_parallel(c, inputs, outputs),
          ContextEnum::DiffMinMaxProb(c) => run_batch_parallel(c, inputs, outputs),
          ContextEnum::DiffTopKProofs(c) => run_batch_parallel(c, inputs, outputs),
          ContextEnum::DiffTopBottomKClauses(c) => run_batch_parallel(c, inputs, outputs),
        };
        res.map_err(PyErr::from)
      }

      fn run_batch_non_parallel(&self, inputs: HashMap<String, Vec<&PyList>>, outputs: Vec<&str>) -> PyResult<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>> {
        let res = match &self.ctx {
          ContextEnum::Unit(c) => run_batch_non_parallel(c, inputs, outputs),
          ContextEnum::MinMaxProb(c) => run_batch_non_parallel(c, inputs, outputs),
          ContextEnum::AddMultProb(c) => run_batch_non_parallel(c, inputs, outputs),
          ContextEnum::DiffMinMaxProb(c) => run_batch_non_parallel(c, inputs, outputs),
          ContextEnum::DiffTopKProofs(c) => run_batch_non_parallel(c, inputs, outputs),
          ContextEnum::DiffTopBottomKClauses(c) => run_batch_non_parallel(c, inputs, outputs),
        };
        res.map_err(PyErr::from)
      }

      fn relation(&self, r: &str) -> PyResult<Vec<Py<PyAny>>> {
        let res = match &self.ctx {
          ContextEnum::Unit(c) => c.py_relation(r),
          ContextEnum::MinMaxProb(c) => c.py_relation(r),
          ContextEnum::AddMultProb(c) => c.py_relation(r),
          ContextEnum::DiffMinMaxProb(c) => c.py_relation(r),
          ContextEnum::DiffTopKProofs(c) => c.py_relation(r),
          ContextEnum::DiffTopBottomKClauses(c) => c.py_relation(r),
        };
        res.map_err(PyErr::from)
      }
    }
  }
}

fn generate_helper_functions() -> TokenStream {
  quote! {
    fn run_batch_parallel<C>(
      c: &StaticContext<C>,
      inputs: HashMap<String, Vec<&PyList>>,
      outputs: Vec<&str>,
    ) -> Result<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>, BindingError>
    where
      C: PythonProvenance + std::marker::Sync,
      <C as Provenance>::InputTag: std::marker::Sync,
      <C as Provenance>::OutputTag: std::marker::Sync + std::marker::Send,
      <C as Provenance>::Tag: std::marker::Sync + std::marker::Send,
    {
      let batch_size = inputs.iter().next().unwrap().1.len();

      // Create contexts for parallel computing
      let prep_inputs = inputs
        .into_iter()
        .map(|(r, batch)| {
          let ty = c.edb.type_of(&r).ok_or(BindingError(format!("Unknown relation `{}`", r)))?;
          let prep_batch = batch.into_iter().map(|facts| C::process_typed_py_facts(facts, &ty)).collect::<Result<_, BindingError>>()?;
          Ok((r, prep_batch))
        })
        .collect::<Result<HashMap<String, Vec<Vec<(Option<C::InputTag>, Tuple)>>>, BindingError>>()?;

      // Actually execute the program in parallel
      let base_results = (0..batch_size)
        .into_par_iter()
        .map(|i| {
          let mut temp_ctx = c.clone();
          for (r, batch) in prep_inputs.iter() {
            temp_ctx.add_facts(r, batch[i].clone())?;
          }
          temp_ctx.run();
          let input_tags = C::get_input_tags(&temp_ctx.prov_ctx);
          let relations = outputs.iter().map(|o| temp_ctx.relation(o)).collect::<Result<Vec<_>, _>>()?;
          Ok((input_tags, relations))
        })
        .collect::<Result<Vec<_>, BindingError>>()?;

      // Recover python collection
      let results = base_results
        .into_iter()
        .map(|(input_tags, relations)| {
          (input_tags, relations.into_iter().map(|rs| rs.into_iter().map(C::to_output_py_fact).collect()).collect())
        })
        .collect();

      // Return
      Ok(results)
    }

    fn run_batch_non_parallel<C>(
      c: &StaticContext<C>,
      inputs: HashMap<String, Vec<&PyList>>,
      outputs: Vec<&str>,
    ) -> Result<Vec<(Option<Vec<Py<PyAny>>>, Vec<Vec<Py<PyAny>>>)>, BindingError>
    where
      C: PythonProvenance,
    {
      let batch_size = inputs.iter().next().unwrap().1.len();

      // Create contexts for parallel computing
      let prep_inputs = inputs
        .into_iter()
        .map(|(r, batch)| {
          let ty = c.edb.type_of(&r).ok_or(BindingError(format!("Unknown relation `{}`", r)))?;
          let prep_batch = batch.into_iter().map(|facts| C::process_typed_py_facts(facts, &ty)).collect::<Result<_, BindingError>>()?;
          Ok((r, prep_batch))
        })
        .collect::<Result<HashMap<String, Vec<Vec<(Option<C::InputTag>, Tuple)>>>, BindingError>>()?;

      // Actually execute the program in parallel
      let base_results = (0..batch_size)
        .into_iter()
        .map(|i| {
          let mut temp_ctx = c.clone();
          for (r, batch) in prep_inputs.iter() {
            temp_ctx.add_facts(r, batch[i].clone())?;
          }
          temp_ctx.run();
          let input_tags = C::get_input_tags(&temp_ctx.prov_ctx);
          let relations = outputs.iter().map(|o| temp_ctx.relation(o)).collect::<Result<Vec<_>, _>>()?;
          Ok((input_tags, relations))
        })
        .collect::<Result<Vec<_>, BindingError>>()?;

      // Recover python collection
      let results = base_results
        .into_iter()
        .map(|(input_tags, relations)| {
          (input_tags, relations.into_iter().map(|rs| rs.into_iter().map(C::to_output_py_fact).collect()).collect())
        })
        .collect();

      // Return
      Ok(results)
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
            Symbol(_) => unimplemented!(),
            SymbolString(_) => unimplemented!(),
            DateTime(_) => unimplemented!(),
            Duration(_) => unimplemented!(),
            Entity(i) => Python::with_gil(|py| i.to_object(py)),
            Tensor(_) => unimplemented!(),
            TensorValue(_) => unimplemented!(),
          }
        }
      }
    }

    fn from_python_tuple(v: &PyAny, ty: &TupleType) -> PyResult<Tuple> {
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
          ValueType::Symbol => unimplemented!(),
          ValueType::DateTime => unimplemented!(),
          ValueType::Duration => unimplemented!(),
          ValueType::Entity => Ok(Tuple::Value(Value::Entity(v.extract()?))),
          ValueType::Tensor => unimplemented!(),
        },
      }
    }

    #[derive(Clone)]
    pub struct ExtTag {
      pub tag: Py<PyAny>,
    }

    impl FromTensor for ExtTag {
      #[allow(unused)]
      #[cfg(not(feature = "torch-tensor"))]
      fn from_tensor(tensor: Tensor) -> Option<Self> {
        None
      }

      #[cfg(feature = "torch-tensor")]
      fn from_tensor(tensor: Tensor) -> Option<Self> {
        use super::torch::*;
        Python::with_gil(|py| {
          let py_tensor = PyTensor(tensor.tensor);
          let py_obj: Py<PyAny> = py_tensor.into_py(py);
          let ext_tag: ExtTag = py_obj.into();
          Some(ext_tag)
        })
      }
    }

    impl From<Py<PyAny>> for ExtTag {
      fn from(tag: Py<PyAny>) -> Self {
        Self { tag }
      }
    }

    impl From<&PyAny> for ExtTag {
      fn from(tag: &PyAny) -> Self {
        Self { tag: tag.into() }
      }
    }

    impl Into<Py<PyAny>> for ExtTag {
      fn into(self) -> Py<PyAny> {
        self.tag
      }
    }

    pub trait ExtTagVec {
      fn into_vec(self) -> Vec<Py<PyAny>>;
    }

    impl ExtTagVec for Vec<ExtTag> {
      fn into_vec(self) -> Vec<Py<PyAny>> {
        self.into_iter().map(|v| v.tag).collect()
      }
    }

    pub trait ExtTagOption {
      fn into_option(self) -> Option<Py<PyAny>>;
    }

    impl ExtTagOption for Option<ExtTag> {
      fn into_option(self) -> Option<Py<PyAny>> {
        self.map(|v| v.tag)
      }
    }

    impl ExtTagOption for Option<&ExtTag> {
      fn into_option(self) -> Option<Py<PyAny>> {
        self.map(|v| v.tag.clone())
      }
    }

    trait PythonProvenance: Provenance {
      /// Process a list of python facts while the tuples are typed with `tuple_type`
      fn process_typed_py_facts(facts: &PyList, tuple_type: &TupleType) -> Result<Vec<(Option<Self::InputTag>, Tuple)>, BindingError> {
        let facts: Vec<&PyAny> = facts.extract()?;
        facts
          .into_iter()
          .map(|fact| {
            let (maybe_py_tag, py_tup) = Self::split_py_fact(fact)?;
            let tag = Self::process_optional_py_tag(maybe_py_tag)?;
            let tup = from_python_tuple(py_tup, tuple_type).map_err(|_| BindingError(format!("Cannot parse tuple `{}` with type `{}`", py_tup, tuple_type)))?;
            Ok((tag, tup))
          })
          .collect::<Result<Vec<_>, _>>()
      }

      /// Split a python object into (Option<PyTag>, PyTup) pair
      fn split_py_fact(fact: &PyAny) -> Result<(Option<&PyAny>, &PyAny), BindingError> {
        let (py_tag, py_tup) = fact.extract()?;
        Ok((py_tag, py_tup))
      }

      /// Convert a python object into an optional input tag for this provenance context
      fn process_optional_py_tag(maybe_py_tag: Option<&PyAny>) -> Result<Option<Self::InputTag>, BindingError> {
        if let Some(py_tag) = maybe_py_tag {
          Ok(Self::process_py_tag(py_tag)?)
        } else {
          Ok(None)
        }
      }

      /// Convert a python object into an optional input tag for this provenance context
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError>;

      /// Convert an output tag into a python object
      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny>;

      /// Convert a fact into a python fact
      fn to_output_py_fact((tag, tup): (Self::OutputTag, Tuple)) -> Py<PyAny> {
        Python::with_gil(|py| (Self::to_output_py_tag(&tag), to_python_tuple(&tup)).to_object(py))
      }

      /// Get input tags
      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        None
      }
    }

    impl PythonProvenance for unit::UnitProvenance {
      fn split_py_fact(fact: &PyAny) -> Result<(Option<&PyAny>, &PyAny), BindingError> {
        Ok((None, fact))
      }

      fn process_py_tag(_: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        Ok(None)
      }

      fn to_output_py_tag(_: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| ().to_object(py))
      }

      fn to_output_py_fact((_, tup): (Self::OutputTag, Tuple)) -> Py<PyAny> {
        to_python_tuple(&tup)
      }
    }

    impl PythonProvenance for proofs::ProofsProvenance<ArcFamily> {
      fn process_py_tag(disj_id: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let disj_id: usize = disj_id.extract()?;
        Ok(Some(Exclusion::Exclusive(disj_id)))
      }

      fn to_output_py_tag(proofs: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| {
          proofs
            .proofs
            .iter()
            .map(|proof| proof.facts.iter().collect::<Vec<_>>())
            .collect::<Vec<_>>()
            .to_object(py)
        })
      }
    }

    impl PythonProvenance for min_max_prob::MinMaxProbProvenance {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        tag.extract().map(Some).map_err(BindingError::from)
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| tag.to_object(py))
      }
    }

    impl PythonProvenance for add_mult_prob::AddMultProbProvenance {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        tag.extract().map(Some).map_err(BindingError::from)
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| tag.to_object(py))
      }
    }

    impl PythonProvenance for top_k_proofs::TopKProofsProvenance<ArcFamily> {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let (prob, disj_id): (Option<f64>, Option<usize>) = tag.extract().map_err(BindingError::from)?;
        if let Some(prob) = prob {
          Ok(Some((prob, disj_id).into()))
        } else {
          Ok(None)
        }
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| tag.to_object(py))
      }
    }

    impl PythonProvenance for top_bottom_k_clauses::TopBottomKClausesProvenance<ArcFamily> {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let (prob, disj_id): (Option<f64>, Option<usize>) = tag.extract().map_err(BindingError::from)?;
        if let Some(prob) = prob {
          Ok(Some((prob, disj_id).into()))
        } else {
          Ok(None)
        }
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| tag.to_object(py))
      }
    }

    impl PythonProvenance for diff_min_max_prob::DiffMinMaxProbProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let prob: f64 = tag.extract().map_err(BindingError::from)?;
        let tag: ExtTag = tag.into();
        Ok(Some((prob, Some(tag)).into()))
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        match tag.2 {
          1 => Python::with_gil(|py| (1, tag.3.as_ref().into_option().unwrap()).to_object(py)),
          0 => Python::with_gil(|py| (0, tag.0).to_object(py)),
          _ => Python::with_gil(|py| (-1, tag.3.as_ref().into_option().unwrap()).to_object(py)),
        }
      }
    }

    impl PythonProvenance for diff_add_mult_prob::DiffAddMultProbProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let prob: f64 = tag.extract().map_err(BindingError::from)?;
        let tag: ExtTag = tag.into();
        Ok(Some((prob, Some(tag)).into()))
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
      }

      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        Some(self.input_tags().into_vec())
      }
    }

    impl PythonProvenance for diff_nand_mult_prob::DiffNandMultProbProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let prob: f64 = tag.extract().map_err(BindingError::from)?;
        let tag: ExtTag = tag.into();
        Ok(Some((prob, Some(tag)).into()))
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
      }

      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        Some(self.input_tags().into_vec())
      }
    }

    impl PythonProvenance for diff_max_mult_prob::DiffMaxMultProbProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let prob: f64 = tag.extract().map_err(BindingError::from)?;
        let tag: ExtTag = tag.into();
        Ok(Some((prob, Some(tag)).into()))
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
      }

      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        Some(self.input_tags().into_vec())
      }
    }

    impl PythonProvenance for diff_nand_min_prob::DiffNandMinProbProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let prob: f64 = tag.extract()?;
        let tag: ExtTag = tag.into();
        Ok(Some((prob, Some(tag)).into()))
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
      }

      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        Some(self.input_tags().into_vec())
      }
    }

    impl PythonProvenance for diff_sample_k_proofs::DiffSampleKProofsProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let tag_disj_id: (&PyAny, Option<usize>) = tag.extract()?;
        if let Some(prob) = tag_disj_id.0.extract()? {
          let tag: ExtTag = tag_disj_id.0.into();
          Ok(Some((prob, tag, tag_disj_id.1).into()))
        } else {
          Ok(None)
        }
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
      }

      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        Some(self.input_tags().into_vec())
      }
    }

    impl PythonProvenance for diff_top_k_proofs::DiffTopKProofsProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let tag_disj_id: (&PyAny, Option<usize>) = tag.extract()?;
        if let Some(prob) = tag_disj_id.0.extract()? {
          let tag: ExtTag = tag_disj_id.0.into();
          Ok(Some((prob, tag, tag_disj_id.1).into()))
        } else {
          Ok(None)
        }
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
      }

      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        Some(self.input_tags().into_vec())
      }
    }

    impl PythonProvenance for diff_top_bottom_k_clauses::DiffTopBottomKClausesProvenance<ExtTag, ArcFamily> {
      fn process_py_tag(tag: &PyAny) -> Result<Option<Self::InputTag>, BindingError> {
        let tag_disj_id: (&PyAny, Option<usize>) = tag.extract()?;
        if let Some(prob) = tag_disj_id.0.extract()? {
          let tag: ExtTag = tag_disj_id.0.into();
          Ok(Some((prob, tag, tag_disj_id.1).into()))
        } else {
          Ok(None)
        }
      }

      fn to_output_py_tag(tag: &Self::OutputTag) -> Py<PyAny> {
        Python::with_gil(|py| (tag.0, tag.1.clone()).to_object(py))
      }

      fn get_input_tags(&self) -> Option<Vec<Py<PyAny>>> {
        Some(self.input_tags().into_vec())
      }
    }
  }
}
