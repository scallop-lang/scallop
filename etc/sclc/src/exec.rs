use proc_macro2::TokenStream;
use quote::quote;
use std::env;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::process::Command;

use scallop_core::compiler;

use super::options::*;

pub fn create_executable(opt: &Options, compile_opt: compiler::CompileOptions, ram: &compiler::ram::Program) {
  // Turn the ram module into a sequence of rust tokens
  let module = ram.to_rs_module(&compile_opt);

  // Print the module string if debugging
  if opt.debug_rs {
    println!("{}", module);
  }

  // Get the name of the program
  let program_name = opt.input.file_prefix().unwrap().to_str().unwrap();

  // Generate command line options
  let cmd_line_opt_struct = cmd_line_option_struct(opt, program_name);

  // Generate execution code
  let output_code = ram.to_rs_output("res");
  let run_fn = run_function(output_code);

  // Generate main body
  let main_fn = main_body(opt);

  // Generate full executable code
  let full_executable = quote! {
    use structopt::StructOpt;
    use scallop_core::runtime::provenance::*;
    use scallop_core::utils::*;
    mod scallop_module { #module }
    #cmd_line_opt_struct
    #run_fn
    fn main() {
      #main_fn
    }
  };

  // Rust source
  let file_content = full_executable.to_string();

  // Create a folder
  let parent_dir = opt.input.parent().unwrap();
  let tmp_dir = parent_dir.join(format!(".{}.exec.sclcmpl", program_name));
  let scallop_source_dir = env::var("SCALLOPDIR").expect(
    "Please set envrionment variable `SCALLOPDIR` to be the root of Scallop source directory before using `sclc`.",
  );

  // Create a temporary directory holding the cargo project
  fs::create_dir_all(&tmp_dir).unwrap();
  fs::create_dir_all(&tmp_dir.join("src")).unwrap();

  // Create a Cargo.toml file
  let mut cargo_toml_file = File::create(tmp_dir.join("Cargo.toml")).unwrap();
  cargo_toml_file
    .write_all(
      format!(
        r#"[package]
name = "{}"
version = "1.0.0"
edition = "2018"
[dependencies]
scallop-core = {{ path = "{}/core" }}
structopt = "0.3"
[workspace]
"#,
        program_name, scallop_source_dir,
      )
      .as_bytes(),
    )
    .unwrap();

  // Create a main.rs file
  let mut main_rs_file = File::create(tmp_dir.join("src/main.rs")).unwrap();
  main_rs_file.write_all(file_content.as_bytes()).unwrap();

  // Compile the file: create command
  let mut cmd = Command::new("cargo");

  // Add arguments
  cmd.current_dir(&tmp_dir).arg("build").arg("--release");

  // Run the command
  let output = cmd.output().unwrap();
  if output.status.success() {
    // If we want to copy executable
    if !opt.do_not_copy_artifact {
      fs::copy(
        tmp_dir.join(format!("target/release/{}", program_name)),
        parent_dir.join(program_name),
      )
      .unwrap();
    }

    // If we do not want to keep the temporary directory
    if opt.do_not_keep_temporary_directory {
      fs::remove_dir_all(tmp_dir).unwrap();
    }
  } else {
    println!("{}", std::str::from_utf8(&output.stderr).unwrap());
  }
}

fn cmd_line_option_struct(opt: &Options, program_name: &str) -> TokenStream {
  if let Some(_) = opt.provenance {
    quote! {}
  } else {
    quote! {
      #[derive(Debug, StructOpt)]
      #[structopt(name = #program_name)]
      struct Options {
        #[structopt(short, long, default_value = "unit")]
        provenance: String,
        #[structopt(long, default_value = "3")]
        top_k: usize,
      }
    }
  }
}

fn run_function(output_code: TokenStream) -> TokenStream {
  quote! {
    fn run<C: Provenance>(mut ctx: C) {
      let res = scallop_module::run(&mut ctx);
      #output_code
    }
  }
}

fn main_body(opt: &Options) -> TokenStream {
  if let Some(p) = &opt.provenance {
    let top_k = opt.top_k;
    match p.as_str() {
      "unit" => quote! { run(unit::UnitProvenance::default()); },
      "bool" => quote! { run(boolean::BooleanContext::default()); },
      "minmaxprob" => quote! { run(min_max_prob::MinMaxProbContext::default()); },
      "addmultprob" => quote! { run(add_mult_prob::AddMultProbContext::default()); },
      "topkproofs" => quote! { run(top_k_proofs::TopKProofsContext::<RcFamily>::new(#top_k)); },
      "samplekproofs" => quote! { run(sample_k_proofs::SampleKProofsContext::new(#top_k)); },
      "topbottomkclauses" => quote! { run(top_bottom_k_clauses::TopBottomKClausesContext::<RcFamily>::new(#top_k)); },
      p => panic!("Unknown provenance `{}`. Aborting", p),
    }
  } else {
    quote! {
      let opt = Options::from_args();
      match opt.provenance.as_str() {
        "unit" => run(unit::UnitProvenance::default()),
        "bool" => run(unit::UnitProvenance::default()),
        "minmaxprob" => run(min_max_prob::MinMaxProbContext::default()),
        "addmultprob" => run(add_mult_prob::AddMultProbContext::default()),
        "topkproofs" => run(top_k_proofs::TopKProofsContext::<RcFamily>::new(opt.top_k)),
        "samplekproofs" => run(sample_k_proofs::SampleKProofsContext::new(opt.top_k)),
        "topbottomkclauses" => run(top_bottom_k_clauses::TopBottomKClausesContext::<RcFamily>::new(opt.top_k)),
        p => println!("Unknown provenance `{}`. Aborting", p),
      }
    }
  }
}
