#![feature(path_file_prefix)]

use proc_macro2::TokenStream;
use quote::quote;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::process::Command;
use structopt::StructOpt;

use scallop_core::compiler;

#[derive(Debug, StructOpt)]
#[structopt(name = "sclc", about = "Scallop Compiler")]
struct Options {
  #[structopt(parse(from_os_str))]
  input: PathBuf,

  #[structopt(long)]
  debug_rs: bool,

  #[structopt(long, default_value = "executable")]
  mode: String,
}

impl From<&Options> for compiler::CompileOptions {
  fn from(_: &Options) -> Self {
    Self {
      ..Default::default()
    }
  }
}

fn main() {
  // Command line arguments
  let opt = Options::from_args();

  // Compile
  let compile_opt = compiler::CompileOptions::from(&opt);
  let ram = match compiler::compile_file_to_ram_with_options(&opt.input, &compile_opt) {
    Ok(ram) => ram,
    Err(errs) => {
      for err in errs {
        println!("{}", err);
      }
      return;
    }
  };

  // Turn the ram module into a sequence of rust tokens
  let module = ram.to_rs_module(&compile_opt);

  // Print the module string if debugging
  if opt.debug_rs {
    println!("{}", module);
  }

  // Depending on the mode, create artifacts
  match opt.mode.as_str() {
    "executable" => create_executable(&opt, &ram, module),
    m => panic!("Unknown compilation mode --mode `{}`", m),
  };
}

fn create_executable(opt: &Options, ram: &compiler::ram::Program, module: TokenStream) {
  let output_code = ram.to_rs_output("res");

  // Generate full executable code
  let full_executable = quote! {
    mod scallop_module { #module }
    fn main() {
      use scallop_core::runtime::provenance;
      let mut ctx = provenance::unit::UnitContext::default();
      let res = scallop_module::run(&mut ctx);
      #output_code
    }
  };

  // Rust source
  let file_content = full_executable.to_string();

  // Create a folder
  let program_name = opt.input.file_prefix().unwrap().to_str().unwrap();
  let parent_dir = opt.input.parent().unwrap();
  let tmp_dir = parent_dir.join(format!(".{}.sclcmpl", program_name));

  // Create a temporary directory holding the cargo project
  fs::create_dir_all(&tmp_dir).unwrap();
  fs::create_dir_all(&tmp_dir.join("src")).unwrap();

  // Create a Cargo.toml file
  let mut cargo_toml_file = File::create(tmp_dir.join("Cargo.toml")).unwrap();
  cargo_toml_file
    .write_all(
      format!(
        r#"
    [package]
    name = "{}"
    version = "1.0.0"
    edition = "2018"
    [dependencies]
    scallop-core = {{ path = "/Users/liby99/Local/Projects/scallop-v2/core" }}
    structopt = "0.3"
    proc-macro2 = "1.0"
    quote = "1.0"
    [workspace]
  "#,
        program_name
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
    fs::copy(
      tmp_dir.join(format!("target/release/{}", program_name)),
      parent_dir.join(program_name),
    )
    .unwrap();
    // fs::remove_dir_all(tmp_dir).unwrap();
  } else {
    println!("{}", std::str::from_utf8(&output.stderr).unwrap());
  }
}
