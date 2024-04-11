use std::fs::File;
use std::io::prelude::*;
use std::process::Command;

use scallop_core::compiler::CompileOptions;

use scallop_core::compiler;
use sclc_core::exec;

pub fn check_compile_exec_from_program_string(program_name: &str, program_string: &str) {
  let tmp_dir = tempfile::tempdir()
    .expect("Unable to create temporary directory")
    .into_path();
  let scl_file_dir = tmp_dir.join(format!("{}.scl", program_name));

  // Write into file_dir
  let mut file = File::create(&scl_file_dir).expect("Unable to create file in temporary directory");
  file
    .write_all(program_string.as_bytes())
    .expect("Unable to write to scallop file in temporary directory");

  // Compile the file into scallop exec project
  let opt = CompileOptions::default();
  let ram = match compiler::compile_file_to_ram_with_options(&scl_file_dir, &opt) {
    Ok(ram) => ram,
    Err(errs) => {
      for err in errs {
        eprintln!("{}", err);
      }
      return;
    }
  };
  let sclc_opt = sclc_core::options::Options {
    input: scl_file_dir.clone(),
    ..Default::default()
  };
  let (_, proj_dir) = exec::generate_exec_rust_project(&sclc_opt, opt, &ram);

  // Create command for cargo check on proj dir
  let mut cmd = Command::new("cargo");
  cmd.current_dir(&proj_dir).arg("check");

  // Run the command
  let output = cmd.output().unwrap();
  if !output.status.success() {
    panic!("Cargo check failure: {}", std::str::from_utf8(&output.stderr).unwrap());
  }
}
