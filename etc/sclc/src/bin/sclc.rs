use structopt::StructOpt;

use scallop_core::compiler;
use sclc::*;

fn main() {
  // Command line arguments
  let opt = options::Options::from_args();

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

  // Depending on the mode, create artifacts
  match opt.mode.as_str() {
    "executable" | "exec" => exec::create_executable(&opt, compile_opt, &ram),
    "pylib" => pylib::create_pylib(&opt, compile_opt, &ram).unwrap(),
    m => panic!("Unknown compilation mode --mode `{}`", m),
  };
}
