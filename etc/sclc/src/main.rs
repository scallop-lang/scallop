#![feature(path_file_prefix)]

mod exec;
mod options;
mod py;

use structopt::StructOpt;

use scallop_core::compiler;

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

  // Turn the ram module into a sequence of rust tokens
  let module = ram.to_rs_module(&compile_opt);

  // Print the module string if debugging
  if opt.debug_rs {
    println!("{}", module);
  }

  // Depending on the mode, create artifacts
  match opt.mode.as_str() {
    "executable" => exec::create_executable(&opt, &ram, module),
    m => panic!("Unknown compilation mode --mode `{}`", m),
  };
}
