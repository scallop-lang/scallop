use std::path::PathBuf;

use super::*;

pub fn compile_source_to_ram<S: front::Source>(source: S) -> Result<ram::Program, CompileErrors> {
  compile_source_to_ram_with_options(source, &CompileOptions::default())
}

pub fn compile_source_to_ram_with_options<S: front::Source>(
  source: S,
  options: &CompileOptions,
) -> Result<ram::Program, CompileErrors> {
  // Construct the compilation context
  let mut front_context = front::FrontContext::new();
  match front_context.compile_source(source) {
    Ok(_) => {}
    Err(error_ctx) => {
      if options.report_front_errors {
        error_ctx.report_errors();
      }
      return Err(vec![CompileError::Front(error_ctx)]);
    }
  }

  // Debug
  if options.debug || options.debug_front {
    println!("======== Front Program ========");
    front_context.dump_ir();
    println!("");
  }

  // Construct back ir
  let mut back_ir = front_context.to_back_program();
  if let Err(e) = back_ir.apply_optimizations(&options) {
    return Err(vec![CompileError::Back(e)]);
  }

  // Debug
  if options.debug || options.debug_back {
    println!("======== Back Program ========");
    println!("{}", back_ir);
    println!("");
  }

  // Construct ram ir
  let mut ram = match back_ir.to_ram_program(&options) {
    Ok(ram) => ram,
    Err(e) => {
      return Err(vec![CompileError::Back(e)]);
    }
  };

  // Apply ram optimizations
  if !options.do_not_optimize_ram {
    ram::optimizations::optimize_ram(&mut ram);
  }

  // Debug
  if options.debug || options.debug_ram {
    println!("======== RAM Program ========");
    println!("{}", ram);
  }

  // Success!
  Ok(ram)
}

pub fn compile_string_to_ram(string: String) -> Result<ram::Program, CompileErrors> {
  compile_string_to_ram_with_options(string, &CompileOptions::default())
}

pub fn compile_string_to_ram_with_options(
  string: String,
  options: &CompileOptions,
) -> Result<ram::Program, CompileErrors> {
  let source = front::StringSource::new(string);
  compile_source_to_ram_with_options(source, options)
}

pub fn compile_file_to_ram(file_name: &PathBuf) -> Result<ram::Program, CompileErrors> {
  compile_file_to_ram_with_options(file_name, &CompileOptions::default())
}

pub fn compile_file_to_ram_with_options(
  file_name: &PathBuf,
  options: &CompileOptions,
) -> Result<ram::Program, CompileErrors> {
  // Construct the source
  let source = match front::FileSource::new(file_name) {
    Ok(source) => source,
    Err(err) => {
      return Err(vec![CompileError::Front(front::FrontCompileError::singleton(err))]);
    }
  };

  // Compile
  compile_source_to_ram_with_options(source, options)
}
