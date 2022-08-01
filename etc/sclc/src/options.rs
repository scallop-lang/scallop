use std::path::PathBuf;
use structopt::StructOpt;

use scallop_core::compiler;

#[derive(Debug, StructOpt)]
#[structopt(name = "sclc", about = "Scallop Compiler")]
pub struct Options {
  #[structopt(parse(from_os_str))]
  pub input: PathBuf,

  #[structopt(long)]
  pub debug_rs: bool,

  #[structopt(long, default_value = "executable")]
  pub mode: String,
}

impl From<&Options> for compiler::CompileOptions {
  fn from(_: &Options) -> Self {
    Self { ..Default::default() }
  }
}
