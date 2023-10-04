use std::path::PathBuf;
use structopt::StructOpt;

use scallop_core::compiler;

#[derive(Debug, Default, StructOpt)]
#[structopt(name = "sclc", about = "Scallop Compiler")]
pub struct Options {
  #[structopt(parse(from_os_str))]
  pub input: PathBuf,

  #[structopt(long)]
  pub debug_rs: bool,

  #[structopt(long, default_value = "executable")]
  pub mode: String,

  #[structopt(long)]
  pub do_not_copy_artifact: bool,

  #[structopt(long)]
  pub do_not_keep_temporary_directory: bool,

  #[structopt(long)]
  pub dump_rs: bool,

  #[structopt(long)]
  pub provenance: Option<String>,

  #[structopt(short = "k", long, default_value = "3")]
  pub top_k: usize,

  #[structopt(long)]
  pub wmc_with_disjunctions: bool,
}

impl From<&Options> for compiler::CompileOptions {
  fn from(_: &Options) -> Self {
    Self { ..Default::default() }
  }
}
