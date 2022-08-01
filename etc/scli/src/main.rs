use scallop_core::common::predicate_set::PredicateSet;
use scallop_core::utils::RcFamily;
use std::path::PathBuf;
use structopt::StructOpt;

use scallop_core::compiler;
use scallop_core::runtime::dynamic;
use scallop_core::runtime::monitor;
use scallop_core::runtime::provenance;

#[derive(Debug, StructOpt)]
#[structopt(name = "scli", about = "Scallop Interpreter")]
struct Options {
  #[structopt(parse(from_os_str))]
  input: PathBuf,

  #[structopt(short, long, default_value = "unit")]
  provenance: String,

  #[structopt(short = "k", long, default_value = "3")]
  top_k: usize,

  #[structopt(short = "q", long)]
  query: Option<String>,

  /// General debug option
  #[structopt(short, long)]
  debug: bool,

  /// Debug front ir
  #[structopt(long)]
  debug_front: bool,

  /// Debug back ir
  #[structopt(long)]
  debug_back: bool,

  /// Debug ram program
  #[structopt(long)]
  debug_ram: bool,

  /// Monitor tagging
  #[structopt(long)]
  debug_tag: bool,

  /// Output all relations (including hidden ones)
  #[structopt(long)]
  output_all: bool,

  /// Do not remove unused relations
  #[structopt(long)]
  do_not_remove_unused_relations: bool,
}

impl From<&Options> for compiler::CompileOptions {
  fn from(opt: &Options) -> Self {
    Self {
      debug: opt.debug,
      debug_front: opt.debug_front,
      debug_back: opt.debug_back,
      debug_ram: opt.debug_ram,
      do_not_remove_unused_relations: opt.do_not_remove_unused_relations,
      output_all: opt.output_all,
      report_front_errors: true,
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

  // Generate interpret options
  let interpret_options = dynamic::InterpretOptions {
    // We do not need anything to be returned
    return_relations: PredicateSet::None,

    // We want everything to be printed
    print_relations: if let Some(q) = &opt.query {
      PredicateSet::Some(vec![q.clone()])
    } else {
      PredicateSet::All
    },

    // Others options are set to default
    ..Default::default()
  };

  // Run the ram program
  if !opt.debug_tag {
    match opt.provenance.as_str() {
      "unit" => {
        let mut ctx = provenance::unit::UnitContext::default();
        dynamic::interpret_with_options(ram, &mut ctx, &interpret_options).expect("Runtime error");
      }
      "bool" => {
        let mut ctx = provenance::boolean::BooleanContext::default();
        dynamic::interpret_with_options(ram, &mut ctx, &interpret_options).expect("Runtime error");
      }
      "proofs" => {
        let mut ctx = provenance::proofs::ProofsContext::default();
        dynamic::interpret_with_options(ram, &mut ctx, &interpret_options).expect("Runtime error");
      }
      "minmaxprob" => {
        let mut ctx = provenance::min_max_prob::MinMaxProbContext::default();
        dynamic::interpret_with_options(ram, &mut ctx, &interpret_options).expect("Runtime error");
      }
      "addmultprob" => {
        let mut ctx = provenance::add_mult_prob::AddMultProbContext::default();
        dynamic::interpret_with_options(ram, &mut ctx, &interpret_options).expect("Runtime error");
      }
      "topkproofs" => {
        let mut ctx = provenance::top_k_proofs::TopKProofsContext::<RcFamily>::new(opt.top_k);
        dynamic::interpret_with_options(ram, &mut ctx, &interpret_options).expect("Runtime error");
      }
      "topbottomkclauses" => {
        let mut ctx = provenance::top_bottom_k_clauses::TopBottomKClausesContext::<RcFamily>::new(opt.top_k);
        dynamic::interpret_with_options(ram, &mut ctx, &interpret_options).expect("Runtime error");
      }
      _ => {
        println!("Unknown provenance semiring `{}`", opt.provenance);
        return;
      }
    };
  } else {
    let m = monitor::DebugTagsMonitor;

    match opt.provenance.as_str() {
      "unit" => {
        let mut ctx = provenance::unit::UnitContext::default();
        dynamic::interpret_with_options_and_monitor(ram, &mut ctx, &interpret_options, &m).expect("Runtime error");
      }
      "bool" => {
        let mut ctx = provenance::boolean::BooleanContext::default();
        dynamic::interpret_with_options_and_monitor(ram, &mut ctx, &interpret_options, &m).expect("Runtime error");
      }
      "proofs" => {
        let mut ctx = provenance::proofs::ProofsContext::default();
        dynamic::interpret_with_options_and_monitor(ram, &mut ctx, &interpret_options, &m).expect("Runtime error");
      }
      "minmaxprob" => {
        let mut ctx = provenance::min_max_prob::MinMaxProbContext::default();
        dynamic::interpret_with_options_and_monitor(ram, &mut ctx, &interpret_options, &m).expect("Runtime error");
      }
      "addmultprob" => {
        let mut ctx = provenance::add_mult_prob::AddMultProbContext::default();
        dynamic::interpret_with_options_and_monitor(ram, &mut ctx, &interpret_options, &m).expect("Runtime error");
      }
      "topkproofs" => {
        let mut ctx = provenance::top_k_proofs::TopKProofsContext::<RcFamily>::new(opt.top_k);
        dynamic::interpret_with_options_and_monitor(ram, &mut ctx, &interpret_options, &m).expect("Runtime error");
      }
      "topbottomkclauses" => {
        let mut ctx = provenance::top_bottom_k_clauses::TopBottomKClausesContext::<RcFamily>::new(opt.top_k);
        dynamic::interpret_with_options_and_monitor(ram, &mut ctx, &interpret_options, &m).expect("Runtime error");
      }
      _ => {
        println!("Unknown provenance semiring `{}`", opt.provenance);
        return;
      }
    };
  }
}
