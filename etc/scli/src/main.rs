use std::path::PathBuf;
use structopt::StructOpt;

use scallop_core::common::constants::*;
use scallop_core::common::predicate_set::*;
use scallop_core::compiler;
use scallop_core::integrate;
use scallop_core::runtime::dynamic;
use scallop_core::runtime::env;
use scallop_core::runtime::monitor;
use scallop_core::runtime::provenance;
use scallop_core::utils::*;

#[derive(Debug, StructOpt)]
#[structopt(name = "scli", about = "Scallop Interpreter")]
struct Options {
  #[structopt(parse(from_os_str))]
  input: PathBuf,

  #[structopt(short, long, default_value = "unit")]
  provenance: String,

  #[structopt(short = "k", long, default_value = "3")]
  top_k: usize,

  #[structopt(long)]
  wmc_with_disjunctions: bool,

  #[structopt(short = "q", long)]
  query: Option<String>,

  #[structopt(long)]
  iter_limit: Option<usize>,

  #[structopt(long)]
  seed: Option<u64>,

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

  /// Monitor runtime
  #[structopt(long)]
  debug_runtime: bool,

  /// Output all relations (including hidden ones)
  #[structopt(long)]
  output_all: bool,

  /// Random seed
  #[structopt(long)]
  no_early_discard: bool,

  /// Do not remove unused relations
  #[structopt(long)]
  do_not_remove_unused_relations: bool,
}

struct MonitorOptions {
  pub debug_tag: bool,
  pub debug_runtime: bool,
}

impl From<&Options> for MonitorOptions {
  fn from(opt: &Options) -> Self {
    Self {
      debug_tag: opt.debug_tag,
      debug_runtime: opt.debug_runtime,
    }
  }
}

impl MonitorOptions {
  fn needs_monitor(&self) -> bool {
    self.debug_tag || self.debug_runtime
  }

  fn build<Prov: provenance::Provenance>(&self) -> monitor::DynamicMonitors<Prov> {
    let mut monitor = monitor::DynamicMonitors::new();
    if self.debug_tag {
      monitor.add(monitor::DebugTagsMonitor);
    }
    if self.debug_runtime {
      monitor.add(monitor::DebugRuntimeMonitor);
    }
    monitor
  }
}

fn main() -> Result<(), String> {
  // Command line arguments
  let opt = Options::from_args();

  // Integration options
  let integrate_opt = integrate::IntegrateOptions {
    compiler_options: compiler::CompileOptions {
      debug: opt.debug,
      debug_front: opt.debug_front,
      debug_back: opt.debug_back,
      debug_ram: opt.debug_ram,
      do_not_remove_unused_relations: opt.do_not_remove_unused_relations,
      output_all: opt.output_all,
      ..Default::default()
    },
    execution_options: dynamic::ExecutionOptions {
      type_check: false,
      incremental_maintain: false,
      retain_internal_when_recover: false,
    },
    runtime_environment_options: env::RuntimeEnvironmentOptions {
      random_seed: opt.seed.unwrap_or(DEFAULT_RANDOM_SEED),
      early_discard: !opt.no_early_discard,
      iter_limit: opt.iter_limit,
    },
  };

  // Create a set of output predicates
  let predicate_set = if let Some(q) = &opt.query {
    PredicateSet::Some(vec![q.clone()])
  } else {
    PredicateSet::All
  };

  // Monitor options
  let monitor_options = MonitorOptions::from(&opt);

  // Use the specified provenance
  match opt.provenance.as_str() {
    "unit" => {
      let ctx = provenance::unit::UnitProvenance::default();
      interpret(ctx, &opt.input, integrate_opt, predicate_set, monitor_options)
    }
    "bool" => {
      let ctx = provenance::boolean::BooleanProvenance::default();
      interpret(ctx, &opt.input, integrate_opt, predicate_set, monitor_options)
    }
    "proofs" => {
      let ctx = provenance::proofs::ProofsProvenance::<RcFamily>::default();
      interpret(ctx, &opt.input, integrate_opt, predicate_set, monitor_options)
    }
    "minmaxprob" => {
      let ctx = provenance::min_max_prob::MinMaxProbProvenance::default();
      interpret(ctx, &opt.input, integrate_opt, predicate_set, monitor_options)
    }
    "addmultprob" => {
      let ctx = provenance::add_mult_prob::AddMultProbProvenance::default();
      interpret(ctx, &opt.input, integrate_opt, predicate_set, monitor_options)
    }
    "topkproofs" => {
      let ctx = provenance::top_k_proofs::TopKProofsProvenance::<RcFamily>::new(opt.top_k, opt.wmc_with_disjunctions);
      interpret(ctx, &opt.input, integrate_opt, predicate_set, monitor_options)
    }
    "topbottomkclauses" => {
      let ctx = provenance::top_bottom_k_clauses::TopBottomKClausesProvenance::<RcFamily>::new(
        opt.top_k,
        opt.wmc_with_disjunctions,
      );
      interpret(ctx, &opt.input, integrate_opt, predicate_set, monitor_options)
    }
    _ => Err(format!("Unknown provenance semiring `{}`", opt.provenance)),
  }
}

fn interpret<Prov: provenance::Provenance>(
  prov: Prov,
  file_name: &PathBuf,
  opt: integrate::IntegrateOptions,
  predicate_set: PredicateSet,
  monitor_options: MonitorOptions,
) -> Result<(), String> {
  let mut interpret_ctx =
    match integrate::InterpretContext::<_, RcFamily>::new_from_file_with_options(file_name, prov, opt) {
      Ok(ctx) => ctx,
      Err(err) => {
        eprintln!("{}", err);
        return Err(err.kind().to_string());
      }
    };

  // Check if we have any specified monitors, and run the program
  if !monitor_options.needs_monitor() {
    // If not, directly run without monitor
    interpret_ctx.run().expect("Runtime Error");
  } else {
    // And then run the context with monitor
    let monitor = monitor_options.build();
    interpret_ctx.run_with_monitor(&monitor).expect("Runtime Error");
  }

  // Get the resulting IDB, and print them
  let idb = interpret_ctx.idb();
  for (predicate, relation) in idb {
    if predicate_set.contains(&predicate) {
      println!("{}: {}", predicate, relation);
    }
  }

  // Ok
  Ok(())
}
