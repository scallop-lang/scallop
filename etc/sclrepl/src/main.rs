use linefeed::{Interface, ReadResult};
use scallop_core::utils::RcFamily;
use structopt::StructOpt;

use scallop_core::common;
use scallop_core::compiler;
use scallop_core::runtime;

#[derive(Debug, StructOpt)]
#[structopt(name = "sclrepl", about = "Scallop Interactive REPL")]
struct Options {
  #[structopt(short, long, default_value = "unit")]
  provenance: String,

  #[structopt(short, long)]
  debug: bool,

  #[structopt(long)]
  debug_front: bool,

  #[structopt(long)]
  debug_back: bool,

  #[structopt(long)]
  debug_ram: bool,
}

impl From<&Options> for compiler::CompileOptions {
  fn from(opt: &Options) -> Self {
    Self {
      debug: opt.debug,
      debug_front: opt.debug_front,
      debug_back: opt.debug_back,
      debug_ram: opt.debug_ram,
      report_front_errors: true,
      ..Default::default()
    }
  }
}

fn main() -> std::io::Result<()> {
  // Command line arguments
  let cmd_args = Options::from_args();

  // Initialize provenance context and run
  match cmd_args.provenance.as_str() {
    "unit" => {
      let ctx = runtime::provenance::unit::UnitProvenance::default();
      run(cmd_args, ctx)
    }
    "bool" => {
      let ctx = runtime::provenance::boolean::BooleanProvenance::default();
      run(cmd_args, ctx)
    }
    "minmaxprob" => {
      let ctx = runtime::provenance::min_max_prob::MinMaxProbProvenance::default();
      run(cmd_args, ctx)
    }
    "topkproofs" => {
      let ctx = runtime::provenance::top_k_proofs::TopKProofsProvenance::<RcFamily>::default();
      run(cmd_args, ctx)
    }
    _ => {
      println!("Unknown provenance semiring `{}`", cmd_args.provenance);
      Ok(())
    }
  }
}

fn run<C>(cmd_args: Options, ctx: C) -> std::io::Result<()>
where
  C: runtime::provenance::Provenance,
{
  // Interactive REPL
  let reader = Interface::new("scl")?;
  reader.set_prompt("scl> ")?;

  // Compile context
  let options = compiler::CompileOptions::from(&cmd_args);
  let mut front_context = compiler::front::FrontContext::new();
  let runtime_env = runtime::env::RuntimeEnvironment::default();
  let mut exec_context =
    runtime::dynamic::DynamicExecutionContext::<_, RcFamily>::new_with_options(runtime::dynamic::ExecutionOptions {
      incremental_maintain: true,
      retain_internal_when_recover: true,
      ..Default::default()
    });

  // Main Loop
  let mut repl_id = 0;
  while let ReadResult::Input(input) = reader.read_line()? {
    reader.add_history(input.clone());

    // Construct a source
    let source = compiler::front::ReplSource::new(repl_id, input);

    // Compile the source with the context
    match front_context.compile_source(source) {
      Ok(source_id) => {
        let items = front_context.items_of_source_id(source_id).collect::<Vec<_>>();

        // Debug
        if options.debug || options.debug_front {
          println!("======== Front Program ========");
          for item in &items {
            println!("{}", item);
          }
          println!("");
        }

        // Collect the queries
        let queries = items
          .iter()
          .filter_map(|item| {
            if let compiler::front::ast::Item::QueryDecl(q) = item {
              Some(q.query().create_relation_name())
            } else {
              None
            }
          })
          .collect::<Vec<_>>();

        // If query is non-empty, compile and execute the program
        if !queries.is_empty() {
          // This is a query event, turn all things in the context into a back program
          let mut back_ir = front_context.to_back_program();
          if let Err(e) = back_ir.apply_optimizations(&options) {
            println!("{}", e);
            return Ok(());
          }

          // Set the output to be only the query
          back_ir.outputs.clear();
          for q in &queries {
            back_ir
              .outputs
              .insert(q.clone(), common::output_option::OutputOption::Default);
          }

          // Debug
          if options.debug || options.debug_back {
            println!("======== Back Program ========");
            println!("{}", back_ir);
            println!("");
          }

          // Compile the back ir into ram
          let ram = match back_ir.to_ram_program(&options) {
            Ok(ram) => ram,
            Err(e) => {
              println!("{}", e);
              return Ok(());
            }
          };

          // Debug
          if options.debug || options.debug_ram {
            println!("======== RAM Program ========");
            println!("{}", ram);
          }

          // Interpret the ram
          match exec_context.incremental_execute(ram, &runtime_env, &ctx) {
            Ok(()) => {}
            Err(e) => {
              println!("{:?}", e);
            }
          }

          // Print the result
          for q in &queries {
            exec_context.recover(q, &runtime_env, &ctx);
            println!("{}: {}", q, exec_context.relation(q).unwrap());
          }
        }
      }
      Err(error_context) => {
        error_context.report_errors();
      }
    }

    // Increment repl id
    repl_id += 1;
  }
  Ok(())
}
