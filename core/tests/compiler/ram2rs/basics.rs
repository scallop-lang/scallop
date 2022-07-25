use scallop_core::compiler::front::StringSource;
use scallop_core::compiler::{compile_source_to_ram, CompileOptions};

#[test]
fn ram2rs_edge_path() {
  let opt = CompileOptions::default();
  let program = r#"
    rel edge = {(0, 1), (1, 2), (2, 3)}
    rel path(a, b) = edge(a, b) or (path(a, c) and edge(c, b))
  "#;
  let source = StringSource::new(program.into());
  let ram = compile_source_to_ram(source).unwrap();
  let _ = ram.to_rs_module(&opt);
}
