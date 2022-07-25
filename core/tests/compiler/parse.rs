use scallop_core::compiler::front::parser::*;

#[test]
fn parse_type_decl() {
  assert!(str_to_item("type symbol = usize").is_ok());
  assert!(str_to_item("type object_id <: symbol").is_ok());
}

#[test]
fn parse_constant_set() {
  assert!(str_to_item("rel edge :- {(0, 1), (1, 2)}").is_ok());
  assert!(str_to_item("rel is_ok :- {(0, true), (1, false)}").is_ok());

  // Probabilities
  assert!(str_to_item(r#"rel edge :- { 0.9::(0, 1), 0.8::(1, 2) }"#).is_ok());
  assert!(str_to_item(r#"rel color :- { 0.9::(0, "blue"), 0.8::(1, "green") }"#).is_ok());
}

#[test]
fn parse_fact() {
  assert!(str_to_item("rel edge(1, 2)").is_ok());
  assert!(str_to_item("rel 0.9::color(0, \"blue\")").is_ok());
}

#[test]
fn parse_rule() {
  assert!(str_to_item(r#"rel path(a, b) :- edge(a, b)"#).is_ok());
  assert!(str_to_item(r#"rel path(a, b) :- path(a, c), edge(c, b)"#).is_ok());
  assert!(str_to_item(r#"rel path(a, b) :- path(a, c) /\ edge(c, b)"#).is_ok());
  assert!(str_to_item(r#"rel path(a, b) :- edge(a, b) \/ path(a, c) /\ edge(c, b)"#).is_ok());
}
