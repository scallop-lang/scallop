use scallop_core::compiler::front::ast::*;
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

#[test]
fn ignore_comment_1() {
  assert!(str_to_item(r#"rel relate = { /* this is a comment */ }"#).is_ok());
  assert!(str_to_item(r#"rel relate = /* this is a comment */ {  }"#).is_ok());
  assert!(str_to_item(r#"rel relate /* this is a comment */ = {  }"#).is_ok());
}

#[test]
fn ignore_comment_2() {
  let items = str_to_items(
    r#"
    rel relate = { /* this is a comment */ }
    rel another_relate() = // this is another comment
      some_atom() /* this is another comment */
  "#,
  )
  .expect("Compile failure");
  assert_eq!(items.len(), 2);
}

#[test]
fn ignore_comment_3() {
  let items = str_to_items(
    r#"
    rel relate = { (3, 5 /* , 4, pretending to be commented out */) }
  "#,
  )
  .expect("Compile failure");
  assert_eq!(items.len(), 1);
}

#[test]
fn test_parse_specialized_predicate_1() {
  let (id, args) = str_to_specialized_predicate("range<usize>").expect("Cannot parse");
  assert_eq!(id.name(), "range");
  assert_eq!(id.location().offset_span.start, 0);
  assert_eq!(id.location().offset_span.end, 5);
  assert_eq!(args.len(), 1);
  assert_eq!(args[0].name(), "usize");
  assert_eq!(args[0].location().offset_span.start, 6);
  assert_eq!(args[0].location().offset_span.end, 11);
}

#[test]
fn test_parse_specialized_predicate_2() {
  let (id, args) = str_to_specialized_predicate("range<   usize,usize   >").expect("Cannot parse");
  assert_eq!(id.name(), "range");
  assert_eq!(id.location().offset_span.start, 0);
  assert_eq!(id.location().offset_span.end, 5);
  assert_eq!(args.len(), 2);
  assert_eq!(args[0].name(), "usize");
  assert_eq!(args[0].location().offset_span.start, 9);
  assert_eq!(args[0].location().offset_span.end, 14);
  assert_eq!(args[1].name(), "usize");
  assert_eq!(args[1].location().offset_span.start, 15);
  assert_eq!(args[1].location().offset_span.end, 20);
}

#[test]
fn test_parse_specialized_predicate_3() {
  let (id, args) = str_to_specialized_predicate("dasdf<usize, f32>").expect("Cannot parse");
  assert_eq!(id.name(), "dasdf");
  assert_eq!(id.location().offset_span.start, 0);
  assert_eq!(id.location().offset_span.end, 5);
  assert_eq!(args.len(), 2);
  assert_eq!(args[0].name(), "usize");
  assert_eq!(args[0].location().offset_span.start, 6);
  assert_eq!(args[0].location().offset_span.end, 11);
  assert_eq!(args[1].name(), "f32");
  assert_eq!(args[1].location().offset_span.start, 13);
  assert_eq!(args[1].location().offset_span.end, 16);
}

#[test]
fn test_parse_specialized_predicate_4() {
  let (id, args) = str_to_specialized_predicate("dasdf  <  usize    , f32 >").expect("Cannot parse");
  assert_eq!(id.name(), "dasdf");
  assert_eq!(id.location().offset_span.start, 0);
  assert_eq!(id.location().offset_span.end, 5);
  assert_eq!(args.len(), 2);
  assert_eq!(args[0].name(), "usize");
  assert_eq!(args[0].location().offset_span.start, 10);
  assert_eq!(args[0].location().offset_span.end, 15);
  assert_eq!(args[1].name(), "f32");
  assert_eq!(args[1].location().offset_span.start, 21);
  assert_eq!(args[1].location().offset_span.end, 24);
}
