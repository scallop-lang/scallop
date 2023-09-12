use scallop_core::compiler::front::ast;
use scallop_core::compiler::front::attribute::*;
use scallop_core::compiler::front::FrontContext;
use scallop_core::testing::*;

mod attr_1 {
  use super::*;

  #[derive(Clone)]
  struct Foo;

  impl AttributeProcessor for Foo {
    fn name(&self) -> String {
      "foo".to_string()
    }

    fn apply(&self, _: &ast::Item, attr: &ast::Attribute) -> Result<AttributeAction, AttributeError> {
      if attr.num_pos_args() != 3 {
        Err(AttributeError::new_custom(
          "foo attribute requires 3 arguments".to_string(),
        ))
      } else {
        if attr.pos_arg(0).and_then(|arg| Some(arg.is_tuple())) == Some(true) {
          Ok(AttributeAction::Nothing)
        } else {
          Err(AttributeError::new_custom(
            "foo attribute requires a tuple as the first argument".to_string(),
          ))
        }
      }
    }
  }

  #[test]
  fn attr_1_test_1() {
    expect_compile(
      r#"
      @foo((1, 2), 3, 4)
      type my_relation(a: i32, b: i32)
      "#,
    );
  }

  #[test]
  fn attr_1_test_2() {
    expect_front_compile_failure_with_modifier(
      r#"
      @foo((1, 2), 3)
      type my_relation(a: i32, b: i32)
      "#,
      |ctx: &mut FrontContext| {
        ctx
          .register_attribute_processor(Foo)
          .expect("Cannot register attribute");
      },
      |s| s.contains("foo attribute requires 3 arguments"),
    );
  }

  #[test]
  fn attr_1_test_3() {
    expect_front_compile_failure_with_modifier(
      r#"
      @foo("asdfasdf", 3, 5)
      type my_relation(a: i32, b: i32)
      "#,
      |ctx: &mut FrontContext| {
        ctx
          .register_attribute_processor(Foo)
          .expect("Cannot register attribute");
      },
      |s| s.contains("foo attribute requires a tuple as the first argument"),
    );
  }
}
