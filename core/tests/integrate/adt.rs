use scallop_core::testing::*;

#[test]
fn adt_arith_formula_eval_1() {
  expect_interpret_result(
    r#"
      type Expr = Const(i32) | Add(Expr, Expr)

      rel eval(x, y)       = case x is Const(y)
      rel eval(x, y1 + y2) = case x is Add(x1, x2) and
                             eval(x1, y1) and eval(x2, y2)

      const MY_EXPR = Add(Const(5), Add(Const(3), Const(6)))

      rel result(y) = eval(MY_EXPR, y)
    "#,
    ("result", vec![(14i32,)]),
  )
}

#[test]
fn adt_list_1() {
  expect_interpret_result(
    r#"
      type List = Nil() | Cons(i32, List)

      rel list_sum(l, 0)      = case l is Nil()
      rel list_sum(l, hd + s) = case l is Cons(hd, tl) and list_sum(tl, s)

      const MY_LIST = Cons(1, Cons(2, Cons(3, Nil())))

      rel result(y) = list_sum(MY_LIST, y)
    "#,
    ("result", vec![(6i32,)]),
  )
}

#[test]
fn adt_binary_tree_1() {
  expect_interpret_result(
    r#"
      type Tree = Nil() | Node(i32, Tree, Tree)

      rel tree_depth(t, 0)                = case t is Nil()
      rel tree_depth(t, $max(ld, rd) + 1) = case t is Node(_, lt, rt) and
                                            tree_depth(lt, ld) and tree_depth(rt, rd)

      const MY_TREE = Node(1, Node(2, Nil(), Node(3, Nil(), Nil())), Node(4, Nil(), Nil()))

      rel result(y) = tree_depth(MY_TREE, y)
    "#,
    ("result", vec![(3i32,)]),
  )
}

const RE_PROGRAM: &'static str = r#"
  type RE = Char(char) | Nil() | Con(RE, RE) | Or(RE, RE) | Star(RE)

  rel match(r, i, i)     = case r is Nil(), string_chars(s, i, _), input_string(s)
  rel match(r, i, i + 1) = case r is Char(c), input_string(s), string_chars(s, i, c)
  rel match(r, s, e)     = case r is Con(r1, r2), match(r1, s, m), match(r2, m, e)
  rel match(r, s, e)     = case r is Or(r1, r2), match(r1, s, e)
  rel match(r, s, e)     = case r is Or(r1, r2), match(r2, s, e)
  rel match(r, i, i)     = case r is Star(r1), string_chars(s, i, _), input_string(s)
  rel match(r, s, e)     = case r is Star(r1), match(r1, s, e)
  rel match(r, s, e)     = case r is Star(r1), match(r1, s, m), match(r, m, e)
"#;

#[test]
fn adt_regex_1() {
  expect_interpret_result(
    &format!(
      "{RE_PROGRAM}\n{}",
      r#"
        const MY_RE = Con(Char('a'), Char('b'))
        rel input_string("ab")
        rel result() = match(MY_RE, 0, 2)
      "#,
    ),
    ("result", vec![()]),
  )
}

#[test]
fn adt_regex_2() {
  expect_interpret_result(
    &format!(
      "{RE_PROGRAM}\n{}",
      r#"
        const MY_RE = Con(Star(Char('a')), Char('b'))
        rel input_string("aaaaaaaab")
        rel result() = match(MY_RE, 0, 9)
      "#,
    ),
    ("result", vec![()]),
  )
}

const CLEVR_PROGRAM: &'static str = r#"
  type Color = RED | GREEN | BLUE
  type Size = LARGE | SMALL
  type SpatialRela = LEFT | RIGHT
  type Expr = Scene() | Color(Color, Expr) | Size(Size, Expr) | Rela(SpatialRela, Expr, Expr) | RelaInv(SpatialRela, Expr, Expr)

  rel eval(e, output_obj) = case e is Scene(), input_obj_ids(output_obj)
  rel eval(e, output_obj) = case e is Color(c, e1), eval(e1, output_obj), input_obj_color(output_obj, c)
  rel eval(e, output_obj) = case e is Size(s, e1), eval(e1, output_obj), input_obj_size(output_obj, s)
  rel eval(e, o2) = case e is Rela(r, e1, e2), eval(e1, o1), eval(e2, o2), input_obj_rela(r, o1, o2)
  rel eval(e, o1) = case e is RelaInv(r, e1, e2), eval(e1, o1), eval(e2, o2), input_obj_rela(r, o1, o2)
"#;

#[test]
fn adt_clevr_1() {
  expect_interpret_result(
    &format!(
      "{CLEVR_PROGRAM}\n{}",
      r#"
        rel input_obj_ids = {0, 1}
        rel input_obj_color = {(0, RED), (1, GREEN)}
        rel input_obj_size = {(0, LARGE), (1, SMALL)}
        rel input_obj_rela = {(0, 1, LEFT), (1, 0, RIGHT)}

        const MY_EXPR = Color(RED, Scene())

        rel result(o) = eval(MY_EXPR, o)
      "#,
    ),
    ("result", vec![(0usize,)]),
  )
}

const EQSAT_1_PROGRAM: &'static str = r#"
  // The language for simple symbolic arithmetic expression
  type Expr = Const(i32)
            | Var(String)
            | Add(Expr, Expr)

  // A relation `to_string` for visualizing
  rel to_string(p, i as String) = case p is Const(i)
  rel to_string(p, v) = case p is Var(v)
  rel to_string(p, $format("({} + {})", s1, s2)) = case p is Add(p1, p2) and to_string(p1, s1) and to_string(p2, s2)

  // Relation for expression
  rel expr(p) = case p is Const(_) or case p is Var(_) or case p is Add(_, _)

  // Definition of rewrite rules suggesting equivalence
  rel equivalent(p, p) = expr(p)
  rel equivalent(p1, p3) = equivalent(p1, p2) and equivalent(p2, p3)
  rel equivalent(p, new Add(b, a)) = case p is Add(a, b)
  rel equivalent(p1, new Add(a2, b2)) = case p1 is Add(a1, b1) and equivalent(a1, a2) and equivalent(b1, b2)
  rel equivalent(p, new Const(a + b)) = case p is Add(Const(a), Const(b))
  rel equivalent(p, p1) = case p is Add(p1, Const(0))

  // Definition of weight
  rel weight(p, 1) = case p is Const(_)
  rel weight(p, 1) = case p is Var(_)
  rel weight(p, w1 + w2 + 1) = case p is Add(p1, p2) and weight(p1, w1) and weight(p2, w2)

  // Compute equivalent programs
  rel equiv_programs(sp) = input_program(p) and equivalent(p, sp)

  // Find the best program (minimum weight) among all programs equivalent to p
  rel best_program(p) = w := min[p](w: equiv_programs(p) and weight(p, w))
  rel best_program_str(s) = best_program(best_prog) and to_string(best_prog, s)
  query best_program_str
"#;

#[test]
fn equality_saturation_1() {
  expect_interpret_result(
    &format!(
      "{EQSAT_1_PROGRAM}\n{}",
      r#"
        const PROGRAM = Add(Add(Const(3), Const(-3)), Var("a"))
        rel input_program(PROGRAM)
      "#,
    ),
    ("best_program_str", vec![("a".to_string(),)]),
  )
}

#[test]
fn equality_saturation_2() {
  expect_interpret_result(
    &format!(
      "{EQSAT_1_PROGRAM}\n{}",
      r#"
        const PROGRAM = Add(Add(Const(3), Const(-3)), Const(5))
        rel input_program(PROGRAM)
      "#,
    ),
    ("best_program_str", vec![("5".to_string(),)]),
  )
}

const TYPE_INF_1_PROGRAM: &'static str = r#"
  type Op = EQ | NEQ | GEQ | LEQ | GT | LT | AND | OR | XOR | ADD | SUB | MUL | DIV | NEG | NOT

  type Expr = Number(i32)
            | Boolean(bool)
            | Variable(String)
            | Binary(Op, Expr, Expr)
            | Unary(Op, Expr)
            | Let(String, Expr, Expr)
            | Ite(Expr, Expr, Expr)

  type Type = BOOL | INT

  type input_program(expr: Expr)

  // =================

  // Pretty printing of operators
  rel op_to_string = {
    (EQ, "=="), (NEQ, "!="),
    (GEQ, ">="), (LEQ, "<="), (GT, ">"), (LT, "<"),
    (AND, "&&"), (OR, "||"), (XOR, "^"),
    (ADD, "+"), (SUB, "-"), (MUL, "*"), (DIV, "/"),
    (NEG, "-"), (NOT, "!")
  }

  // Pretty printing of type
  rel ty_to_string = {(BOOL, "bool"), (INT, "int")}

  // Pretty printing of expressions
  rel expr_to_string(e, x as String) = case e is Number(x)
  rel expr_to_string(e, x as String) = case e is Boolean(x)
  rel expr_to_string(e, x) = case e is Variable(x)
  rel expr_to_string(e, $format("({} {} {})", op1_str, op_str, op2_str)) = case e is Binary(op, op1, op2) and expr_to_string(op1, op1_str) and expr_to_string(op2, op2_str) and op_to_string(op, op_str)
  rel expr_to_string(e, $format("({}{})", op_str, op1_str)) = case e is Unary(op, op1) and expr_to_string(op1, op1_str) and op_to_string(op, op_str)
  rel expr_to_string(e, $format("let {} = {} in {}", x, b_str, i_str)) = case e is Let(x, b, i) and expr_to_string(b, b_str) and expr_to_string(i, i_str)
  rel expr_to_string(e, $format("if {} then {} else {}", cs, ts, es)) = case e is Ite(c, t, e) and expr_to_string(c, cs) and expr_to_string(t, ts) and expr_to_string(e, es)

  // =================

  // Basic types of operators
  rel eq_op = {EQ, NEQ}
  rel comp_op = {GEQ, LEQ, GT, LT}
  rel logical_op = {AND, OR, XOR}
  rel arith_op = {ADD, SUB, MUL, DIV}
  rel unary_arith_op = {NEG}
  rel unary_logical_op = {NOT}

  // Typing environment
  type Env = Empty() | Cons(String, Type, Env)
  const EMPTY_ENV = Empty()

  // Find a variable stored in the typing environment
  type find_type(bound env: Env, bound var: String, free ty: Type)
  rel find_type(env, var, ty) = case e is Cons(var, ty, _)
  rel find_type(env, var, ty) = case e is Cons(vp, _, tl) and vp != var and find_type(tl, var, ty)

  // The type (`ty`) of an expression (`expr`) under an environment (`env`)
  type type_of(bound env: Env, bound expr: Expr, ty: Type)

  // Typing rules
  rel type_of(env, e, BOOL) = case e is Boolean(_)
  rel type_of(env, e, INT) = case e is Number(_)
  rel type_of(env, e, ty) = case e is Variable(x) and find_type(env, x, ty)
  rel type_of(env, e, BOOL) = case e is Binary(op, op1, op2) and eq_op(op) and type_of(env, op1, ty) and type_of(env, op2, ty)
  rel type_of(env, e, BOOL) = case e is Binary(op, op1, op2) and comp_op(op) and type_of(env, op1, INT) and type_of(env, op2, INT)
  rel type_of(env, e, BOOL) = case e is Binary(op, op1, op2) and logical_op(op) and type_of(env, op1, BOOL) and type_of(env, op2, BOOL)
  rel type_of(env, e, INT) = case e is Binary(op, op1, op2) and arith_op(op) and type_of(env, op1, INT) and type_of(env, op2, INT)
  rel type_of(env, e, BOOL) = case e is Unary(op, op1) and unary_logical_op(op) and type_of(env, op1, BOOL)
  rel type_of(env, e, INT) = case e is Unary(op, op1) and unary_arith_op(op) and type_of(env, op1, INT)
  rel type_of(env, e, ty_i) = to_infer_let_cons(env, e, sub_env, i) and type_of(sub_env, i, ty_i)
  rel type_of(env, e, ty) = case e is Ite(c, t, e) and type_of(env, c, BOOL) and type_of(env, t, ty) and type_of(env, e, ty)

  // Helpers
  type to_infer_let_cons(bound env: Env, bound expr: Expr, out_env: Env, cont: Expr)
  rel to_infer_let_cons(env, e, new Cons(x, ty_b, env), i) = case e is Let(x, b, i) and type_of(env, b, ty_b)

  // The result if the type of the input program
  rel result(expr_str, ty_str) = input_program(p) and expr_to_string(p, expr_str) and type_of(EMPTY_ENV, p, ty) and ty_to_string(ty, ty_str)
  query result
"#;

#[test]
fn type_inf_1() {
  expect_interpret_result(
    &format!(
      "{TYPE_INF_1_PROGRAM}\n{}",
      r#"
        const PROGRAM = Let("x", Number(3), Binary(EQ, Variable("x"), Number(4)))
        rel input_program(PROGRAM)
      "#,
    ),
    (
      "result",
      vec![("let x = 3 in (x == 4)".to_string(), "bool".to_string())],
    ),
  )
}

#[test]
fn type_inf_2() {
  expect_interpret_empty_result(
    &format!(
      "{TYPE_INF_1_PROGRAM}\n{}",
      r#"
        const PROGRAM = Let("x", Number(3), Binary(ADD, Variable("x"), Boolean(false)))
        rel input_program(PROGRAM)
      "#,
    ),
    "result",
  )
}

#[test]
fn adt_dynamic_1() {
  expect_interpret_result(
    r#"
    type Expr = Const(f32) | Add(Expr, Expr) | Sub(Expr, Expr)

    rel my_entity($parse_entity("Add(Const(1), Const(3))"))

    type eval(bound e: Expr, v: f32)
    rel eval(e, v) = case e is Const(v)
    rel eval(e, v1 + v2) = case e is Add(e1, e2) and eval(e1, v1) and eval(e2, v2)
    rel eval(e, v1 - v2) = case e is Sub(e1, e2) and eval(e1, v1) and eval(e2, v2)

    rel result(v) = my_entity(e) and eval(e, v)

    query result
    "#,
    ("result", vec![(4.0f32,)]),
  )
}

#[test]
fn foreign_predicate_dynamic_adt_1() {
  use scallop_core::common::foreign_predicate::ForeignPredicate;
  use scallop_core::common::input_tag::DynamicInputTag;
  use scallop_core::common::value::Value;
  use scallop_core::common::value_type::ValueType;
  use scallop_core::integrate;
  use scallop_core::runtime::*;
  use scallop_core::utils::*;

  #[derive(Clone, Debug)]
  struct DummySemanticParser;

  impl ForeignPredicate for DummySemanticParser {
    fn name(&self) -> String {
      "dummy_semantic_parser".to_string()
    }

    fn arity(&self) -> usize {
      2
    }

    fn num_bounded(&self) -> usize {
      1
    }

    fn argument_type(&self, i: usize) -> ValueType {
      match i {
        0 => ValueType::String,
        1 => ValueType::Entity,
        _ => panic!("Should not happen"),
      }
    }

    fn evaluate(&self, args: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
      println!("{:?}", args);
      match args[0].as_str() {
        "prompt_1" => vec![(
          DynamicInputTag::None,
          vec![Value::EntityString("Add(Const(1), Const(2))".to_string())],
        )],
        "prompt_2" => vec![(
          DynamicInputTag::None,
          vec![Value::EntityString("Const(10)".to_string())],
        )],
        _ => vec![],
      }
    }
  }

  // Initialize a context
  let prov_ctx = provenance::unit::UnitProvenance::default();
  let mut ctx = integrate::IntegrateContext::<_, RcFamily>::new(prov_ctx);

  // Register the foreign predicate
  ctx.register_foreign_predicate(DummySemanticParser).unwrap();

  // Add a program
  ctx
    .add_program(
      r#"
    type Expr = Const(f32) | Add(Expr, Expr)
    rel eval(x, y)       = case x is Const(y)
    rel eval(x, y1 + y2) = case x is Add(x1, x2) and eval(x1, y1) and eval(x2, y2)
    rel prompt = {(1, "prompt_1"), (2, "prompt_2")}
    rel result(i, y) = prompt(i, x) and dummy_semantic_parser(x, e) and eval(e, y)
    "#,
    )
    .unwrap();

  // Run the context
  ctx.run().unwrap();

  // Test the result
  expect_output_collection(
    "result",
    ctx.computed_relation_ref("result").unwrap(),
    vec![(1i32, 3.0f32), (2, 10.0f32)],
  );
}
