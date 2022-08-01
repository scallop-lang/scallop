use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn codegen_edge_path_left_recursion() {
  mod edge_path {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel path(a, b) = edge(a, b)
      rel path(a, c) = path(a, b) and edge(b, c)
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = edge_path::run(&mut ctx);
  expect_static_output_collection(&result.edge, vec![(0, 1), (1, 2)]);
  expect_static_output_collection(&result.path, vec![(0, 1), (1, 2), (0, 2)]);
}

#[test]
fn codegen_edge_path_right_recursion() {
  mod edge_path {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel path(a, b) = edge(a, b)
      rel path(a, c) = edge(a, b) and path(b, c)
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = edge_path::run(&mut ctx);
  expect_static_output_collection(&result.edge, vec![(0, 1), (1, 2)]);
  expect_static_output_collection(&result.path, vec![(0, 1), (1, 2), (0, 2)]);
}

#[test]
fn codegen_edge_path_binary_recursion() {
  mod edge_path {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel path(a, b) = edge(a, b)
      rel path(a, c) = path(a, b) and path(b, c)
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = edge_path::run(&mut ctx);
  expect_static_output_collection(&result.edge, vec![(0, 1), (1, 2)]);
  expect_static_output_collection(&result.path, vec![(0, 1), (1, 2), (0, 2)]);
}

#[test]
fn codegen_odd_even() {
  mod odd_even {
    use scallop_codegen::scallop;
    scallop! {
      rel even(0)
      rel odd(x) = even(x - 1), x < 10
      rel even(x) = odd(x - 1), x < 10
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = odd_even::run(&mut ctx);
  expect_static_output_collection(&result.odd, vec![(1,), (3,), (5,), (7,), (9,)]);
  expect_static_output_collection(&result.even, vec![(0,), (2,), (4,), (6,), (8,)]);
}

#[test]
fn codegen_difference_1() {
  mod difference_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel a = {(0, 1), (1, 2)}
      rel b = {(1, 1), (1, 2)}
      rel s(x, y) = a(x, y), ~b(x, y)
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = difference_1::run(&mut ctx);
  expect_static_output_collection(&result.s, vec![(0, 1)]);
}

#[test]
fn codegen_bmi_test_1() {
  mod bmi_test_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel student = {
        (1, 185, 80, "Mary"),
        (2, 175, 70, "John"),
        (3, 165, 55, "Maomao"),
      }

      rel height(id, h) = student(id, h, _, _)
      rel weight(id, w) = student(id, _, w, _)

      rel bmi(id, w as f32 / ((h * h) as f32 / 10000.0)) = height(id, h), weight(id, w)
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = bmi_test_1::run(&mut ctx);
  expect_static_output_collection(&result.height, vec![(1, 185), (2, 175), (3, 165)]);
  expect_static_output_collection(&result.weight, vec![(1, 80), (2, 70), (3, 55)]);
  expect_static_output_collection(&result.bmi, vec![(1, 23.374), (2, 22.857), (3, 20.202)]);
}

#[test]
fn codegen_bmi_test_2() {
  mod bmi_test_2 {
    use scallop_codegen::scallop;
    scallop! {
      type student(usize, f32, f32, String)

      rel student = {
        (1, 185, 80, "Mary"),
        (2, 175, 70, "John"),
        (3, 165, 55, "Maomao"),
      }

      rel height(id, h) = student(id, h, _, _)
      rel weight(id, w) = student(id, _, w, _)

      rel bmi(id, w / (h * h / 10000.0)) = height(id, h), weight(id, w)
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = bmi_test_2::run(&mut ctx);
  expect_static_output_collection(&result.height, vec![(1, 185.0), (2, 175.0), (3, 165.0)]);
  expect_static_output_collection(&result.weight, vec![(1, 80.0), (2, 70.0), (3, 55.0)]);
  expect_static_output_collection(&result.bmi, vec![(1, 23.374), (2, 22.857), (3, 20.202)]);
}

#[test]
fn codegen_const_fold_test_1() {
  mod const_fold_test_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel E(1)
      rel R(s, a) = s == x + z, x == y + 1, y == z + 1, z == 1, E(a)
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = const_fold_test_1::run(&mut ctx);
  expect_static_output_collection(&result.R, vec![(4i32, 1usize)]);
}

#[test]
fn codegen_digit_sum_test_1() {
  mod digit_sum_test_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel digit = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
      }
      rel sum_2(0, 1, x + y) = digit(0, x), digit(1, y)
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = digit_sum_test_1::run(&mut ctx);
  expect_static_output_collection(
    &result.sum_2,
    vec![
      (0, 1, 0),
      (0, 1, 1),
      (0, 1, 2),
      (0, 1, 3),
      (0, 1, 4),
      (0, 1, 5),
      (0, 1, 6),
      (0, 1, 7),
      (0, 1, 8),
    ],
  );
}

#[test]
fn codegen_digit_sum_test_2() {
  mod digit_sum_test_2 {
    use scallop_codegen::scallop;
    scallop! {
      rel digit = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
      }
      rel sum_2(a, b, c) = digit(a, x), digit(b, y), c == x + y, a == 0, b == 1
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = digit_sum_test_2::run(&mut ctx);
  expect_static_output_collection(
    &result.sum_2,
    vec![
      (0, 1, 0),
      (0, 1, 1),
      (0, 1, 2),
      (0, 1, 3),
      (0, 1, 4),
      (0, 1, 5),
      (0, 1, 6),
      (0, 1, 7),
      (0, 1, 8),
    ],
  );
}

#[test]
fn codegen_expr_test_1() {
  mod expr_test_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel eval(e, c) = constant(e, c)
      rel eval(e, a + b) = binary(e, "+", l, r), eval(l, a), eval(r, b)
      rel eval(e, a - b) = binary(e, "-", l, r), eval(l, a), eval(r, b)
      rel result(y) = eval(e, y), goal(e)

      rel constant = { (0, 1), (1, 2), (2, 3) }
      rel binary = { (3, "+", 0, 1), (4, "-", 3, 2) }
      rel goal(4)
      query result
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = expr_test_1::run(&mut ctx);
  expect_static_output_collection(&result.result, vec![(0,)])
}

#[test]
fn codegen_fib_test_1() {
  mod fib_test_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel fib :- {(0, 1), (1, 1)}
      rel fib(x, a + b) :- fib(x - 1, a), fib(x - 2, b), x <= 7
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = fib_test_1::run(&mut ctx);
  expect_static_output_collection(
    &result.fib,
    vec![(0i32, 1i32), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8), (6, 13), (7, 21)],
  );
}

#[test]
fn codegen_count_edge_1() {
  mod count_edge {
    use scallop_codegen::scallop;
    scallop! {
      rel edge :- {(0, 1), (1, 2)}
      rel num_edges(n) :- n = count(a, b: edge(a, b))
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = count_edge::run(&mut ctx);
  expect_static_output_collection(&result.num_edges, vec![(2,)]);
}

#[test]
fn codegen_out_degree_1() {
  mod out_degree {
    use scallop_codegen::scallop;
    scallop! {
      rel edge :- {(0, 1), (1, 2)}
      rel out_degree(x, n) :- n = count(y: edge(x, y))
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = out_degree::run(&mut ctx);
  expect_static_output_collection(&result.out_degree, vec![(0, 1), (1, 1)]);
}

#[test]
fn codegen_out_degree_2() {
  mod out_degree {
    use scallop_codegen::scallop;
    scallop! {
      rel node :- {0, 1, 2}
      rel edge :- {(0, 1), (1, 2)}
      rel out_degree(x, n) :- n = count(y: edge(x, y) where x: node(x))
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = out_degree::run(&mut ctx);
  expect_static_output_collection(&result.out_degree, vec![(0, 1), (1, 1), (2, 0)]);
}

#[test]
fn codegen_sum_1() {
  mod sum_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel color_num_obj :- {("blue", 1), ("red", 3), ("yellow", 6)}
      rel num_obj(n) :- n = sum(y: color_num_obj(_, y))
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = sum_1::run(&mut ctx);
  expect_static_output_collection(&result.num_obj, vec![(10,)]);
}

#[test]
fn codegen_max_1() {
  mod max_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel color_num_obj :- {("blue", 1), ("red", 3), ("yellow", 6)}
      rel max_num_per_color(n) :- n = max(y: color_num_obj(_, y))
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = max_1::run(&mut ctx);
  expect_static_output_collection(&result.max_num_per_color, vec![(6,)]);
}

#[test]
fn codegen_min_1() {
  mod min_1 {
    use scallop_codegen::scallop;
    scallop! {
      rel color_num_obj :- {("blue", 1), ("red", 3), ("yellow", 6)}
      rel min_num_per_color(n) :- n = min(y: color_num_obj(_, y))
    }
  }

  let mut ctx = unit::UnitContext::default();
  let result = min_1::run(&mut ctx);
  expect_static_output_collection(&result.min_num_per_color, vec![(1,)]);
}

#[test]
fn codegen_simple_test_1() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel path(a, b) = edge(a, b)
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.path, vec![(0, 1), (1, 2)]);
}

#[test]
fn codegen_simple_test_2() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2), (2, 2)}
      rel self_edge(a, a) :- edge(a, a)
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.self_edge, vec![(2, 2)]);
}

#[test]
fn codegen_simple_test_3() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel something(a, 2) :- edge(a, b)
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.something, vec![(0, 2), (1, 2)]);
}

#[test]
fn codegen_simple_test_4() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel something(a, 2) :- edge(a, b), b > 1
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.something, vec![(1, 2)]);
}

#[test]
fn codegen_simple_test_5() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel S = {(1, 2), (2, 3), (3, 4)}
      rel R = {(1, 2), (4, 3)}
      rel O(a, b) = S(b, a), R(a, b)
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.O, vec![(4, 3)]);
}

#[test]
fn codegen_simple_test_6() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel S = {(1, 2), (2, 3), (3, 4), (4, 3)}
      rel R = {(1, 2), (3, 4), (4, 3)}
      rel O(a, b) = S(b, a), S(a, b), R(a, b), R(b, a)
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.O, vec![(3, 4), (4, 3)]);
}

#[test]
fn codegen_simple_test_7() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel S = {(0, 1), (1, 2), (2, 3)}
      rel R = {(1), (2)}
      rel O(a, b) = S(a, b), ~R(b)
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.O, vec![(2, 3)]);
}

#[test]
fn codegen_simple_test_8() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel S = {(0, 1), (1, 2), (2, 3)}
      rel R = {(1, 2), (2, 3)}
      rel O(a, b) = S(a, b), ~R(b, c)
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.O, vec![(2, 3)]);
}

#[test]
fn codegen_simple_test_9() {
  mod simple_test {
    use scallop_codegen::scallop;
    scallop! {
      rel S = {(0, 1), (1, 2), (2, 3)}
      rel R = {(1, 2), (2, 3), (2, 2)}
      rel O(a, b) = S(a, b), ~R(a, a)
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = simple_test::run(&mut ctx);
  expect_static_output_collection(&result.O, vec![(0, 1), (1, 2)]);
}

#[test]
fn codegen_srl_1() {
  mod srl_1_test {
    use scallop_codegen::scallop;
    scallop! {
      rel verb = {
        (1, "play"),
        (4, "play"),
      }

      rel noun = {
        (2, "Tom", "Person"),
        (3, "Jerry", "Person")
      }

      rel arg1 = {
        (1, 2),
        (1, 3),
      }

      rel synonym = {
        (1, 4),
      }

      rel plays_soccer(n) =
        verb(vid0, "play"),
        arg1(vid, n),
        noun(n, _, "Person"),
        synonym(vid, vid0)

      rel how_many_play_soccer(c) = c = count(n: plays_soccer(n))
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = srl_1_test::run(&mut ctx);
  expect_static_output_collection(&result.how_many_play_soccer, vec![(2,)]);
}

#[test]
fn codegen_class_student_grade_1() {
  mod srl_1_test {
    use scallop_codegen::scallop;
    scallop! {
      rel class_student_grade = {
        (0, "tom", 50),
        (0, "jerry", 70),
        (0, "alice", 60),
        (1, "bob", 80),
        (1, "sherry", 90),
        (1, "frank", 30),
      }

      rel class_top_student(c, s) = _ = max[s](g: class_student_grade(c, s, g))
    }
  }
  let mut ctx = unit::UnitContext::default();
  let result = srl_1_test::run(&mut ctx);
  expect_static_output_collection(
    &result.class_top_student,
    vec![(0, "jerry".to_string()), (1, "sherry".to_string())],
  );
}
