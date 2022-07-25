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
fn fib_test_1() {
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
    vec![
      (0i32, 1i32),
      (1, 1),
      (2, 2),
      (3, 3),
      (4, 5),
      (5, 8),
      (6, 13),
      (7, 21),
    ],
  );
}
