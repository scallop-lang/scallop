use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;
use scallop_core::utils::*;

#[test]
fn basic_edge_path_left_recursion() {
  expect_interpret_result(
    r#"
      rel edge = {(0, 2), (1, 2), (2, 3)}
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ edge(c, b)
      query path
    "#,
    ("path", vec![(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]),
  );
}

#[test]
fn basic_edge_path_right_recursion() {
  expect_interpret_result(
    r#"
      rel edge = {(0, 2), (1, 2), (2, 3)}
      rel path(a, b) = edge(a, b) \/ edge(a, c) /\ path(c, b)
      query path
    "#,
    ("path", vec![(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]),
  );
}

#[test]
fn basic_edge_path_binary_recursion() {
  expect_interpret_result(
    r#"
      rel edge = {(0, 2), (1, 2), (2, 3)}
      rel path(a, b) = edge(a, b) \/ path(a, c) /\ path(c, b)
      query path
    "#,
    ("path", vec![(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]),
  );
}

#[test]
fn basic_odd_even() {
  expect_interpret_multi_result(
    r#"
      rel even(0)
      rel odd(x) = even(x - 1), x < 10
      rel even(x) = odd(x - 1), x < 10
    "#,
    vec![
      ("odd", vec![(1,), (3,), (5,), (7,), (9,)].into()),
      ("even", vec![(0,), (2,), (4,), (6,), (8,)].into()),
    ],
  );
}

#[test]
fn basic_difference_1() {
  expect_interpret_result(
    r#"
      rel a = {(0, 1), (1, 2)}
      rel b = {(1, 1), (1, 2)}
      rel s(x, y) = a(x, y), ~b(x, y)
      query s
    "#,
    ("s", vec![(0, 1)]),
  );
}

#[test]
fn bmi_test_1() {
  expect_interpret_multi_result(
    r#"
      rel student = {
        (1, 185, 80, "Mary"),
        (2, 175, 70, "John"),
        (3, 165, 55, "Maomao"),
      }

      rel height(id, h) = student(id, h, _, _)
      rel weight(id, w) = student(id, _, w, _)

      rel bmi(id, w as f32 / ((h * h) as f32 / 10000.0)) = height(id, h), weight(id, w)
    "#,
    vec![
      ("height", vec![(1, 185), (2, 175), (3, 165)].into()),
      ("weight", vec![(1, 80), (2, 70), (3, 55)].into()),
      ("bmi", vec![(1, 23.374f32), (2, 22.857), (3, 20.202)].into()),
    ],
  )
}

#[test]
fn bmi_test_2() {
  expect_interpret_multi_result(
    r#"
      type student(usize, f32, f32, String)

      rel student = {
        (1, 185, 80, "Mary"),
        (2, 175, 70, "John"),
        (3, 165, 55, "Maomao"),
      }

      rel height(id, h) = student(id, h, _, _)
      rel weight(id, w) = student(id, _, w, _)

      rel bmi(id, w / (h * h / 10000.0)) = height(id, h), weight(id, w)
    "#,
    vec![
      ("height", vec![(1usize, 185f32), (2, 175.0), (3, 165.0)].into()),
      ("weight", vec![(1usize, 80f32), (2, 70.0), (3, 55.0)].into()),
      ("bmi", vec![(1usize, 23.374f32), (2, 22.857), (3, 20.202)].into()),
    ],
  )
}

#[test]
fn const_fold_test_1() {
  expect_interpret_result(
    r#"
      rel E(1)
      rel R(s, a) = s == x + z, x == y + 1, y == z + 1, z == 1, E(a)
    "#,
    ("R", vec![(4, 1)]),
  );
}

#[test]
fn const_fold_test_2() {
  expect_interpret_result(
    r#"
      rel R(s) = s == x + z, x == y + 1, y == z + 1, z == 1
    "#,
    ("R", vec![(4,)]),
  );
}

#[test]
fn count_test_1() {
  expect_interpret_result(
    r#"
      type R(usize, String)
      type S(usize)

      rel R = {(0, "a"), (1, "b"), (1, "a"), (0, "c"), (0, "d")}
      rel S(i) :- i = count(s: R(o, s))
    "#,
    ("S", vec![(4usize,)]),
  );
}

#[test]
fn count_test_2() {
  expect_interpret_result(
    r#"
      type R(usize, String)
      type S(usize)

      rel R = {(0, "a"), (1, "b"), (1, "a"), (0, "c"), (0, "d")}
      rel O(o, i) :- i = count(s: R(o, s))
    "#,
    ("O", vec![(0usize, 3usize), (1, 2)]),
  );
}

#[test]
fn topk_test_1() {
  expect_interpret_result(
    r#"
      rel r1 = {(0, "x"), (0, "y"), (1, "y"), (1, "z")}
      rel r2(id, sym) :- sym = top<1>(s: r1(id, s))
    "#,
    ("r2", vec![(0, "x".to_string()), (1, "y".to_string())]),
  );
}

#[test]
fn digit_sum_test_1() {
  expect_interpret_result(
    r#"
      rel digit = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
      }
      rel sum_2(0, 1, x + y) = digit(0, x), digit(1, y)
    "#,
    (
      "sum_2",
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
    ),
  );
}

#[test]
fn digit_sum_test_2() {
  expect_interpret_result(
    r#"
      rel digit = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
      }
      rel sum_2(a, b, c) = digit(a, x), digit(b, y), c == x + y, a == 0, b == 1
    "#,
    (
      "sum_2",
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
    ),
  );
}

#[test]
fn expr_test_1() {
  expect_interpret_result(
    r#"
      rel eval(e, c) = constant(e, c)
      rel eval(e, a + b) = binary(e, "+", l, r), eval(l, a), eval(r, b)
      rel eval(e, a - b) = binary(e, "-", l, r), eval(l, a), eval(r, b)
      rel result(y) = eval(e, y), goal(e)

      rel constant = { (0, 1), (1, 2), (2, 3) }
      rel binary = { (3, "+", 0, 1), (4, "-", 3, 2) }
      rel goal(4)
      query result
    "#,
    ("result", vec![(0i32,)]),
  );
}

#[test]
fn fib_test_0() {
  expect_interpret_result(
    r#"
      rel fib :- {(0, 1), (1, 1)}
      rel fib(x, a + b) :- fib(x - 1, a), fib(x - 2, b), x <= 5
    "#,
    ("fib", vec![(0i32, 1i32), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8)]),
  );
}

#[test]
fn fib_test_1() {
  expect_interpret_result(
    r#"
      rel fib :- {(0, 1), (1, 1)}
      rel fib(x, a + b) :- fib(x - 1, a), fib(x - 2, b), x <= 7
    "#,
    (
      "fib",
      vec![(0i32, 1i32), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8), (6, 13), (7, 21)],
    ),
  );
}

#[test]
fn fib_dt_test_1() {
  expect_interpret_result(
    r#"
      rel d_fib :- {(7)}
      rel d_fib(x - 1) :- d_fib(x), x > 1
      rel d_fib(x - 2) :- d_fib(x), x > 1

      rel fib :- {(0, 1), (1, 1)}
      rel fib(x, a + b) :- d_fib(x), fib(x - 1, a), fib(x - 2, b), x > 1

      rel result(y) :- fib(7, y)
    "#,
    ("result", vec![(21,)]),
  );
}

#[test]
fn aggregate_argmax_1() {
  expect_interpret_result(
    r#"
      rel exam_grades = {("tom", 50.0), ("mary", 60.0)}
      rel best_student(n) = n := argmax[n](s: exam_grades(n, s))
    "#,
    ("best_student", vec![("mary".to_string(),)]),
  )
}

#[test]
fn obj_color_test_1() {
  expect_interpret_result(
    r#"
      rel object_color(0, "blue")
      rel object_color(1, "green")
      rel object_color(2, "blue")
      rel object_color(3, "green")
      rel object_color(4, "green")
      rel object_color(5, "red")

      rel color_count(c, n) :- n := count(o: object_color(o, c))
      rel max_color(c) :- c := argmax[c](n: color_count(c, n))
    "#,
    ("max_color", vec![("green".to_string(),)]),
  );
}

#[test]
fn obj_color_test_2() {
  expect_interpret_result(
    r#"
      rel object_color = {
        (0, "blue"),
        (1, "green"),
        (2, "blue"),
        (3, "green"),
        (4, "green"),
        (5, "blue"),
      }
      rel max_color(c) = c := argmax[c](n: n := count(o: object_color(o, c)))
    "#,
    ("max_color", vec![("blue".to_string(),), ("green".to_string(),)]),
  );
}

#[test]
fn simple_test_1() {
  expect_interpret_result(
    r#"
      rel edge = {(0, 1), (1, 2)}
      rel path(a, b) = edge(a, b)
    "#,
    ("path", vec![(0, 1), (1, 2)]),
  );
}

#[test]
fn simple_test_2() {
  expect_interpret_result(
    r#"
      rel edge = {(0, 1), (1, 2), (2, 2)}
      rel self_edge(a, a) :- edge(a, a)
    "#,
    ("self_edge", vec![(2, 2)]),
  );
}

#[test]
fn simple_test_3() {
  expect_interpret_result(
    r#"
      rel edge = {(0, 1), (1, 2)}
      rel something(a, 2) :- edge(a, b)
    "#,
    ("something", vec![(0, 2), (1, 2)]),
  );
}

#[test]
fn simple_test_4() {
  expect_interpret_result(
    r#"
      rel edge = {(0, 1), (1, 2)}
      rel something(a, 2) :- edge(a, b), b > 1
    "#,
    ("something", vec![(1, 2)]),
  );
}

#[test]
fn simple_test_5() {
  expect_interpret_result(
    r#"
      rel S = {(1, 2), (2, 3), (3, 4)}
      rel R = {(1, 2), (4, 3)}
      rel O(a, b) = S(b, a), R(a, b)
    "#,
    ("O", vec![(4, 3)]),
  );
}

#[test]
fn simple_test_6() {
  expect_interpret_result(
    r#"
      rel S = {(1, 2), (2, 3), (3, 4), (4, 3)}
      rel R = {(1, 2), (3, 4), (4, 3)}
      rel O(a, b) = S(b, a), S(a, b), R(a, b), R(b, a)
    "#,
    ("O", vec![(3, 4), (4, 3)]),
  );
}

#[test]
fn simple_test_7() {
  expect_interpret_result(
    r#"
      rel S = {(0, 1), (1, 2), (2, 3)}
      rel R = {(1), (2)}
      rel O(a, b) = S(a, b), ~R(b)
    "#,
    ("O", vec![(2, 3)]),
  );
}

#[test]
fn simple_test_8() {
  expect_interpret_result(
    r#"
      rel S = {(0, 1), (1, 2), (2, 3)}
      rel R = {(1, 2), (2, 3)}
      rel O(a, b) = S(a, b), ~R(b, c)
    "#,
    ("O", vec![(2, 3)]),
  )
}

#[test]
fn simple_test_9() {
  expect_interpret_result(
    r#"
      rel S = {(0, 1), (1, 2), (2, 3)}
      rel R = {(1, 2), (2, 3), (2, 2)}
      rel O(a, b) = S(a, b), ~R(a, a)
    "#,
    ("O", vec![(0, 1), (1, 2)]),
  )
}

#[test]
fn srl_1() {
  expect_interpret_result(
    r#"
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
    "#,
    ("how_many_play_soccer", vec![(2usize,)]),
  );
}

#[test]
fn class_student_grade_1() {
  expect_interpret_result(
    r#"
      rel class_student_grade = {
        (0, "tom", 50),
        (0, "jerry", 70),
        (0, "alice", 60),
        (1, "bob", 80),
        (1, "sherry", 90),
        (1, "frank", 30),
      }

      rel class_top_student(c, s) = s := argmax[s](g: class_student_grade(c, s, g))
    "#,
    (
      "class_top_student",
      vec![(0, "jerry".to_string()), (1, "sherry".to_string())],
    ),
  )
}

#[test]
fn class_student_grade_2() {
  expect_interpret_result(
    r#"
      rel class_student_grade = {
        (0, "tom", 50),
        (0, "jerry", 70),
        (0, "alice", 60),
        (1, "bob", 80),
        (1, "sherry", 90),
        (1, "frank", 30),
      }

      rel avg_score((s as f32) / (n as f32)) =
        s := sum[class, name](x: class_student_grade(class, name, x)),
        n := count(class, name: class_student_grade(class, name, _))
    "#,
    ("avg_score", vec![(63.333f32,)]),
  )
}

#[test]
fn aggr_sum_test_1() {
  expect_interpret_result(
    r#"
      rel sales = {(0, 100), (1, 100), (2, 200)}
      rel total_sale(t) = t := sum[p](s: sales(p, s))
    "#,
    ("total_sale", vec![(400i32,)]),
  )
}

#[test]
fn aggr_sum_test_2() {
  expect_interpret_result(
    r#"
      rel sales = {("market", "tom", 100), ("ads", "tom", 100), ("ads", "jenny", 200)}
      rel total_sale(t) = t := sum[d, p](s: sales(d, p, s))
    "#,
    ("total_sale", vec![(400i32,)]),
  )
}

#[test]
fn aggr_sum_test_3() {
  expect_interpret_result(
    r#"
      rel sales = {("market", "tom", 100), ("ads", "tom", 100), ("ads", "jenny", 100)}
      rel dp_sale(d, t) = t := sum[p](s: sales(d, p, s))
    "#,
    (
      "dp_sale",
      vec![("market".to_string(), 100i32), ("ads".to_string(), 200i32)],
    ),
  )
}

#[test]
fn unused_relation_1() {
  expect_interpret_result(
    r#"
      rel A = {(0, 1), (1, 2)}
      rel B = {("haha"), ("wow")}
      rel S(b, 1) = B(b)
      query S
    "#,
    ("S", vec![("haha".to_string(), 1), ("wow".to_string(), 1)]),
  )
}

#[test]
fn atomic_query_1() {
  expect_interpret_multi_result(
    r#"
      rel S = {(0, 1), (1, 2), (0, 2)}
      query S(0, _)
      query S(_, 2)
    "#,
    vec![
      ("S(0, _)", vec![(0, 1), (0, 2)].into()),
      ("S(_, 2)", vec![(0, 2), (1, 2)].into()),
    ],
  )
}

#[test]
fn atomic_query_2() {
  expect_interpret_multi_result(
    r#"
      rel S = {(1, 2), (2, 4), (1, 1), (2, 2)}
      query S(a, a * 2)
    "#,
    vec![("S(a, (a * 2))", vec![(1i32, 2i32), (2, 4)].into())],
  )
}

#[test]
fn negate_query_1() {
  expect_interpret_multi_result(
    r#"
      rel A = {(0, 1)}
      rel B = {(0, 1, "O")}
      rel Q() = A(a, b), ~B(a, b, "O")
    "#,
    vec![("Q", TestCollection::empty())],
  )
}

#[test]
fn negate_query_2() {
  expect_interpret_multi_result(
    r#"
      rel B("Alice")
      rel A() :- ~B(_)
      query A
    "#,
    vec![("A", TestCollection::empty())],
  )
}

#[test]
fn join_and_arith_1() {
  expect_interpret_multi_result(
    r#"
      rel eventually(start, start, goal) = event(start, goal)
      rel eventually(start, end, goal) =
        event(next, _) and
        eventually(next, end, goal) and
        next == start + 1
    "#,
    vec![("eventually", TestCollection::empty())],
  )
}

#[test]
fn join_and_arith_2() {
  expect_interpret_result(
    r#"
      rel event = {
        (0, "V1"),
        (1, "V1"),
        (2, "V2"),
        (3, "O"),
      }
      rel eventually(end, end, goal) = event(end, goal)
      rel eventually(start, end, goal) =
        event(start, g) and
        ~event(start, goal) and
        event(next, _) and
        eventually(next, end, goal) and
        next == start + 1
    "#,
    (
      "eventually",
      vec![
        (0, 0, "V1".to_string()),
        (1, 1, "V1".to_string()),
        (0, 2, "V2".to_string()),
        (1, 2, "V2".to_string()),
        (2, 2, "V2".to_string()),
        (0, 3, "O".to_string()),
        (1, 3, "O".to_string()),
        (2, 3, "O".to_string()),
        (3, 3, "O".to_string()),
      ],
    ),
  )
}

#[test]
fn equal_v1_v2() {
  expect_interpret_multi_result(
    r#"
      rel S = {(0, 1), (1, 2)}
      rel Q(a, b) = S(a, b), a == a
    "#,
    vec![("Q", vec![(0, 1), (1, 2)].into())],
  )
}

#[test]
fn test_count_with_where_clause() {
  expect_interpret_multi_result(
    r#"
      // There are three classes
      rel classes = {0, 1, 2}

      // There are 6 students, 2 in each class
      rel student = {
        (0, "tom"), (0, "jenny"), // Class 0
        (1, "alice"), (1, "bob"), // Class 1
        (2, "liby"), (2, "john"), // Class 2
      }

      // Each student is enrolled in a course (Math or CS)
      rel enroll = {
        ("tom", "CS"), ("jenny", "Math"), // Class 0
        ("alice", "CS"), ("bob", "CS"), // Class 1
        ("liby", "Math"), ("john", "Math"), // Class 2
      }

      // Count how many student enrolls in CS class in each class
      rel count_enroll_cs_in_class(c, n) :- n = count(s: student(c, s), enroll(s, "CS") where c: classes(c))
    "#,
    vec![("count_enroll_cs_in_class", vec![(0, 1usize), (1, 2), (2, 0)].into())],
  )
}

#[test]
fn test_exists_path_1() {
  expect_interpret_multi_result(
    r#"
      rel edge = {(0, 1), (1, 2)}
      rel path(x, y) = edge(x, y) or (path(x, z) and edge(z, y))
      rel result1(b) = b := exists(path(0, 2))
      rel result2(b) = b := exists(path(0, 3))
    "#,
    vec![("result1", vec![(true,)].into()), ("result2", vec![(false,)].into())],
  )
}

#[test]
fn test_exists_with_where_clause() {
  expect_interpret_multi_result(
    r#"
      // A set of all the shapes
      rel all_shapes = {"cube", "cylinder", "sphere"}

      // Each object has two attributes: color and shape
      rel color = {(0, "red"), (1, "green"), (2, "blue"), (3, "blue")}
      rel shape = {(0, "cube"), (1, "cylinder"), (2, "sphere"), (3, "cube")}

      // Is there a blue object?
      rel exists_blue_obj(b) = b = exists(o: color(o, "blue"))

      // For each shape, is there a blue object of that shape?
      rel exists_blue_obj_of_shape(s, b) :-
        b = exists(o: color(o, "blue"), shape(o, s) where s: all_shapes(s))
    "#,
    vec![
      ("exists_blue_obj", vec![(true,)].into()),
      (
        "exists_blue_obj_of_shape",
        vec![
          ("cube".to_string(), true),
          ("cylinder".to_string(), false),
          ("sphere".to_string(), true),
        ]
        .into(),
      ),
    ],
  )
}

#[test]
fn test_exists_with_where_clause_2() {
  expect_interpret_multi_result(
    r#"
      // A set of all the shapes
      rel all_shapes = {"cube", "cylinder", "sphere"}

      // Each object has two attributes: color and shape
      rel color = {(0, "red"), (1, "green"), (2, "blue"), (3, "blue")}
      rel shape = {(0, "cube"), (1, "cylinder"), (2, "sphere"), (3, "cube")}

      // Is there a blue object?
      rel exists_blue_obj() = exists(o: color(o, "blue"))

      // For each shape, is there a blue object of that shape?
      rel exists_blue_obj_of_shape(s) :- exists(o: color(o, "blue"), shape(o, s) where s: all_shapes(s))
    "#,
    vec![
      ("exists_blue_obj", vec![()].into()),
      (
        "exists_blue_obj_of_shape",
        vec![("cube".to_string(),), ("sphere".to_string(),)].into(),
      ),
    ],
  )
}

#[test]
fn test_not_exists_1() {
  expect_interpret_result(
    r#"
      rel color = {(0, "red"), (1, "green")}
      rel result() :- not exists(o: color(o, "blue"))
    "#,
    ("result", vec![()].into()),
  )
}

#[test]
fn type_cast_to_string_1() {
  expect_interpret_result(
    r#"
    rel r = {1, 2, 3}
    rel s(x as String) = r(x)
  "#,
    ("s", vec![("1".to_string(),), ("2".to_string(),), ("3".to_string(),)]),
  )
}

#[test]
fn implies_1() {
  expect_interpret_result(
    r#"
    rel obj = {1, 2}
    rel color = {(1, "blue"), (2, "red")}
    rel shape = {(1, "cube"), (2, "cube")}

    // The object `o` such that `o` is cube implies that `o` is blue
    rel answer(o) = obj(o) and (shape(o, "cube") => color(o, "blue"))
    "#,
    ("answer", vec![(1,)]),
  )
}

#[test]
fn implies_2() {
  expect_interpret_result(
    r#"
    rel obj = {1, 2}
    rel color = {(1, "blue"), (2, "red")}
    rel shape = {(1, "cube"), (2, "sphere")}

    // The object `o` such that `o` is cube implies that `o` is blue
    rel answer(o) = obj(o) and (shape(o, "cube") => color(o, "blue"))
    "#,
    ("answer", vec![(1,), (2,)]),
  )
}

#[test]
fn has_three_objs_1() {
  expect_interpret_result(
    r#"
    rel obj = {1, 2, 3}
    rel answer(b) = b == (n == 3), n = count(o: obj(o))
    "#,
    ("answer", vec![(true,)]),
  )
}

#[test]
fn has_three_objs_2() {
  expect_interpret_result(
    r#"
    rel obj = {1, 2, 3}
    rel answer(n == 3) = n = count(o: obj(o))
    "#,
    ("answer", vec![(true,)]),
  )
}

#[test]
fn forall_1() {
  expect_interpret_result(
    r#"
    rel color = {(1, "blue"), (2, "red")}
    rel shape = {(1, "cube"), (2, "cube")}

    // For all cube `o`, `o` is blue
    rel answer(b) = b = forall(o: shape(o, "cube") => color(o, "blue"))
    "#,
    ("answer", vec![(false,)]),
  )
}

#[test]
fn forall_2() {
  expect_interpret_result(
    r#"
    rel color = {(1, "blue"), (2, "red")}
    rel shape = {(1, "cube"), (2, "sphere")}

    // For all cube `o`, `o` is blue
    rel answer(b) = b = forall(o: shape(o, "cube") => color(o, "blue"))
    "#,
    ("answer", vec![(true,)]),
  )
}

#[test]
fn forall_3() {
  expect_interpret_result(
    r#"
    rel all_colors = {"blue", "red", "green"}

    // Scene graph
    rel color = {(1, "blue"), (2, "red"), (3, "red")}
    rel shape = {(1, "cube"), (2, "sphere"), (3, "cube")}
    rel material = {(1, "metal"), (2, "metal"), (3, "rubber")}

    // For each color `c`, is all the cube of material rubber?
    rel answer(c, b) = b = forall(o: color(o, c) and shape(o, "cube") => material(o, "rubber") where c: all_colors(c))
    "#,
    (
      "answer",
      vec![
        ("blue".to_string(), false),
        ("red".to_string(), true),
        ("green".to_string(), true),
      ],
    ),
  )
}

#[test]
fn forall_4() {
  expect_interpret_result(
    r#"
    rel all_colors = {"blue", "red", "green"}

    // Scene graph
    rel color = {(1, "blue"), (2, "red"), (3, "red")}
    rel shape = {(1, "cube"), (2, "sphere"), (3, "cube")}
    rel material = {(1, "metal"), (2, "metal"), (3, "rubber")}

    // For each color `c`, is all the cube of material rubber?
    rel answer(c) = forall(o: color(o, c) and shape(o, "cube") => material(o, "rubber") where c: all_colors(c))
    "#,
    ("answer", vec![("red".to_string(),), ("green".to_string(),)]),
  )
}

#[test]
fn string_to_usize() {
  expect_interpret_result(
    r#"
    rel input_string = {"13", "14"}
    rel result(x as usize) = input_string(x)
    "#,
    ("result", vec![(13usize,), (14,)]),
  )
}

#[test]
fn string_to_i32() {
  expect_interpret_result(
    r#"
    rel input_string = {"13", "14"}
    rel result(x as i32) = input_string(x)
    "#,
    ("result", vec![(13i32,), (14,)]),
  )
}

#[test]
fn character_test() {
  expect_interpret_result(
    r#"
    rel chars = {'1', '0', '\t', ' '}
    "#,
    ("chars", vec![('1',), ('0',), ('\t',), (' ',)]),
  )
}

#[test]
fn string_char_at_test_1() {
  expect_interpret_result(
    r#"
    rel result($string_char_at("hello world", 3))
    "#,
    ("result", vec![('l',)]),
  )
}

#[test]
fn string_char_at_test_2() {
  expect_interpret_result(
    r#"
    rel input("1357")
    rel string_char_at(0, $string_char_at(s, 0)) :- input(s), 0 < $string_length(s)
    rel string_char_at(i, $string_char_at(s, i)) :- input(s), i < $string_length(s), string_char_at(i - 1, _)
    "#,
    ("string_char_at", vec![(0usize, '1'), (1, '3'), (2, '5'), (3, '7')]),
  )
}

#[test]
fn string_char_at_failure_1() {
  expect_interpret_empty_result(
    r#"
    rel output($string_char_at("", 0))
    "#,
    "output",
  )
}

#[test]
fn ff_max_1() {
  expect_interpret_result(
    r#"
    rel output($max(0, 1, 2, 3))
    "#,
    ("output", vec![(3,)]),
  )
}

#[test]
fn ff_max_2() {
  expect_interpret_result(
    r#"
    rel R = {1}
    rel S = {3}
    rel output($max(a, b)) = R(a), S(b)
    "#,
    ("output", vec![(3,)]),
  )
}

#[test]
fn const_variable_1() {
  expect_interpret_result(
    r#"
    const VAR1 = 135
    const VAR2 = 246
    rel r(VAR1, VAR2)
    "#,
    ("r", vec![(135, 246)]),
  )
}

#[test]
fn const_variable_2() {
  expect_interpret_result(
    r#"
    const VAR1: u8 = 135
    const VAR2: i32 = 246
    rel r(VAR1, VAR2)
    "#,
    ("r", vec![(135u8, 246i32)]),
  )
}

#[test]
fn const_variable_3() {
  expect_interpret_result(
    r#"
    const VAR1: u8 = 135
    const VAR2: i32 = 246
    rel r = {(VAR1, VAR2)}
    "#,
    ("r", vec![(135u8, 246i32)]),
  )
}

#[test]
fn const_variable_4() {
  expect_interpret_result(
    r#"
    const VAR1: u8 = 135
    const VAR2: i32 = 246
    rel r(VAR1 + 1, VAR2)
    "#,
    ("r", vec![(136u8, 246i32)]),
  )
}

#[test]
fn const_variable_5() {
  expect_interpret_result(
    r#"
    const UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3
    rel r(UP, RIGHT, DOWN, LEFT)
    "#,
    ("r", vec![(0, 1, 2, 3)]),
  )
}

#[test]
fn const_variable_6() {
  expect_interpret_result(
    r#"
    const UP: u8 = 0, RIGHT: u32 = 1, DOWN: i32 = 2, LEFT: usize = 3
    rel r(UP, RIGHT, DOWN, LEFT)
    "#,
    ("r", vec![(0u8, 1u32, 2i32, 3usize)]),
  )
}

#[test]
fn const_variable_7() {
  expect_interpret_result(
    r#"
    type Action = UP | RIGHT | DOWN | LEFT
    rel r(UP, RIGHT, DOWN, LEFT)
    "#,
    ("r", vec![(0usize, 1usize, 2usize, 3usize)]),
  )
}

#[test]
fn const_variable_8() {
  expect_interpret_result(
    r#"
    type Action = UP = 10 | RIGHT | DOWN | LEFT
    rel r(UP, RIGHT, DOWN, LEFT)
    "#,
    ("r", vec![(10usize, 11usize, 12usize, 13usize)]),
  )
}

#[test]
fn const_variable_9() {
  expect_interpret_result(
    r#"
    type Action = UP | RIGHT | DOWN = 10 | LEFT
    rel r(UP, RIGHT, DOWN, LEFT)
    "#,
    ("r", vec![(0usize, 1usize, 10usize, 11usize)]),
  )
}

#[test]
fn const_variable_10() {
  expect_interpret_result(
    r#"
    type Action = UP = 3 | RIGHT | DOWN = 10 | LEFT
    rel r(UP, RIGHT, DOWN, LEFT)
    "#,
    ("r", vec![(3usize, 4usize, 10usize, 11usize)]),
  )
}

#[test]
fn sat_1() {
  let ctx = proofs::ProofsProvenance::<RcFamily>::default();
  expect_interpret_result_with_tag(
    r#"
    type assign(String, bool)

    // Assignments to variables A, B, and C
    rel assign = {1.0::("A", true); 1.0::("A", false)}
    rel assign = {1.0::("B", true); 1.0::("B", false)}
    rel assign = {1.0::("C", true); 1.0::("C", false)}

    // Boolean formula (A and !B) or (B and !C)
    rel bf_var = {(1, "A"), (2, "B"), (3, "B"), (4, "C")}
    rel bf_not = {(5, 2), (6, 4)}
    rel bf_and = {(7, 1, 5), (8, 3, 6)}
    rel bf_or = {(9, 7, 8)}
    rel bf_root = {9}

    // Evaluation
    rel eval_bf(bf, r) :- bf_var(bf, v), assign(v, r)
    rel eval_bf(bf, !r) :- bf_not(bf, c), eval_bf(c, r)
    rel eval_bf(bf, lr && rr) :- bf_and(bf, lbf, rbf), eval_bf(lbf, lr), eval_bf(rbf, rr)
    rel eval_bf(bf, lr || rr) :- bf_or(bf, lbf, rbf), eval_bf(lbf, lr), eval_bf(rbf, rr)
    rel eval(r) :- bf_root(bf), eval_bf(bf, r)
    "#,
    ctx,
    (
      "eval",
      vec![(proofs::Proofs::one(), (false,)), (proofs::Proofs::one(), (true,))],
    ),
    |_, _| true,
  )
}

#[test]
fn sat_2() {
  let ctx = proofs::ProofsProvenance::<RcFamily>::default();
  expect_interpret_result_with_tag(
    r#"
    type assign(String, bool)

    // Assignments to variables A and B
    rel assign = {1.0::("A", true); 1.0::("A", false)}
    rel assign = {1.0::("B", true); 1.0::("B", false)}

    // Boolean formula (A and !A) or (B and !B)
    rel bf_var = {(1, "A"), (2, "B")}
    rel bf_not = {(3, 1), (4, 2)}
    rel bf_and = {(5, 1, 3), (6, 2, 4)}
    rel bf_or = {(7, 5, 6)}
    rel bf_root = {7}

    // Evaluation
    rel eval_bf(bf, r) :- bf_var(bf, v), assign(v, r)
    rel eval_bf(bf, !r) :- bf_not(bf, c), eval_bf(c, r)
    rel eval_bf(bf, lr && rr) :- bf_and(bf, lbf, rbf), eval_bf(lbf, lr), eval_bf(rbf, rr)
    rel eval_bf(bf, lr || rr) :- bf_or(bf, lbf, rbf), eval_bf(lbf, lr), eval_bf(rbf, rr)
    rel eval(r) :- bf_root(bf), eval_bf(bf, r)
    "#,
    ctx,
    ("eval", vec![(proofs::Proofs::one(), (false,))]),
    |_, _| true,
  )
}

#[test]
fn no_nan_1() {
  expect_interpret_result(
    r#"
    rel R = {0.0, 3.0, 5.0}
    rel P = {0.0, 1.0}
    rel Q(a / b) = R(a) and P(b)
    "#,
    ("Q", vec![(0.0f32,), (3.0f32,), (5.0f32,), (std::f32::INFINITY,)]),
  )
}

#[test]
fn string_plus_string_1() {
  expect_interpret_result(
    r#"
    rel first_last_name = {("Alice", "Lee")}
    rel full_name(first + " " + last) = first_last_name(first, last)
    "#,
    ("full_name", vec![("Alice Lee".to_string(),)]),
  )
}

#[test]
fn disjunctive_1() {
  let prov = proofs::ProofsProvenance::<RcFamily>::default();

  // Pre-generate true tags and false tags
  let true_tag = proofs::Proofs::from_proofs(
    vec![
      proofs::Proof::from_facts(vec![0, 1, 2, 4].into_iter()),
      proofs::Proof::from_facts(vec![0, 1, 2, 5].into_iter()),
      proofs::Proof::from_facts(vec![0, 1, 3, 4].into_iter()),
    ]
    .into_iter(),
  );
  let false_tag =
    proofs::Proofs::from_proofs(vec![proofs::Proof::from_facts(vec![0, 1, 3, 5].into_iter())].into_iter());

  // Test
  expect_interpret_result_with_tag(
    r#"
    rel var = {1, 2}
    rel { assign(x, true); assign(x, false) } = var(x)
    rel result(a || b) = assign(1, a) and assign(2, b)
    "#,
    prov,
    ("result", vec![(true_tag, (true,)), (false_tag, (false,))]),
    proofs::Proofs::eq,
  )
}

#[test]
fn escape_single_newline_char() {
  expect_interpret_result(
    r#"
      rel str = {"Hello\nWorld"}
    "#,
    ("str", vec![("Hello\nWorld".to_string(),)]),
  );
}

#[test]
fn escape_multiple_newline_char() {
  expect_interpret_result(
    r#"
      rel str = {"Hello\n\n\nWorld"}
    "#,
    ("str", vec![("Hello\n\n\nWorld".to_string(),)]),
  );
}

#[test]
fn escape_newline_char_end() {
  expect_interpret_result(
    r#"
      rel str = {"Scallop\n"}
    "#,
    ("str", vec![("Scallop\n".to_string(),)]),
  );
}

#[test]
fn escape_newline_char_beginning() {
  expect_interpret_result(
    r#"
      rel str = {"\nScallop"}
    "#,
    ("str", vec![("\nScallop".to_string(),)]),
  );
}

#[test]
fn escape_tab_char() {
  expect_interpret_result(
    r#"
      rel str = {"Hello\tWorld"}
    "#,
    ("str", vec![("Hello\tWorld".to_string(),)]),
  );
}

#[test]
fn escape_null_char() {
  expect_interpret_result(
    r#"
      rel str = {"Null\0"}
    "#,
    ("str", vec![("Null\0".to_string(),)]),
  );
}

#[test]
fn escape_single_quote() {
  expect_interpret_result(
    r#"
      rel str = {"Here is a quote: \'Hi\'"}
    "#,
    ("str", vec![("Here is a quote: \'Hi\'".to_string(),)]),
  );
}

#[test]
fn escape_double_quote() {
  expect_interpret_result(
    r#"
      rel str = {"Here is a quote: \"Hi\""}
    "#,
    ("str", vec![("Here is a quote: \"Hi\"".to_string(),)]),
  );
}

#[test]
fn escape_backslash() {
  expect_interpret_result(
    r#"
      rel str = {"Back \\ Slash"}
    "#,
    ("str", vec![("Back \\ Slash".to_string(),)]),
  );
}

#[test]
fn escape_carriage_return() {
  expect_interpret_result(
    r#"
      rel str = {"Carriage Return\r"}
    "#,
    ("str", vec![("Carriage Return\r".to_string(),)]),
  );
}

// #[test]
// fn escape_unicode() {
//   expect_interpret_result(
//     r#"
//       rel str = {"Thumbs up: \u{1F44D}"}
//     "#,
//     ("str", vec![("Thumbs up: \u{1F44D}".to_string(),)]),
//   );
// }

#[test]
fn escape_emoji_unicode() {
  expect_interpret_result(
    r#"
      rel str = {"Thumbs up: 👍"}
    "#,
    ("str", vec![("Thumbs up: \u{1F44D}".to_string(),)]),
  );
}

#[test]
fn multiline_string_1() {
  expect_interpret_result(
    r#"
      rel str = {"""This
is
a
multiline
string"""}
    "#,
    ("str", vec![("This\nis\na\nmultiline\nstring".to_string(),)]),
  );
}

#[test]
fn indented_multiline_string() {
  expect_interpret_result(
    r#"
      rel str = {"""This
                    is
                    a
                    multiline
                    string"""}
    "#,
    ("str", vec![("This\n                    is\n                    a\n                    multiline\n                    string".to_string(),)]),
  );
}

#[test]
fn escape_unindented_multiline_string_2() {
  expect_interpret_result(
    r#"
      rel str = {"""
This
is
a
multiline
string
      """}
    "#,
    ("str", vec![("\nThis\nis\na\nmultiline\nstring\n      ".to_string(),)]),
  );
}

#[test]
fn mix_multiline_string_before() {
  expect_interpret_result(
    r#"
      rel str = {"""This
is
a
multiline
string""", "A regular string"}
    "#,
    (
      "str",
      vec![
        ("This\nis\na\nmultiline\nstring".to_string(),),
        ("A regular string".to_string(),),
      ],
    ),
  );
}

#[test]
fn mix_multiline_string_after() {
  expect_interpret_result(
    r#"
      rel str = {"First string", """This
is
a
multiline
string"""}
    "#,
    (
      "str",
      vec![
        ("First string".to_string(),),
        ("This\nis\na\nmultiline\nstring".to_string(),),
      ],
    ),
  );
}

#[test]
fn multiple_multiline_string() {
  expect_interpret_result(
    r#"
      rel str = {"""This
is
a
multiline
string""",
"""A
second
multiline
string"""}
    "#,
    (
      "str",
      vec![
        ("This\nis\na\nmultiline\nstring".to_string(),),
        ("A\nsecond\nmultiline\nstring".to_string(),),
      ],
    ),
  );
}

#[test]
fn multiline_string_with_regular_string() {
  expect_interpret_result(
    r#"
      rel str = {"""Here is a multiline string with quote:
"This is a test"
By John Doe"""}
    "#,
    (
      "str",
      vec![("Here is a multiline string with quote:\n\"This is a test\"\nBy John Doe".to_string(),)],
    ),
  );
}

#[test]
fn multiline_string_with_regular_string_2() {
  expect_interpret_result(
    r#"
      rel str = {"""Here is a multiline string with quote:
"This is a test"
By John Doe""", """something else"""}
    "#,
    (
      "str",
      vec![
        ("Here is a multiline string with quote:\n\"This is a test\"\nBy John Doe".to_string(),),
        ("something else".to_string(),),
      ],
    ),
  );
}

#[test]
fn multiline_string_with_multiple_regular_string() {
  expect_interpret_result(
    r#"
      rel str = {"""Here is a multiline string with quote:
"This is a test"
By
"Anonymous"
"""}
    "#,
    (
      "str",
      vec![("Here is a multiline string with quote:\n\"This is a test\"\nBy\n\"Anonymous\"\n".to_string(),)],
    ),
  );
}

#[test]
fn multiline_string_with_escaped_double_quote_string() {
  expect_interpret_result(
    r#"
      rel str = {"""Here is a multiline string with quote:
\"\"This is a test\"\"
By Jane Doe"""}
    "#,
    (
      "str",
      vec![("Here is a multiline string with quote:\n\"\"This is a test\"\"\nBy Jane Doe".to_string(),)],
    ),
  );
}

#[test]
fn multiline_string_with_double_quote_string() {
  expect_interpret_result(
    r#"
      rel str = {"""Here is a multiline string with double quote:
""This is a test""
By Jane Doe"""}
    "#,
    (
      "str",
      vec![("Here is a multiline string with double quote:\n\"\"This is a test\"\"\nBy Jane Doe".to_string(),)],
    ),
  );
}

#[test]
fn multiline_string_with_triple_quote() {
  expect_interpret_result(
    r#"
      rel str = {"""This is not the end
\"\"\"
But this is"""}
    "#,
    ("str", vec![("This is not the end\n\"\"\"\nBut this is".to_string(),)]),
  );
}

#[test]
fn multiline_string_as_single_line() {
  expect_interpret_result(
    r#"
      rel str = {"""This is only one line"""}
    "#,
    ("str", vec![("This is only one line".to_string(),)]),
  );
}

#[test]
fn multiline_string_single_newline_char() {
  expect_interpret_result(
    r#"
      rel str = {"""Hello
\n
World"""}
    "#,
    ("str", vec![("Hello\n\n\nWorld".to_string(),)]),
  );
}

#[test]
fn multiline_string_multiple_newline_char() {
  expect_interpret_result(
    r#"
      rel str = {"""Hello
\n\n\n
World"""}
    "#,
    ("str", vec![("Hello\n\n\n\n\nWorld".to_string(),)]),
  );
}

#[test]
fn multiline_string_newline_char_end() {
  expect_interpret_result(
    r#"
      rel str = {"""Scallop
\n"""}
    "#,
    ("str", vec![("Scallop\n\n".to_string(),)]),
  );
}

#[test]
fn multiline_string_newline_char_beginning() {
  expect_interpret_result(
    r#"
      rel str = {"""\n
Scallop"""}
    "#,
    ("str", vec![("\n\nScallop".to_string(),)]),
  );
}

#[test]
fn multiline_string_tab_char() {
  expect_interpret_result(
    r#"
      rel str = {"""Hello
\tWorld"""}
    "#,
    ("str", vec![("Hello\n\tWorld".to_string(),)]),
  );
}

#[test]
fn multiline_string_null_char() {
  expect_interpret_result(
    r#"
      rel str = {"""Null
\0"""}
    "#,
    ("str", vec![("Null\n\0".to_string(),)]),
  );
}

#[test]
fn multiline_string_single_quote() {
  expect_interpret_result(
    r#"
      rel str = {"""Here is a quote:
\'Hi\'"""}
    "#,
    ("str", vec![("Here is a quote:\n\'Hi\'".to_string(),)]),
  );
}

#[test]
fn multiline_string_double_quote() {
  expect_interpret_result(
    r#"
      rel str = {"""Here is a quote:
\"Hi\""""}
    "#,
    ("str", vec![("Here is a quote:\n\"Hi\"".to_string(),)]),
  );
}

#[test]
fn multiline_string_backslash() {
  expect_interpret_result(
    r#"
      rel str = {"""Back
\\ Slash"""}
    "#,
    ("str", vec![("Back\n\\ Slash".to_string(),)]),
  );
}

#[test]
fn multiline_string_carriage_return() {
  expect_interpret_result(
    r#"
      rel str = {"""Carriage
Return\r"""}
    "#,
    ("str", vec![("Carriage\nReturn\r".to_string(),)]),
  );
}

#[test]
fn multiline_string_emoji_unicode() {
  expect_interpret_result(
    r#"
      rel str = {"""Thumbs up:
👍"""}
    "#,
    ("str", vec![("Thumbs up:\n\u{1F44D}".to_string(),)]),
  );
}
