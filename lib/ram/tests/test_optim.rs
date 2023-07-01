use ram::simplify;

#[test]
fn test_filter_cascade_1() {
  assert_eq!(simplify("(filter (filter ?d ?a) ?b)"), "(filter ?d (&& ?a ?b))")
}

#[test]
fn test_filter_cascade_2() {
  assert_eq!(
    simplify("(filter (filter (filter ?d ?a) ?b) ?c)"),
    "(filter ?d (&& ?c (&& ?a ?b)))"
  )
}

#[test]
fn test_project_cascade_1() {
  assert_eq!(simplify("(project (project ?d ?a) ?b)"), "(project ?d (apply ?b ?a))")
}

#[test]
fn test_project_cascade_2() {
  assert_eq!(
    simplify(
      r#"
  (project
    (project
      ?d
      (tuple-cons
        (index 1)
        (tuple-cons
          (index 0)
          tuple-nil
        )
      )
    )
    (-
      (index 0)
      (index 1)
    )
  )"#
    ),
    simplify("(project ?d (- (index 1) (index 0)))")
  )
}
