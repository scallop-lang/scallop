use ram::simplify;

#[test]
fn test_filter_cascade_1() {
  assert_eq!(simplify("(filter (filter ?d ?a) ?b)"), "(filter ?d (&& ?a ?b))")
}

#[test]
fn test_filter_cascade_2() {
  assert_eq!(simplify("(filter (filter (filter ?d ?a) ?b) ?c)"), "(filter ?d (&& ?c (&& ?a ?b)))")
}

#[test]
fn test_project_cascade_1() {
  assert_eq!(simplify("(project (project ?d ?a) ?b)"), "(project ?d (apply ?b ?a))")
}

#[test]
fn test_project_cascade_2() {
  assert_eq!(simplify(r#"
  (project
    (project
      ?d
      (tuple-cons
        (index-cons 1 index-nil)
        (tuple-cons
          (index-cons 0 index-nil)
          tuple-nil
        )
      )
    )
    (-
      (index-cons 0 index-nil)
      (index-cons 1 index-nil)
    )
  )"#), "(project ?d (- (index-cons 1 index-nil) (index-cons 0 index-nil)))")
}
