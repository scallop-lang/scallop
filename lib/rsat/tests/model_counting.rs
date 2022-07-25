// use rsat::*;

// #[test]
// fn rsat_mc_2() {
//   let a = Variable::new(0);
//   let b = Variable::new(1);
//   let ap = Literal::positive(a);
//   let bp = Literal::positive(b);
//   let cnf = vec![vec![ap, bp], vec![ap.negate(), bp.negate()], vec![ap, bp.negate()], vec![ap.negate(), bp]];
//   let s1 = Solver::new(cnf);
//   assert_eq!(s1.model_counting_with_variable_order(&[a, b]).unwrap_or(0), 0)
// }

// #[test]
// fn rsat_mc_3() {
//   let a = Variable::new(0);
//   let b = Variable::new(1);
//   let c = Variable::new(2);
//   let ap = Literal::positive(a);
//   let bp = Literal::positive(b);
//   let cp = Literal::positive(c);
//   let cnf = vec![vec![ap, bp], vec![bp.negate(), cp], vec![cp.negate(), ap.negate()]];
//   let s1 = Solver::new(cnf);
//   assert_eq!(s1.model_counting_with_variable_order(&[a, b, c]).unwrap_or(0), 2)
// }
