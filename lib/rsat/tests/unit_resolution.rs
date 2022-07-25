// use rsat::*;

// #[test]
// fn rsat_unit_resolution_1() {
//   let a = Literal::positive(Variable::new(0));
//   let b = Literal::positive(Variable::new(1));
//   let c = Literal::positive(Variable::new(2));
//   let cnf = vec![vec![a, b.negate()], vec![c.negate()]];
//   let s1 = Solver::new(cnf);
//   match s1.decide_literal(a) {
//     Ok(s2) => match s2.decide_literal(b) {
//       Ok(s3) => match s3.decide_literal(c) {
//         Ok(_) => panic!("Should not be able to decide c"),
//         Err(clause) => { println!("{:?}", clause) },
//       }
//       Err(clause) => panic!("Cannot decode b: {:?}", clause),
//     }
//     Err(clause) => panic!("Cannot decide a: {:?}", clause),
//   }
// }

// (a \/ b) /\ (~a \/ ~b) /\ (a \/ ~b) /\ (~b \/ a)
// #[test]
// fn rsat_2() {
//   let a = Variable::new(0);
//   let b = Variable::new(1);
//   let ap = Literal::positive(a);
//   let bp = Literal::positive(b);
//   let cnf = vec![vec![ap, bp], vec![ap.negate(), bp.negate()], vec![ap, bp.negate()], vec![ap.negate(), bp]];
//   let s1 = Solver::new(cnf);
//   s1.solve_with_variable_order(&[a, b]).expect_err("Expected UNSAT");
// }

// #[test]
// fn rsat_3() {
//   let a = Variable::new(0);
//   let b = Variable::new(1);
//   let c = Variable::new(2);
//   let ap = Literal::positive(a);
//   let bp = Literal::positive(b);
//   let cp = Literal::positive(c);
//   let cnf = vec![vec![ap, bp], vec![bp.negate(), cp], vec![cp.negate(), ap.negate()]];
//   let s1 = Solver::new(cnf);
//   s1.solve_with_variable_order(&[a, b, c]).unwrap();
// }

// #[test]
// fn rsat_4() {
//   let a = Variable::new(0);
//   let b = Variable::new(1);
//   let c = Variable::new(2);
//   let ap = Literal::positive(a);
//   let bp = Literal::positive(b);
//   let cp = Literal::positive(c);
//   let cnf = vec![vec![ap, bp.negate()], vec![bp, cp.negate()], vec![cp, ap.negate()]];
//   let s1 = Solver::new(cnf);
//   s1.solve_with_variable_order(&[b, a, c]).unwrap();
// }
