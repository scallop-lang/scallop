use scallop_core::runtime::provenance::*;
use scallop_core::utils::RcFamily;

mod diff {
  use super::*;
  use scallop_core::runtime::provenance::diff_top_bottom_k_clauses::*;

  #[test]
  fn test_diff_top_bottom_k_clauses_1() {
    let mut ctx = DiffTopBottomKClausesContext::<(), RcFamily>::new(1);

    // Create a few tags
    let a = ctx.tagging_fn((0.9, ()).into());
    let b = ctx.tagging_fn((0.8, ()).into());
    let c = ctx.tagging_fn((0.2, ()).into());
    let d = ctx.tagging_fn((0.1, ()).into());

    // First proof
    let ab = ctx.mult(&a, &b);
    let cd = ctx.mult(&c, &d);
    let ab_or_cd = ctx.add(&ab, &cd);

    // Should only contain a and b
    assert_eq!(ab_or_cd.clauses.len(), 1);
    assert_eq!(ab_or_cd.clauses[0].literals.len(), 2);
    assert_eq!(ab_or_cd.clauses[0].literals[0], Literal::Pos(0));
    assert_eq!(ab_or_cd.clauses[0].literals[1], Literal::Pos(1));
  }

  #[test]
  fn test_diff_top_bottom_k_clauses_2() {
    let mut ctx = DiffTopBottomKClausesContext::<(), RcFamily>::new(1);

    // Create a few tags
    let a = ctx.tagging_fn((0.1, ()).into());
    let b = ctx.tagging_fn((0.2, ()).into());
    let c = ctx.tagging_fn((0.8, ()).into());
    let d = ctx.tagging_fn((0.9, ()).into());

    // First proof
    let nanb = ctx.mult(&ctx.negate(&a).unwrap(), &ctx.negate(&b).unwrap());
    let cd = ctx.mult(&ctx.negate(&c).unwrap(), &ctx.negate(&d).unwrap());
    let nanb_or_cd = ctx.add(&nanb, &cd);

    // Should only contain a and b
    println!("{:?}", nanb_or_cd);
  }
}

mod normal {
  use super::*;

  #[test]
  fn test_top_bottom_k_clauses_1() {
    let k = 2;
    let mut ctx = BasicCNFDNFClausesContext::new();
    ctx.probabilities.extend(vec![
      0.01, 0.01, 0.01, 0.01, 0.95, 0.01, 0.01, 0.01, 0.95, 0.01, 0.95, 0.01, 0.01, 0.01, 0.01,
    ]);
    let t1 = CNFDNFFormula::cnf(vec![
      Clause::singleton(Literal::Neg(8)),
      Clause::singleton(Literal::Neg(4)),
    ]);
    let t2 = CNFDNFFormula::dnf(vec![
      Clause::singleton(Literal::Pos(13)),
      Clause::singleton(Literal::Pos(14)),
    ]);
    let r = ctx.top_bottom_k_mult(&t1, &t2, k);
    println!("{:?}", r);
  }

  #[test]
  fn test_top_bottom_k_clauses_2() {
    let k = 2;
    let mut ctx = BasicCNFDNFClausesContext::new();
    ctx.probabilities.extend(vec![
      0.01, 0.01, 0.01, 0.01, 0.95, 0.01, 0.01, 0.01, 0.95, 0.01, 0.95, 0.01, 0.01, 0.01, 0.01,
    ]);
    let t1 = CNFDNFFormula::dnf(vec![
      Clause::new(vec![Literal::Pos(4), Literal::Pos(8)]),
      Clause::new(vec![Literal::Pos(3), Literal::Pos(8)]),
    ]);
    println!("{:?}", ctx.dnf2cnf_k(&t1.clauses, k));
    let t2 = CNFDNFFormula::cnf(vec![
      Clause::singleton(Literal::Neg(13)),
      Clause::singleton(Literal::Neg(14)),
    ]);
    let r = ctx.top_bottom_k_mult(&t1, &t2, k);
    println!("{:?}", r);
  }
}
