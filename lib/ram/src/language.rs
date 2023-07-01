use egg::{rewrite as rw, *};

use super::tuple_type::TupleType;

// Define the RAM language
define_language! {
  pub enum Ram {
    // Relational Predicate
    Predicate(String, Id),

    // Relational Algebra Operations
    "empty" = Empty,
    "filter" = Filter([Id; 2]),
    "project" = Project([Id; 2]),
    "sorted" = Sorted(Id),
    "product" = Product([Id; 2]),
    "join" = Join([Id; 2]),

    // Tuple operations
    "apply" = Apply([Id; 2]),
    "tuple-cons" = TupleCons([Id; 2]),
    "tuple-nil" = TupleNil,

    // Indexing operations
    "index" = Index(Id),
    "index-cons" = IndexCons([Id; 2]),
    "index-nil" = IndexNil,

    // Value operations
    "+" = Add([Id; 2]),
    "-" = Sub([Id; 2]),
    "*" = Mult([Id; 2]),
    "/" = Div([Id; 2]),
    "&&" = And([Id; 2]),
    "||" = Or([Id; 2]),
    "!" = Not(Id),

    // Any symbol
    Bool(bool),
    Number(i32),
    Symbol(Symbol),
  }
}

pub type EGraph = egg::EGraph<Ram, ()>;

fn var(s: &str) -> Var {
  s.parse().unwrap()
}

fn is_constant(_v: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
  move |_egraph, _, _subst| false
}

/// All the rewrite rules for the language
pub fn ram_rewrite_rules() -> Vec<Rewrite<Ram, ()>> {
  vec![
    // Relational level rewrites
    rw!("filter-cascade"; "(filter (filter ?d ?a) ?b)" => "(filter ?d (&& ?a ?b))"),
    rw!("filter-true"; "(filter ?d true)" => "?d"),
    rw!("filter-false"; "(filter ?d false)" => "empty"),
    rw!("project-cascade"; "(project (project ?d ?a) ?b)" => "(project ?d (apply ?b ?a))"),
    rw!("product-transpose"; "(product ?a ?b)" => "(project (product ?b ?a) (tuple-cons (index 1) (tuple-cons (index-cons 0 index-nil) tuple-nil))))"),
    rw!("join-transpose"; "(join ?a ?b)" => "(project (join ?b ?a) (tuple-cons (index 0) (tuple-cons (index 2) (tuple-cons (index 1) tuple-nil))))"),
    // Tuple level application rewrites
    rw!("access-nil"; "(apply index-nil ?a)" => "?a"),
    rw!("access-tuple-base"; "(apply (index-cons 0 ?x) (tuple-cons ?a ?b))" => "(apply ?x ?a)"),
    rw!("access-tuple-ind"; "(apply (index-cons ?n ?x) (tuple-cons ?a ?b))" => "(apply (index-cons (- ?n 1) ?x) ?b)"),
    rw!("apply-tuple-nil"; "(apply tuple-nil ?a)" => "tuple-nil"),
    rw!("apply-tuple-cons"; "(apply (tuple-cons ?a ?b) ?c)" => "(tuple-cons (apply ?a ?c) (apply ?b ?c))"),
    // Expression level application rewrites
    rw!("apply-add"; "(apply (+ ?a ?b) ?t)" => "(+ (apply ?a ?t) (apply ?b ?t))"),
    rw!("apply-sub"; "(apply (- ?a ?b) ?t)" => "(- (apply ?a ?t) (apply ?b ?t))"),
    rw!("apply-mult"; "(apply (* ?a ?b) ?t)" => "(* (apply ?a ?t) (apply ?b ?t))"),
    rw!("apply-div"; "(apply (/ ?a ?b) ?t)" => "(/ (apply ?a ?t) (apply ?b ?t))"),
    rw!("apply-and"; "(apply (&& ?a ?b) ?t)" => "(&& (apply ?a ?t) (apply ?b ?t))"),
    rw!("apply-or"; "(apply (|| ?a ?b) ?t)" => "(|| (apply ?a ?t) (apply ?b ?t))"),
    rw!("apply-not"; "(apply (! ?a) ?t)" => "(! (apply ?a ?t))"),
    rw!("apply-const"; "(apply ?e ?t)" => "?e" if is_constant(var("?e"))),
    // Value level rewrites
    rw!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
    rw!("add-identity"; "(+ ?a 0)" => "?a"),
    rw!("mult-comm"; "(* ?a ?b)" => "(* ?b ?a)"),
    rw!("mult-identity"; "(* ?a 1)" => "?a"),
    rw!("and-comm"; "(&& ?a ?b)" => "(&& ?b ?a)"),
    rw!("and-identity"; "(&& ?a true)" => "?a"),
    rw!("and-idempotent"; "(&& ?a ?a)" => "?a"),
    rw!("or-comm"; "(|| ?a ?b)" => "(|| ?b ?a)"),
    rw!("or-identity"; "(|| ?a false)" => "?a"),
    rw!("or-idempotent"; "(|| ?a ?a)" => "?a"),
    rw!("not-true"; "(! true)" => "false"),
    rw!("not-false"; "(! false)" => "true"),
    rw!("not-not"; "(! (! ?a))" => "?a"),
    // Simple arithmetic rewrites for index calculations
    rw!("index-desugar"; "(index ?x)" => "(index-cons ?x index-nil)"),
    rw!("dec-1"; "(- 1 1)" => "0"),
    rw!("dec-2"; "(- 2 1)" => "1"),
    rw!("dec-3"; "(- 3 1)" => "2"),
    rw!("dec-4"; "(- 4 1)" => "3"),
    rw!("dec-5"; "(- 5 1)" => "4"),
  ]
}

struct RamCostFunction;

impl CostFunction<Ram> for RamCostFunction {
  type Cost = i32;

  fn cost<C>(&mut self, enode: &Ram, mut costs: C) -> Self::Cost
  where
    C: FnMut(Id) -> Self::Cost,
  {
    let op_cost = match enode {
      Ram::Empty => 0,
      Ram::Filter(_) => 100,
      Ram::Project(_) => 100,
      Ram::Product(_) => 100,
      Ram::Join(_) => 100,
      Ram::Sorted(_) => 500,
      Ram::Apply(_) => 10,
      Ram::Index(_) => 10,
      Ram::TupleCons(_) => 0,
      Ram::TupleNil => 0,
      Ram::IndexCons(_) => 0,
      Ram::IndexNil => 0,
      _ => 1,
    };
    enode.fold(op_cost, |sum, id| sum + costs(id))
  }
}

pub struct RamNodeData {
  pub tuple_type: Option<TupleType>,
}

/// parse an expression, simplify it using egg, and pretty print it back out
pub fn simplify(s: &str) -> String {
  // parse the expression, the type annotation tells it which Language to use
  let expr: RecExpr<Ram> = s.parse().unwrap();

  // simplify the expression using a Runner, which creates an e-graph with
  // the given expression and runs the given rules over it
  let rules = ram_rewrite_rules();
  let runner = Runner::default().with_expr(&expr).run(&rules);

  // the Runner knows which e-class the expression given with `with_expr` is in
  let root = runner.roots[0];

  // use an Extractor to pick the best element of the root eclass
  let extractor = Extractor::new(&runner.egraph, RamCostFunction);
  let (_, best) = extractor.find_best(root);
  best.to_string()
}
