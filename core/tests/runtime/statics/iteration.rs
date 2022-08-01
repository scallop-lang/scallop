use scallop_core::runtime::provenance::*;
use scallop_core::runtime::statics::*;
use scallop_core::testing::*;

#[test]
fn test_static_iter_edge_path() {
  let mut prov = unit::UnitContext::default();
  let mut iter = StaticIteration::<unit::Unit>::new(&mut prov);

  // Add relations
  let edge = iter.create_relation::<(usize, usize)>();
  let path = iter.create_relation::<(usize, usize)>();
  let path_inv = iter.create_relation::<(usize, usize)>();

  // Add facts
  edge.insert_untagged(iter.provenance_context, vec![(0, 1), (1, 2)]);

  // Main loop
  while iter.changed() || iter.is_first_iteration() {
    iter.insert_dataflow(&path, &edge);
    iter.insert_dataflow(&path_inv, dataflow::project(&path, |t| (t.1, t.0)));
    iter.insert_dataflow(&path, dataflow::project(iter.join(&path_inv, &edge), |t| (t.1, t.2)));
    iter.step();
  }

  // Peek the path
  let path = iter.complete(&path);
  expect_static_collection(&path, vec![(0, 1), (1, 2), (0, 2)]);
}

#[test]
fn test_static_iter_odd_even_3() {
  let mut prov = unit::UnitContext::default();

  struct Stratum0Result<C: ProvenanceContext> {
    numbers: StaticCollection<(i32,), C::Tag>,
    _numbers_perm_0_: StaticCollection<(i32, ()), C::Tag>,
    _numbers_perm_0_0: StaticCollection<(i32, i32), C::Tag>,
  }

  fn stratum_0<C: ProvenanceContext>(prov: &mut C) -> Stratum0Result<C> {
    let mut iter = StaticIteration::<C::Tag>::new(prov);
    let numbers = iter.create_relation::<(i32,)>();
    let _numbers_perm_0_ = iter.create_relation::<(i32, ())>();
    let _numbers_perm_0_0 = iter.create_relation::<(i32, i32)>();
    numbers.insert_untagged(iter.provenance_context, vec![(0,)]);
    while iter.changed() || iter.is_first_iteration() {
      iter.insert_dataflow(
        &numbers,
        dataflow::project(
          dataflow::project(
            dataflow::filter(dataflow::project(&numbers, |t| (t.0 + 1,)), |t| t.0 <= 10),
            |t| t.0,
          ),
          |t| (t,),
        ),
      );
      iter.insert_dataflow(&_numbers_perm_0_0, dataflow::project(&numbers, |t| (t.0, t.0)));
      iter.insert_dataflow(&_numbers_perm_0_, dataflow::project(&numbers, |t| (t.0, ())));
      iter.step();
    }
    Stratum0Result {
      numbers: iter.complete(&numbers),
      _numbers_perm_0_: iter.complete(&_numbers_perm_0_),
      _numbers_perm_0_0: iter.complete(&_numbers_perm_0_0),
    }
  }

  struct Stratum1Result<C: ProvenanceContext> {
    odd: StaticCollection<(i32,), C::Tag>,
  }

  fn stratum_1<C: ProvenanceContext>(prov: &mut C, stratum_0_result: &Stratum0Result<C>) -> Stratum1Result<C> {
    let mut iter = StaticIteration::<C::Tag>::new(prov);
    let _temp_0 = iter.create_relation::<(i32, i32)>();
    let odd = iter.create_relation::<(i32,)>();
    let _odd_perm_0 = iter.create_relation::<(i32, ())>();
    while iter.changed() || iter.is_first_iteration() {
      iter.insert_dataflow(
        &_temp_0,
        dataflow::project(
          dataflow::collection(&stratum_0_result._numbers_perm_0_0, iter.is_first_iteration()),
          |t| (t.0 - 2, t.0),
        ),
      );
      iter.insert_dataflow(
        &odd,
        dataflow::project(
          dataflow::project(
            dataflow::find(
              dataflow::collection(&stratum_0_result._numbers_perm_0_, iter.is_first_iteration()),
              1,
            ),
            |t| (t,),
          ),
          |_| (1,),
        ),
      );
      iter.insert_dataflow(
        &odd,
        dataflow::project(dataflow::project(iter.join(&_temp_0, &_odd_perm_0), |t| t.1), |t| (t,)),
      );
      iter.insert_dataflow(&_odd_perm_0, dataflow::project(&odd, |t| (t.0, ())));
      iter.step();
    }
    Stratum1Result {
      odd: iter.complete(&odd),
    }
  }

  struct Stratum2Result<C: ProvenanceContext> {
    even: StaticCollection<(i32,), C::Tag>,
  }

  fn stratum_2<C: ProvenanceContext>(
    prov: &mut C,
    stratum_0_result: &Stratum0Result<C>,
    stratum_1_result: &Stratum1Result<C>,
  ) -> Stratum2Result<C> {
    let mut iter = StaticIteration::<C::Tag>::new(prov);
    let even = iter.create_relation::<(i32,)>();
    while iter.changed() || iter.is_first_iteration() {
      iter.insert_dataflow(
        &even,
        dataflow::project(
          dataflow::project(
            iter.difference(
              dataflow::collection(&stratum_0_result.numbers, iter.is_first_iteration()),
              dataflow::collection(&stratum_1_result.odd, iter.is_first_iteration()),
            ),
            |t| t.0,
          ),
          |t| (t,),
        ),
      );
      iter.step();
    }
    Stratum2Result {
      even: iter.complete(&even),
    }
  }

  // Execute
  let stratum_0_result = stratum_0(&mut prov);
  let stratum_1_result = stratum_1(&mut prov, &stratum_0_result);
  let stratum_2_result = stratum_2(&mut prov, &stratum_0_result, &stratum_1_result);

  // Check result
  expect_static_collection(&stratum_1_result.odd, vec![(1,), (3,), (5,), (7,), (9,)]);
  expect_static_collection(&stratum_2_result.even, vec![(0,), (2,), (4,), (6,), (8,), (10,)]);
}

#[test]
fn test_static_out_degree_join() {
  struct Stratum0Result<C: ProvenanceContext> {
    node_temp: StaticCollection<(usize, ()), C::Tag>,
    edge: StaticCollection<(usize, usize), C::Tag>,
  }

  fn stratum_0<C: ProvenanceContext>(prov: &mut C) -> Stratum0Result<C> {
    let mut iter = StaticIteration::<C::Tag>::new(prov);

    // Add relations
    let node = iter.create_relation::<(usize,)>();
    let node_temp = iter.create_relation::<(usize, ())>();
    let edge = iter.create_relation::<(usize, usize)>();

    // Add facts
    node.insert_untagged(iter.provenance_context, vec![(0,), (1,), (2,)]);
    edge.insert_untagged(iter.provenance_context, vec![(0, 1), (1, 2)]);

    // Main loop
    while iter.changed() || iter.is_first_iteration() {
      iter.insert_dataflow(&node_temp, dataflow::project(&node, |(x,)| (x, ())));
      iter.step();
    }

    // Result
    Stratum0Result {
      node_temp: iter.complete(&node_temp),
      edge: iter.complete(&edge),
    }
  }

  struct Stratum1Result<C: ProvenanceContext> {
    out_degree: StaticCollection<(usize, usize), C::Tag>,
  }

  fn stratum_1<C: ProvenanceContext>(prov: &mut C, stratum_0_result: &Stratum0Result<C>) -> Stratum1Result<C> {
    let mut iter = StaticIteration::<C::Tag>::new(prov);

    // Add relations
    let out_degree = iter.create_relation::<(usize, usize)>();

    // Main loop
    while iter.changed() || iter.is_first_iteration() {
      iter.insert_dataflow(
        &out_degree,
        dataflow::project(
          iter.aggregate_join_group(
            CountAggregator::new(),
            dataflow::collection(&stratum_0_result.node_temp, iter.is_first_iteration()),
            dataflow::collection(&stratum_0_result.edge, iter.is_first_iteration()),
          ),
          |(x, (), y)| (x, y),
        ),
      );
      iter.step();
    }

    Stratum1Result {
      out_degree: iter.complete(&out_degree),
    }
  }

  let mut prov = unit::UnitContext::default();

  // Execute
  let stratum_0_result = stratum_0(&mut prov);
  let stratum_1_result = stratum_1(&mut prov, &stratum_0_result);

  // Check result
  expect_static_collection(&stratum_1_result.out_degree, vec![(0, 1), (1, 1), (2, 0)]);
}

#[test]
fn test_static_out_degree_implicit_group() {
  struct Stratum0Result<C: ProvenanceContext> {
    edge: StaticCollection<(usize, usize), C::Tag>,
  }

  fn stratum_0<C: ProvenanceContext>(prov: &mut C) -> Stratum0Result<C> {
    let mut iter = StaticIteration::<C::Tag>::new(prov);

    // Add relations
    let edge = iter.create_relation::<(usize, usize)>();

    // Add facts
    edge.insert_untagged(iter.provenance_context, vec![(0, 1), (1, 2)]);

    // Main loop
    while iter.changed() || iter.is_first_iteration() {
      iter.step();
    }

    // Result
    Stratum0Result {
      edge: iter.complete(&edge),
    }
  }

  struct Stratum1Result<C: ProvenanceContext> {
    out_degree: StaticCollection<(usize, usize), C::Tag>,
  }

  fn stratum_1<C: ProvenanceContext>(prov: &mut C, stratum_0_result: &Stratum0Result<C>) -> Stratum1Result<C> {
    let mut iter = StaticIteration::<C::Tag>::new(prov);

    // Add relations
    let out_degree = iter.create_relation::<(usize, usize)>();

    // Main loop
    while iter.changed() || iter.is_first_iteration() {
      iter.insert_dataflow(
        &out_degree,
        iter.aggregate_implicit_group(
          CountAggregator::new(),
          dataflow::collection(&stratum_0_result.edge, iter.is_first_iteration()),
        ),
      );
      iter.step();
    }

    Stratum1Result {
      out_degree: iter.complete(&out_degree),
    }
  }

  let mut prov = unit::UnitContext::default();

  // Execute
  let stratum_0_result = stratum_0(&mut prov);
  let stratum_1_result = stratum_1(&mut prov, &stratum_0_result);

  // Check result
  expect_static_collection(&stratum_1_result.out_degree, vec![(0, 1), (1, 1)]);
}

#[test]
fn test_static_num_edges() {
  struct Stratum0Result<C: ProvenanceContext> {
    edge: StaticCollection<(usize, usize), C::Tag>,
  }

  fn stratum_0<C: ProvenanceContext>(prov: &mut C) -> Stratum0Result<C> {
    let mut iter = StaticIteration::<C::Tag>::new(prov);

    // Add relations
    let edge = iter.create_relation::<(usize, usize)>();

    // Add facts
    edge.insert_untagged(iter.provenance_context, vec![(0, 1), (1, 2), (2, 3)]);

    // Main loop
    while iter.changed() || iter.is_first_iteration() {
      iter.step();
    }

    // Result
    Stratum0Result {
      edge: iter.complete(&edge),
    }
  }

  struct Stratum1Result<C: ProvenanceContext> {
    num_edges: StaticCollection<usize, C::Tag>,
  }

  fn stratum_1<C: ProvenanceContext>(prov: &mut C, stratum_0_result: &Stratum0Result<C>) -> Stratum1Result<C> {
    let mut iter = StaticIteration::<C::Tag>::new(prov);

    // Add relations
    let num_edges = iter.create_relation::<usize>();

    // Main loop
    while iter.changed() || iter.is_first_iteration() {
      iter.insert_dataflow(
        &num_edges,
        iter.aggregate(
          CountAggregator::new(),
          dataflow::collection(&stratum_0_result.edge, iter.is_first_iteration()),
        ),
      );
      iter.step();
    }

    Stratum1Result {
      num_edges: iter.complete(&num_edges),
    }
  }

  let mut prov = unit::UnitContext::default();

  // Execute
  let stratum_0_result = stratum_0(&mut prov);
  let stratum_1_result = stratum_1(&mut prov, &stratum_0_result);

  // Check result
  expect_static_collection(&stratum_1_result.num_edges, vec![3]);
}
