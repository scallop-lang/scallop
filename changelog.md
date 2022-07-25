# v0.1.4, July 24, 2022

`v0.1.4` introduces multiple bug fixes and new features
- forall aggregator (`forall`)
- implies operator (`=>`)
- topkproofs supporting negation and aggregation
- Multiple bugs fixed

# v0.1.3, June 1, 2022

`v0.1.3` introduces multiple bug fixes and few additional features
- WASM Scallop is now able to be compiled
- Fixed the ordering of the relations in the output
- Multiple bugs fixed

# v0.1.2, May 26, 2022

`v0.1.2` introduces a bug fix in the compiler.

# v0.1.1, May 20, 2022

`v0.1.1` introduces
- the implementation of many more aggregators for different provenances
- bug fixed where min/max aggregators are not functioning correctly
- bug fixed where atoms like `pred(a, a)` might cause compilation issue
- adding customization to termination controlled by provenances
- adding `--query` command line argument to `scli`
- allow for disjunction in many provenances that wasn't allowed to

And many more experimental improvements associated with these changes.
