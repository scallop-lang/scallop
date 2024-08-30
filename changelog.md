# v0.2.4, Aug 30, 2024

- Rule tags can now be expressions with potential reference to local variables: `rel 1/n::head() = body(n)`
- Allowing for sparse gradient computation inside Scallopy to minimize memory footprint
- Allowing users to specify per-datapoint output mapping inside Scallopy
- Adding destructor syntax so that ADTs can be used in a more idiomatic way
- Unifying the behavior of integer overflow inside Scallop
- Multiple bugs fixed

# v0.2.3, Jun 23, 2024

# v0.2.2, Oct 25, 2023

- Adding `wmc_with_disjunctions` option for provenances that deal with boolean formulas for more accurate probability estimation
- Adding foreign aggregate interface
- Added aggregators such as `avg`, `enumerate`, `normalize`, `weighted_sum`, and `sort`
- Adding syntax sugar for aggregation
- Unknown attribute will now be flagged and reported during compile time
- Fixed Scallopy Extension compatibility issue with Python 3.8

# v0.2.1, Sep 12, 2023

- Democratizing foreign functions and foreign predicates so that they can be implemented in Python
- Adding foreign attributes which are higher-order functions
- Adding `scallop-ext` the extension library and multiple Scallop plugins, including `scallop-gpt`, `scallop-clip`, and so on.
- Fixed multiple bugs related foreign predicate computation
- Adding `count!` aggregator for non-probabilistic operation
- Fixed sum and product aggregator so that they can accept additional argument
- Multiple bugs fixed; performance improved

# v0.2.0, Jun 11, 2023

- Fixing CSV loading and its performance; adding new modes to specify `keys`
- Adding `Symbol` type to the language
- Adding algebraic data types to the language and supports for entities
- Adding tensors to the language which can be accessed from `scallopy`
- Adding `scallop` CLI (command-line interface) with OpenAI plugins for invoking LLMs
- Adding more documentations
- Multiple bugs fixed

# v0.1.9, Apr 24, 2023

- Supporting (partial) disjunctive Datalog
- Fixed custom provenance's default implementations and dispatcher fallback
- Adding new negation syntax to `exists` and `forall` expression
- Multiple bugs fixed

# v0.1.8, Mar 27, 2023

- Add foreign predicates including soft comparisons,
- Add `DateTime` and `Duration` support
- Add input-mapping support for multi-dimensional inputs in `scallopy`
- Fixing floating point support by rejecting `NaN` values
- Adding back iteration-limit to runtime environment
- Add Scallop book repository

# v0.1.7, Jan 12, 2023

- Better integration with Extensional Databases (EDB) and memory optimizations
- Better handling of mutual exclusive facts in probabilistic/differentiable reasoning
- Add categorical-k and top-k samplers
- Add foreign function interface available to Python through `scallopy`
- Add syntax sugars for `forall` and `exists` aggregators

# v0.1.6, Nov 17, 2022

`v0.1.6` introduces many improvements and features
- `scallopy` forward function interface for better ergonomics
- Better provenance context interface (saturation, sampling, etc.)
- Multiple bug fixes on provenance structures
- Better support for magic-set transformations on tagged-semantics

# v0.1.5, August 21, 2022

`v0.1.5` introduces JIT compilation and many more features
- JIT compilation in `scallopy` (`jit=True`)
- `scallopy` forward function can output multiple relations instead of just one
- Add standard library functions such as `string_length`, `hash`, etc.
- Add more provenance structures such as `diffnandmultprob`, `diffaddminprob`, etc.
- Systematic internal monitoring and `--debug-tag` options

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
