# Scallop Compiler

## Structure

The compiler of Scallop has three levels of internal representations (IR):

### `front`

- Directly reflecting the surface abstract syntax tree (AST)
  - A parser will turn the
  - Each rule body is a formula (involving disjunctions, conjunctions, negations)
  - Aggregations can have sub-formulas
  - Can have complex expressions
- In this stage, source level information will be stored in the AST for easy error reporting
- In this stage, we will perform the following analysis
  - Type inference
  - Boundness analysis
  - Aggregation analysis
  - Wildcard analysis
- Additionally, we apply the following transformations
  - De-sugar rule probability
  - De-sugar query with atoms

### `back`

- A more concise version of Datalog
  - Each rule is a conjunction of literals
  - Complex expressions are flattened into unitary constraints and assignments
  - Subformulas in aggregation are turned into auxilliary rules
  - Source level information is removed
- In this stage, we apply the following optimizations
  - Equality propagation
  - Constant folding/propagation
  - Remove false rules
  - Demand transformation
  - Turn empty rules into facts
- We will perform the following checks in the back IR
  - Stratification

### `ram`

- The low level relational algebra machine
- Close to executable program

## Note

- All temporary variable and temporary relations will have a `'#'` in their name.
