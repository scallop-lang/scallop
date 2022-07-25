use super::*;

#[derive(Clone, Debug)]
pub enum BoundnessDependency {
  /// Argument to a relation
  RelationArg(Loc),

  /// Constant loc, is bounded
  Constant(Loc),

  /// General binary operation, if the first two are bounded, the third
  /// op1, op2, op1 X op2
  BinaryOp(Loc, Loc, Loc),

  /// If one is bounded, the other is bounded
  /// op1, op2, op1 X op2
  ConstraintEquality(Loc, Loc),

  /// a +/- b = c, if any two are bounded, the remaining is bounded
  AddSub(Loc, Loc, Loc),

  /// General unary operation, if the first is bounded, the second is bounded
  /// op1, ~op1
  UnaryOp(Loc, Loc),

  /// If-then-else expression, if the first three (cond, then_br, else_br)
  /// is bounded, the last one is bounded
  IfThenElseOp(Loc, Loc, Loc, Loc),
}
