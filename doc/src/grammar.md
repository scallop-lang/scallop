# Full Scallop Grammar

```
SCALLOP_PROGRAM ::= ITEM*

ITEM ::= TYPE_DECL
       | RELATION_DECL
       | CONST_DECL
       | QUERY_DECL

TYPE ::= u8 | u16 | u32 | u64 | u128 | usize
       | i8 | i16 | i32 | i64 | i128 | isize
       | f32 | f64 | char | bool
       | String
       | CUSTOM_TYPE_NAME

TYPE_DECL ::= type CUSTOM_TYPE_NAME = TYPE
            | type CUSTOM_TYPE_NAME <: TYPE
            | type ENUM_TYPE_NAME = VARIANT1 [= VAL1] | VARIANT2 [= VAL2] | ...
            | type ADT_TYPE_NAME = CONSTRUCTOR1(TYPE*) | CONSTRUCTOR2(TYPE*) | ...
            | type RELATION_NAME(TYPE*)
            | type RELATION_NAME(VAR1: TYPE1, VAR2: TYPE2, ...)
            | type $FUNCTION_NAME(VAR1: TYPE1, VAR2: TYPE2, ...) -> TYPE_RET

CONST_DECL ::= const CONSTANT_NAME : TYPE = CONSTANT
             | const CONSTANT_NAME1 [: TYPE1] = CONSTANT1, CONSTANT_NAME2 [: TYPE2] = CONSTANT2, ...

RELATION_DECL ::= FACT_DECL
                | FACTS_SET_DECL
                | RULE_DECL

CONSTANT ::= true | false | NUMBER_LITERAL | STRING_LITERAL

CONST_TUPLE ::= CONSTANT | (CONSTANT1, CONSTANT2, ...)

FOREIGN_FN ::= hash | string_length | string_concat | substring | abs

BIN_OP ::= + | - | * | / | % | == | != | <= | < | >= | > | && | || | ^

UNARY_OP ::= ! | -

CONST_EXPR ::= CONSTANT
             | CONST_EXPR BIN_OP CONST_EXPR | UNARY_OP CONST_EXPR
             | $ FOREIGN_FN(CONST_EXPR*)
             | if CONST_EXPR then CONST_EXPR else CONST_EXPR
             | ( CONST_EXPR )

TAG ::= true | false | NUMBER_LITERAL  // true/false is for boolean tags; NUMBER_LITERAL is used for probabilities

FACT_DECL ::= rel RELATION_NAME(CONST_EXPR*)         // Untagged fact
            | rel TAG :: RELATION_NAME(CONST_EXPR*)  // Tagged fact

FACTS_SET_DECL ::= rel RELATION_NAME = {CONST_TUPLE1, CONST_TUPLE2, ...}                  // Untagged tuples
                 | rel RELATION_NAME = {TAG1 :: CONST_TUPLE1, TAG2 :: CONST_TUPLE2, ...}  // Tagged tuples
                 | rel RELATION_NAME = {TAG1 :: CONST_TUPLE1; TAG2 :: CONST_TUPLE2; ...}  // Tagged tuples forming annotated disjunction

EXPR ::= VARIABLE
       | CONSTANT
       | EXPR BIN_OP EXPR | UNARY_OP EXPR
       | new CONSTRUCTOR(EXPR*)
       | $ FOREIGN_FN(EXPR*)
       | if EXPR then EXPR else EXPR
       | ( EXPR )

ATOM ::= RELATION_NAME(EXPR*)

RULE_DECL ::= rel ATOM :- FORMULA | rel ATOM = FORMULA                // Normal rule
            | rel TAG :: ATOM :- FORMULA | rel TAG :: ATOM = FORMULA  // Tagged rule

FORMULA ::= ATOM
          | not ATOM | ~ ATOM                                                   // negation
          | FORMULA1, FORMULA2, ... | FORMULA and FORMULA | FORMULA /\ FORMULA  // conjunction
          | FORMULA or FORMULA | FORMULA \/ FORMULA                             // disjunction
          | FORMULA implies FORMULA | FORMULA => FORMULA                        // implies
          | case VARIABLE is ENTITY
          | CONSTRAINT
          | AGGREGATION
          | ( FORMULA )

ENTITY ::= CONSTRUCTOR(ENTITY*)

CONSTRAINT ::= EXPR // When expression returns a boolean value

AGGREGATOR ::= count | sum | prod | min | max | exists | forall | unique

AGGREGATION ::= VAR* = AGGREGATOR(VAR* : FORMULA)                             // Normal aggregation
              | VAR* = AGGREGATOR(VAR* : FORMULA where VAR* : FORMULA)        // Aggregation with group-by condition
              | VAR* = AGGREGATOR[VAR*](VAR* : FORMULA)                       // Aggregation with arg (only applied to AGGREGATOR = min or max)
              | VAR* = AGGREGATOR[VAR*](VAR* : FORMULA where VAR* : FORMULA)  // Aggregation with arg and group-by condition (only applied to AGGREGATOR = min or max)
              | forall(VAR* : FORMULA)
              | exists(VAR* : FORMULA)

QUERY_DECL ::= query RELATION_NAME
             | query ATOM
```
