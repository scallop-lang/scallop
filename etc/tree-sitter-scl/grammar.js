module.exports = grammar({
  name: 'Scallop',
  word: $ => $.name,
  extras: $ => [
    $.whitespace,
    $.inline_comment,
    $.block_comment,
  ],
  rules: {
    top_level: $ => repeat($.item),

    // Tokens
    name: $ => /[a-zA-Z][a-zA-Z_0-9]*/,
    int: $ => /-?[0-9]+/,
    float: $ => /-?\d+(\.\d+)(e-?\d+)?/,
    string: $ => /"[^"]*"/,
    bool: $ => choice("true", "false"),

    // Extras
    whitespace: $ => /\s/,
    inline_comment: $ => /\/\/[^\n\r]*[\n\r]*/,
    block_comment: $ => /\/\*([^\*]*\*+[^\*/])*([^\*]*\*+|[^\*])*\*\//,

    // Basics
    identifier: $ => $.name,
    integer: $ => $.int,
    probability: $ => $.float,
    constant: $ => choice(
      $.bool,
      $.integer,
      $.float,
      $.string,
    ),

    // Attributes
    attribute_pos_arg: $ => $.constant,
    attribute_kw_arg: $ => seq($.identifier, "=", $.constant),
    attribute_arg: $ => choice($.attribute_pos_arg, $.attribute_kw_arg),
    attribute: $ => seq("@", $.identifier, optional(seq("(", separated($.attribute_arg, ","), ")"))),
    attributes: $ => repeat1($.attribute),

    // Type
    type: $ => choice(
      "i8", "i16", "i32", "i64", "i128", "isize",
      "u8", "u16", "u32", "u64", "u128", "usize",
      "f32", "f64", "char", "bool",
      "&str", "String", "Rc<String>",
      $.identifier,
    ),

    // Type declaration
    subtype_decl: $ => seq(optional($.attributes), "type", $.identifier, "<:", $.type),
    alias_type_decl: $ => seq(optional($.attributes), "type", $.identifier, "=", $.type),
    arg_type_binding: $ => choice(seq($.identifier, ":", $.type), $.type),
    relation_type: $ => seq($.identifier, "(", separated($.arg_type_binding, ","), ")"),
    relation_type_decl: $ => seq(optional($.attributes), "type", $.relation_type),
    type_decl: $ => choice($.subtype_decl, $.alias_type_decl, $.relation_type_decl),

    // Input declaration
    input_decl: $ => seq(optional($.attributes), "input", $.identifier, "(", separated($.arg_type_binding, ","), ")"),

    // Relation declaration
    define_symbol: $ => choice(":-", "="),
    relation_keyword: $ => choice("relation", "rel"),
    relation_decl: $ => choice(
      $.constant_set_decl,
      $.fact_decl,
      $.rule_decl,
    ),

    // == Constant set
    constant_tuple: $ => choice($.constant, seq("(", separated($.constant, ","), ")")),
    constant_set_tuple: $ => seq(optional(seq($.probability, "::")), $.constant_tuple),
    constant_set: $ => choice(
      seq("{", separated($.constant_set_tuple, ","), "}"),
      seq("{", at_least_two_separated_strict($.constant_set_tuple, ";"), "}"),
    ),
    constant_set_decl: $ => seq(optional($.attributes), $.relation_keyword, $.identifier, $.define_symbol, $.constant_set),

    // == Fact
    fact_decl: $ => seq(optional($.attributes), $.relation_keyword, optional(seq($.probability, "::")), $.atom),

    // == Rule
    rule: $ => seq($.atom, $.define_symbol, $.formula),
    rule_decl: $ => seq(optional($.attributes), $.relation_keyword, optional(seq($.probability, "::")), $.rule),

    // ==== Expression
    expr: $ => $.if_then_else_expr,
    if_then_else: $ => choice(
      seq("if", $.and_or_expr, "then", $.if_then_else_expr, "else", $.if_then_else_expr),
      seq($.and_or_expr, "?", $.if_then_else_expr, ":", $.if_then_else_expr),
    ),
    if_then_else_expr: $ => choice($.if_then_else, $.and_or_expr),
    and_or: $ => seq($.and_or_expr, choice("&&", "||",  "^"), $.comparison_expr),
    and_or_expr: $ => choice($.and_or, $.comparison_expr),
    comparison: $ => seq($.add_sub_expr, choice("==", "!=", "<", "<=", ">", ">="), $.add_sub_expr),
    comparison_expr: $ => choice($.comparison, $.add_sub_expr),
    add_sub: $ => seq($.add_sub_expr, choice("+", "-"), $.mul_div_mod_expr),
    add_sub_expr: $ => choice($.add_sub, $.mul_div_mod_expr),
    mul_div_mod: $ => seq($.mul_div_mod_expr, choice("*", "/", "%"), $.unary_expr),
    mul_div_mod_expr: $ => choice($.mul_div_mod, $.unary_expr),
    unary: $ => choice(
      seq(choice("+", "-", "!"), $.unit_expr),
      seq($.unit_expr, "as", $.type),
    ),
    unary_expr: $ => choice($.unary, $.unit_expr),
    complex_expr: $ => choice(
      $.and_or,
      $.comparison,
      $.add_sub,
      $.mul_div_mod,
      $.unary,
    ),
    unit_expr: $ => choice(
      seq("(", $.complex_expr, ")"),
      $.wildcard,
      $.constant,
      $.variable,
    ),
    wildcard: $ => "_",
    variable: $ => $.identifier,
    variable_binding: $ => choice(
      $.identifier,
      seq("(", $.identifier, ":", $.type, ")"),
    ),
    variable_or_wildcard: $ => choice($.wildcard, $.variable),

    // ==== Complex Formula
    formula: $ => $.conj_disj_formula,
    conj_disj_formula: $ => choice($.comma_conjunction_formula, $.disjunction_formula),
    comma_conjunction: $ => at_least_two_separated_strict($.neg_atom_formula, ","),
    comma_conjunction_formula: $ => $.comma_conjunction,
    disjunction: $ => at_least_two_separated_strict($.conjunction_formula, choice("\\/", "or")),
    disjunction_formula: $ => choice($.disjunction, $.conjunction_formula),
    conjunction: $ => at_least_two_separated_strict($.neg_atom_formula, choice("/\\", "and")),
    conjunction_formula: $ => choice($.conjunction, $.neg_atom_formula),
    neg_atom: $ => seq("~", $.atom),
    neg_atom_formula: $ => choice($.neg_atom, $.unit_formula),
    complex_formula: $ => choice(
      $.comma_conjunction_formula,
      $.disjunction,
      $.conjunction,
      $.neg_atom,
    ),

    // ==== Unit Formula
    atom: $ => seq($.identifier, "(", separated($.expr, ","), ")"),
    reduce_op: $ => choice("count", "sum", "prod", "min", "max"),
    reduce_args: $ => seq("[", separated($.variable, ","), "]"),
    reduce: $ => choice(
      seq(
        $.variable_or_wildcard,
        "=",
        $.reduce_op,
        optional($.reduce_args),
        "(",
        at_least_one_separated($.variable_binding, ","),
        ":",
        $.formula,
        ")",
      ),
      seq(
        "(",
        at_least_two_separated($.variable_or_wildcard, ","),
        ")",
        "=",
        $.reduce_op,
        optional($.reduce_args),
        "(",
        at_least_one_separated($.variable_binding, ","),
        ":",
        $.formula,
        ")",
      ),
    ),
    constraint: $ => choice($.comparison, $.unary),
    unit_formula: $ => choice(
      seq("(", $.complex_formula, ")"),
      $.constraint,
      $.atom,
      $.reduce,
    ),

    // == Import declaration
    import_decl: $ => seq(optional($.attributes), "import", $.string),

    // == Query declaration
    query: $ => choice($.identifier, $.atom),
    query_decl: $ => seq(optional($.attributes), choice("query", "output"), $.query),

    // Item
    item: $ => choice(
      $.type_decl,
      $.input_decl,
      $.relation_decl,
      $.import_decl,
      $.query_decl,
    ),
  },
});

function separated(token, separator) {
  return seq(repeat(seq(token, separator)), optional(token));
}

function separated_strict(token, separator) {
  return seq(repeat(seq(token, separator)), token);
}

function at_least_one_separated(token, separator) {
  return seq(token, repeat(seq(separator, token)), optional(separator));
}

function at_least_one_separated_strict(token, separator) {
  return seq(token, repeat(seq(separator, token)));
}

function at_least_two_separated(token, separator) {
  return seq(token, repeat1(seq(separator, token)), optional(separator));
}

function at_least_two_separated_strict(token, separator) {
  return seq(token, repeat1(seq(separator, token)));
}
