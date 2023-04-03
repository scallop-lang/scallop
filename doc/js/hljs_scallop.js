hljs.registerLanguage("pen", (hljs) => ({
  name: "Scallop",
  aliases: ["scl", "scallop"],
  keywords: {
    keyword: "import type rel relation query if then else where",
    type: "i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64 bool char String",
    literal: "true false",
    built_in: "count sum prod min max exists forall unique top",
  },
  contains: [
    hljs.C_LINE_COMMENT_MODE,
    hljs.C_BLOCK_COMMENT_MODE,
    {
      className: "string",
      variants: [
        hljs.QUOTE_STRING_MODE,
      ]
    },
    {
      className: "number",
      variants: [
        {
          begin: hljs.C_NUMBER_RE + "[i]",
          relevance: 1
        },
        hljs.C_NUMBER_MODE
      ]
    }
  ],
}));

hljs.initHighlightingOnLoad();
