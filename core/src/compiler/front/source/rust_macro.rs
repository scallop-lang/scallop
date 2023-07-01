use proc_macro2::TokenStream;

use super::*;

#[derive(Clone, Debug)]
pub struct RustMacroSource {
  pub first_line_num: usize,
  pub source_lines: Vec<String>,
  pub source: String,
}

impl RustMacroSource {
  pub fn new(tokens: TokenStream) -> Self {
    let (first_line_num, source_lines) = token_stream_to_src_lines(tokens);
    let source = source_lines.clone().join("\n");
    Self {
      first_line_num,
      source_lines,
      source,
    }
  }
}

impl Source for RustMacroSource {
  fn content(&self) -> &str {
    &self.source
  }

  fn name(&self) -> Option<&str> {
    None
  }

  fn line(&self, line_num: usize) -> &str {
    &self.source_lines[line_num - self.first_line_num]
  }

  fn line_name(&self, line_num: usize) -> String {
    format!("{}", line_num)
  }

  fn num_rows(&self) -> usize {
    self.first_line_num + self.source_lines.len()
  }

  fn row_offset_length(&self, row: usize) -> (usize, usize) {
    if row < self.first_line_num {
      (0, 0)
    } else {
      let curr_row = row - self.first_line_num;
      let offset = self
        .source_lines
        .iter()
        .take(curr_row)
        .fold(0, |agg, l| agg + l.len() + 1);
      let length = self.source_lines[curr_row].len();
      (offset, length)
    }
  }
}

fn token_stream_to_src_lines(tokens: TokenStream) -> (usize, Vec<String>) {
  let mut str_tokens = vec![];
  populate_str_tokens(&mut str_tokens, tokens);

  // Return empty if there is no token
  if str_tokens.is_empty() {
    return (0, vec![]);
  }

  // First get the start line
  let first_line_num = str_tokens[0].1.line;

  // Then accumulate the line
  let mut curr_line_num = first_line_num;
  let mut src_lines = vec![];
  let mut curr_line = String::new();

  // Iterate through tokens to recover the source code
  for (token, start_loc, offset) in str_tokens {
    // Move on to the next line where span lies in
    while start_loc.line > curr_line_num {
      src_lines.push(curr_line);
      curr_line = String::new();
      curr_line_num += 1;
    }

    // Fill in the column prefixes
    if start_loc.column as i32 + offset > curr_line.len() as i32 {
      let spaces = vec![' '; start_loc.column - curr_line.len()]
        .into_iter()
        .collect::<String>();
      curr_line += &spaces;
    }

    // Fill in the token
    curr_line += &token;
  }

  // Add the final line
  src_lines.push(curr_line);

  // Return
  (first_line_num, src_lines)
}

fn populate_str_tokens(str_tokens: &mut Vec<(String, LineColumn, i32)>, tokens: TokenStream) {
  let mut tokens_iter = tokens.into_iter();
  loop {
    if let Some(token) = tokens_iter.next() {
      match token {
        proc_macro2::TokenTree::Group(g) => {
          let span = g.span();
          let (open, offset, close) = match g.delimiter() {
            proc_macro2::Delimiter::Parenthesis => ("(".to_string(), -1, ")".to_string()),
            proc_macro2::Delimiter::Brace => ("{".to_string(), -1, "}".to_string()),
            proc_macro2::Delimiter::Bracket => ("[".to_string(), -1, "]".to_string()),
            proc_macro2::Delimiter::None => ("".to_string(), 0, "".to_string()),
          };
          str_tokens.push((open, span_start_line_column(&span), 0));

          populate_str_tokens(str_tokens, g.stream());

          str_tokens.push((close, span_end_line_column(&span), offset));
        }
        proc_macro2::TokenTree::Ident(i) => {
          let span = i.span();
          str_tokens.push((format!("{}", i), span_start_line_column(&span), 0));
        }
        proc_macro2::TokenTree::Punct(p) => {
          let mut curr_p = p.clone();
          let mut span = curr_p.span();
          let mut op = format!("{}", curr_p.as_char());
          while curr_p.spacing() == proc_macro2::Spacing::Joint {
            let next_token = tokens_iter.next().unwrap();
            if let proc_macro2::TokenTree::Punct(next_p) = next_token {
              span = span.join(next_p.span()).unwrap();
              op += &next_p.as_char().to_string();
              curr_p = next_p;
            } else {
              panic!("Should not happen");
            }
          }
          str_tokens.push((op, span_start_line_column(&span), 0));
        }
        proc_macro2::TokenTree::Literal(l) => {
          let span = l.span();
          str_tokens.push((format!("{}", l), span_start_line_column(&span), 0));
        }
      }
    } else {
      break;
    }
  }
}

#[derive(Debug, Clone, Copy)]
struct LineColumn {
  line: usize,
  column: usize,
}

fn span_start_line_column(span: &proc_macro2::Span) -> LineColumn {
  let s = span.unwrap().start();
  LineColumn {
    line: s.line,
    column: s.column,
  }
}

fn span_end_line_column(span: &proc_macro2::Span) -> LineColumn {
  let s = span.unwrap().end();
  LineColumn {
    line: s.line,
    column: s.column,
  }
}
