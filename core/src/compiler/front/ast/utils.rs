use colored::*;

use super::super::*;

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Location {
  pub row: usize,
  pub col: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span<T> {
  pub start: T,
  pub end: T,
}

impl<T> Span<T> {
  pub fn new(start: T, end: T) -> Self {
    Self { start, end }
  }
}

impl Span<usize> {
  pub fn is_default(&self) -> bool {
    self.start == 0 && self.end == 0
  }

  pub fn length(&self) -> usize {
    self.end - self.start
  }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct AstNodeLocation {
  pub offset_span: Span<usize>,
  pub loc_span: Option<Span<Location>>,
  pub id: Option<usize>,
  pub source_id: usize,
}

impl std::hash::Hash for AstNodeLocation {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    state.write_usize(self.source_id);
    state.write_usize(self.offset_span.start);
    state.write_usize(self.offset_span.end);
  }
}

impl AstNodeLocation {
  pub fn clone_without_id(&self) -> Self {
    Self {
      offset_span: self.offset_span.clone(),
      loc_span: self.loc_span.clone(),
      id: None,
      source_id: self.source_id,
    }
  }

  pub fn from_offset_span(start: usize, end: usize) -> Self {
    Self {
      offset_span: Span::new(start, end),
      loc_span: None,
      id: None,
      source_id: 0,
    }
  }

  pub fn error_prefix(&self) -> String {
    match &self.loc_span {
      Some(loc_span) => format!(
        "[{}:{}-{}:{}] ",
        loc_span.start.row, loc_span.start.col, loc_span.end.row, loc_span.end.col
      ),
      None => {
        if self.offset_span.is_default() {
          String::new()
        } else {
          format!("[{}-{}] ", self.offset_span.start, self.offset_span.end)
        }
      }
    }
  }

  pub fn report(&self, src: &Sources) -> String {
    self.report_with_marker_color(src, Color::Red)
  }

  pub fn report_warning(&self, src: &Sources) -> String {
    self.report_with_marker_color(src, Color::Yellow)
  }

  pub fn report_with_marker_color(&self, src: &Sources, color: Color) -> String {
    let arrow = "-->".yellow().bold();
    let bar = "|".yellow().bold();

    // Print the title
    let s = &src.sources[self.source_id];
    let mut result = if let Some(name) = s.name() {
      format!(" {} {}:\n", arrow, name)
    } else {
      format!("")
    };

    // Gather the lines to print
    let lines = match &self.loc_span {
      Some(span) => (span.start.row..=span.end.row)
        .map(|line_num| {
          let line = s.line(line_num);
          let (prefix_len, highlight_len) = {
            if line_num == span.start.row && line_num == span.end.row {
              (span.start.col, span.end.col - span.start.col)
            } else if line_num == span.start.row {
              (span.start.col, line.len() - span.start.col)
            } else if line_num == span.end.row {
              (0, span.end.col)
            } else {
              (0, 0)
            }
          };
          let highlight = highlight_str(prefix_len, highlight_len);
          (s.line_name(line_num), line, highlight)
        })
        .collect(),
      None => {
        let prefix_len = self.offset_span.start;
        let highlight_len = self.offset_span.end - self.offset_span.start;
        let highlight = highlight_str(prefix_len, highlight_len);
        vec![(s.line_name(0), s.line(0), highlight)]
      }
    };

    // Calculate padding size
    const PADDING: usize = 2;
    let max_padding_length = lines.iter().map(|(n, _, _)| n.len()).max().unwrap();
    let padding_length = PADDING + max_padding_length;
    let whole_padding_str = (0..padding_length).map(|_| " ").collect::<String>();

    // Print each line
    for (line_num, line, highlight) in lines {
      let padding_len = padding_length - line_num.len();
      let padding_str = (0..padding_len).map(|_| " ").collect::<String>();
      result += &format!("{}{} {} {}\n", padding_str, line_num.yellow().bold(), bar, line);
      result += &format!("{} {} {}", whole_padding_str, bar, highlight.color(color));
    }

    // Return
    result
  }
}

fn highlight_str(prefix_len: usize, highlight_len: usize) -> String {
  let prefix = (0..prefix_len).map(|_| " ").collect::<String>();
  let main = (0..highlight_len).map(|_| "^").collect::<String>();
  format!("{}{}", prefix, main)
}

impl std::fmt::Debug for AstNodeLocation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match (&self.id, &self.loc_span) {
      (None, None) => {
        write!(f, "[{}-{}]", self.offset_span.start, self.offset_span.end)
      }
      (Some(id), None) => {
        write!(f, "[#{}, {}-{}]", id, self.offset_span.start, self.offset_span.end)
      }
      (None, Some(loc_span)) => {
        write!(
          f,
          "[{}:{}-{}:{}]",
          loc_span.start.row, loc_span.start.col, loc_span.end.row, loc_span.end.col,
        )
      }
      (Some(id), Some(loc_span)) => {
        write!(
          f,
          "[#{}, {}:{}-{}:{}]",
          id, loc_span.start.row, loc_span.start.col, loc_span.end.row, loc_span.end.col,
        )
      }
    }
  }
}

#[derive(Clone, PartialEq, Eq)]
pub struct AstNode<N> {
  pub loc: AstNodeLocation,
  pub node: N,
}

impl<N: std::fmt::Debug> std::fmt::Debug for AstNode<N> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("{:?} @{:?}", self.node, self.loc))
  }
}

impl<N> AstNode<N> {
  pub fn new(loc: AstNodeLocation, node: N) -> Self {
    Self { loc, node }
  }

  pub fn default(n: N) -> Self {
    Self {
      loc: AstNodeLocation::default(),
      node: n,
    }
  }

  pub fn from_span(start: usize, end: usize, node: N) -> Self {
    Self {
      loc: AstNodeLocation::from_offset_span(start, end),
      node,
    }
  }

  pub fn id(&self) -> usize {
    self.loc.id.unwrap_or(0)
  }

  pub fn source_id(&self) -> usize {
    self.loc.source_id
  }

  pub fn clone_with_new_location(&self) -> Self
  where
    N: Clone,
  {
    Self {
      loc: AstNodeLocation::default(),
      node: self.node.clone(),
    }
  }
}

impl<N> From<N> for AstNode<N> {
  fn from(n: N) -> Self {
    Self::default(n)
  }
}

pub trait WithLocation {
  fn location(&self) -> &AstNodeLocation;

  fn location_mut(&mut self) -> &mut AstNodeLocation;
}

impl<N> WithLocation for AstNode<N> {
  fn location(&self) -> &AstNodeLocation {
    &self.loc
  }

  fn location_mut(&mut self) -> &mut AstNodeLocation {
    &mut self.loc
  }
}
