use colored::*;

use super::super::*;
use super::*;

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct CharLocation {
  pub row: usize,
  pub col: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
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

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct NodeLocation {
  pub offset_span: Span<usize>,
  pub loc_span: Option<Span<CharLocation>>,
  pub id: Option<usize>,
  pub source_id: usize,
}

impl std::hash::Hash for NodeLocation {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    state.write_usize(self.source_id);
    state.write_usize(self.offset_span.start);
    state.write_usize(self.offset_span.end);
  }
}

impl NodeLocation {
  /// When cloning a location, we want to keep everything but not the id.
  pub fn clone_without_id(&self) -> Self {
    Self {
      offset_span: self.offset_span.clone(),
      loc_span: self.loc_span.clone(),
      id: None,
      source_id: self.source_id,
    }
  }

  /// Create a location from a single offset span.
  pub fn from_span(start: usize, end: usize) -> Self {
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

impl std::fmt::Debug for NodeLocation {
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

/// An AST Node trait
pub trait AstNode: Clone {
  /// Obtain a location of the AstNode
  fn location(&self) -> &NodeLocation;

  /// Obtain a mutable location of the AstNode
  fn location_mut(&mut self) -> &mut NodeLocation;

  fn clone_with_loc(&self, loc: NodeLocation) -> Self;
}

#[derive(Clone, Debug, PartialEq, Hash, Serialize)]
pub struct AstNodeWrapper<T> {
  pub _loc: NodeLocation,
  pub _node: T,
}

impl<T: Clone> AstNodeWrapper<T> {
  pub fn clone_without_location_id(&self) -> Self {
    Self {
      _loc: self._loc.clone_without_id(),
      _node: self._node.clone(),
    }
  }

  pub fn location_id(&self) -> Option<usize> {
    self._loc.id.clone()
  }

  pub fn location_source_id(&self) -> usize {
    self._loc.source_id.clone()
  }
}

impl<T: Clone> AstNode for AstNodeWrapper<T> {
  fn location(&self) -> &NodeLocation {
    &self._loc
  }

  fn location_mut(&mut self) -> &mut NodeLocation {
    &mut self._loc
  }

  fn clone_with_loc(&self, loc: NodeLocation) -> Self {
    Self {
      _loc: loc,
      _node: self._node.clone(),
    }
  }
}

pub trait NodeVisitor<N> {
  fn visit(&mut self, node: &N);

  fn visit_mut(&mut self, node: &mut N);
}

#[allow(unused)]
impl<U, V> NodeVisitor<V> for U {
  default fn visit(&mut self, node: &V) {}

  default fn visit_mut(&mut self, node: &mut V) {}
}

macro_rules! impl_node_visitor_tuple {
  ( $($id:ident,)* ) => {
    impl<Node, $($id,)*> NodeVisitor<Node> for ($(&mut $id,)*) {
      default fn visit(&mut self, node: &Node) {
        paste::item! { let ($( [<$id:lower>],)*) = self; }
        $( paste::item! { <$id as NodeVisitor<Node>>::visit([<$id:lower>], node); } )*
      }
      default fn visit_mut(&mut self, node: &mut Node) {
        paste::item! { let ($( [<$id:lower>],)*) = self; }
        $( paste::item! { <$id as NodeVisitor<Node>>::visit_mut([<$id:lower>], node); } )*
      }
    }
  };
}

impl_node_visitor_tuple!(A,);
impl_node_visitor_tuple!(A, B,);
impl_node_visitor_tuple!(A, B, C,);
impl_node_visitor_tuple!(A, B, C, D,);
impl_node_visitor_tuple!(A, B, C, D, E,);
impl_node_visitor_tuple!(A, B, C, D, E, F,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U,);
impl_node_visitor_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V,);

#[allow(unused)]
pub trait AstWalker {
  fn walk<V>(&self, v: &mut V);

  fn walk_mut<V>(&mut self, v: &mut V);
}

macro_rules! derive_ast_walker {
  ($ty:ty) => {
    impl AstWalker for $ty {
      fn walk<V>(&self, _: &mut V) {}

      fn walk_mut<V>(&mut self, _: &mut V) {}
    }
  };
}

derive_ast_walker!(i8);
derive_ast_walker!(i16);
derive_ast_walker!(i32);
derive_ast_walker!(i64);
derive_ast_walker!(i128);
derive_ast_walker!(isize);
derive_ast_walker!(u8);
derive_ast_walker!(u16);
derive_ast_walker!(u32);
derive_ast_walker!(u64);
derive_ast_walker!(u128);
derive_ast_walker!(usize);
derive_ast_walker!(f32);
derive_ast_walker!(f64);
derive_ast_walker!(bool);
derive_ast_walker!(char);
derive_ast_walker!(String);
derive_ast_walker!(crate::common::input_tag::DynamicInputTag);
derive_ast_walker!(crate::common::binary_op::BinaryOp);

impl<T> AstWalker for Vec<T>
where
  T: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    for child in self {
      child.walk(v)
    }
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    for child in self {
      child.walk_mut(v)
    }
  }
}

impl<T> AstWalker for Option<T>
where
  T: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    if let Some(n) = self {
      n.walk(v)
    }
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    if let Some(n) = self {
      n.walk_mut(v)
    }
  }
}

impl<T> AstWalker for Box<T>
where
  T: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    (&**self).walk(v)
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    (&mut **self).walk_mut(v)
  }
}

impl<A> AstWalker for (A,)
where
  A: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    self.0.walk(v);
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    self.0.walk_mut(v);
  }
}

impl<A, B> AstWalker for (A, B)
where
  A: AstWalker,
  B: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    self.0.walk(v);
    self.1.walk(v);
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    self.0.walk_mut(v);
    self.1.walk_mut(v);
  }
}

impl<A, B, C> AstWalker for (A, B, C)
where
  A: AstWalker,
  B: AstWalker,
  C: AstWalker,
{
  fn walk<V>(&self, v: &mut V) {
    self.0.walk(v);
    self.1.walk(v);
    self.2.walk(v);
  }

  fn walk_mut<V>(&mut self, v: &mut V) {
    self.0.walk_mut(v);
    self.1.walk_mut(v);
    self.2.walk_mut(v);
  }
}
