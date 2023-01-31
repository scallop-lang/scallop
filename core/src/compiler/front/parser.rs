use super::*;

pub fn str_to_item(s: &str) -> Result<Item, ParserError> {
  let parser = grammar::ItemParser::new();
  parser.parse(s).map_err(ParserError::from)
}

pub fn str_to_items(s: &str) -> Result<Vec<Item>, ParserError> {
  let parser = grammar::ItemsParser::new();
  parser.parse(s).map_err(ParserError::from)
}

pub fn str_to_relation_type(s: &str) -> Result<Vec<Item>, ParserError> {
  let parser = grammar::RelationTypeParser::new();
  parser.parse(s).map(|q| q.into()).map_err(ParserError::from)
}

pub fn str_to_rule(s: &str) -> Result<Vec<Item>, ParserError> {
  let parser = grammar::RuleParser::new();
  parser.parse(s).map(|q| q.into()).map_err(ParserError::from)
}

pub fn str_to_query(s: &str) -> Result<Vec<Item>, ParserError> {
  let parser = grammar::QueryParser::new();
  parser.parse(s).map(|q| q.into()).map_err(ParserError::from)
}

pub type RawParserError<'a> = lalrpop_util::ParseError<usize, grammar::Token<'a>, &'static str>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserError {
  pub message: String,
  pub source_id: usize,
  pub source_name: Option<String>,
  pub location_span: Option<(usize, usize)>,
  pub location_point: Option<usize>,
}

impl ParserError {
  pub fn set_source_id(&mut self, id: usize) {
    self.source_id = id;
  }

  pub fn set_source_name(&mut self, name: String) {
    self.source_name = Some(name);
  }
}

impl std::fmt::Display for ParserError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if let Some(source_name) = &self.source_name {
      f.write_fmt(format_args!("Syntax error in {}: {}", source_name, self.message))
    } else {
      f.write_fmt(format_args!("Syntax error: {}", self.message))
    }
  }
}

impl FrontCompileErrorTrait for ParserError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, sources: &Sources) -> String {
    if let Some(source_name) = &self.source_name {
      let begin = format!("Syntax error in {}: {}", source_name, self.message);

      // First compute offset span
      let offset_span = if let Some((l, r)) = self.location_span {
        Some((l, r))
      } else if let Some(p) = self.location_point {
        Some((p, p + 1))
      } else {
        None
      };

      // Turn offset span into locations
      if let Some(offset_span) = offset_span {
        let source = &sources[self.source_id];
        let annotator = LocationSpanAnnotator {
          row_offset_length: (0..source.num_rows()).map(|i| source.row_offset_length(i)).collect(),
        };
        let ast_loc = AstNodeLocation {
          offset_span: Span::new(offset_span.0, offset_span.1),
          loc_span: Some(Span::new(
            annotator.row_col_of_offset(&offset_span.0),
            annotator.row_col_of_offset(&offset_span.1),
          )),
          id: None,
          source_id: self.source_id,
        };
        format!("{}\n{}", begin, ast_loc.report(sources))
      } else {
        format!("Syntax error: {}", self.message)
      }
    } else {
      format!("Syntax error: {}", self.message)
    }
  }
}

impl<'a> From<RawParserError<'a>> for ParserError {
  fn from(err: RawParserError<'a>) -> Self {
    // Generate base error
    let mut e = ParserError {
      source_id: 0,
      message: format!("{}", err),
      source_name: None,
      location_point: None,
      location_span: None,
    };

    // Add location information
    match err {
      RawParserError::UnrecognizedEOF { location, .. } => e.location_point = Some(location),
      RawParserError::InvalidToken { location, .. } => e.location_point = Some(location),
      RawParserError::UnrecognizedToken { token, .. } => e.location_span = Some((token.0, token.2)),
      _ => {}
    }

    // Return the error
    e
  }
}
