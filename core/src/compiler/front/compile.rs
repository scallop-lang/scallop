use std::collections::*;
use std::fs;
use std::path::PathBuf;

use super::analysis::*;
use super::analyzers::*;
use super::*;
use crate::common::tuple_type::TupleType;
use crate::common::value_type::ValueType;
use crate::utils::CopyOnWrite;

#[derive(Clone, Debug, Copy)]
pub struct SourceId(usize);

#[derive(Clone, Debug)]
pub struct FrontContext {
  pub sources: Sources,
  pub items: Items,

  // Import
  pub imported_files: HashSet<PathBuf>,

  // Analysis
  pub node_id_annotator: NodeIdAnnotator,
  pub analysis: CopyOnWrite<Analysis>,
}

impl FrontContext {
  pub fn new() -> Self {
    Self {
      sources: Sources::new(),
      items: Vec::new(),

      // Import
      imported_files: HashSet::new(),

      // Annotator
      node_id_annotator: NodeIdAnnotator::new(),
      analysis: CopyOnWrite::new(Analysis::new()),
    }
  }

  pub fn dump_ir(&self) {
    for item in &self.items {
      println!("{}", item);
    }
  }

  pub fn compile_source<S: Source>(
    &mut self,
    s: S,
  ) -> Result<SourceId, FrontErrorReportingContext> {
    self.compile_source_with_parser(s, parser::str_to_items)
  }

  pub fn compile_relation<S>(&mut self, s: S) -> Result<SourceId, FrontErrorReportingContext>
  where
    S: Source,
  {
    self.compile_source_with_parser(s, parser::str_to_relation_type)
  }

  pub fn compile_relation_with_annotator<S, A>(
    &mut self,
    s: S,
    annotator: A,
  ) -> Result<SourceId, FrontErrorReportingContext>
  where
    S: Source,
    A: FnMut(&mut Item),
  {
    self.compile_source_with_parser_and_annotator(s, parser::str_to_relation_type, Some(annotator))
  }

  pub fn compile_rule<S>(&mut self, s: S) -> Result<SourceId, FrontErrorReportingContext>
  where
    S: Source,
  {
    self.compile_source_with_parser(s, parser::str_to_rule)
  }

  pub fn compile_rule_with_annotator<S, A>(
    &mut self,
    s: S,
    annotator: A,
  ) -> Result<SourceId, FrontErrorReportingContext>
  where
    S: Source,
    A: FnMut(&mut Item),
  {
    self.compile_source_with_parser_and_annotator(s, parser::str_to_rule, Some(annotator))
  }

  pub fn compile_query<S>(&mut self, s: S) -> Result<SourceId, FrontErrorReportingContext>
  where
    S: Source,
  {
    self.compile_source_with_parser(s, parser::str_to_query)
  }

  pub fn compile_query_with_annotator<S, A>(
    &mut self,
    s: S,
    annotator: A,
  ) -> Result<SourceId, FrontErrorReportingContext>
  where
    S: Source,
    A: FnMut(&mut Item),
  {
    self.compile_source_with_parser_and_annotator(s, parser::str_to_query, Some(annotator))
  }

  pub fn compile_source_with_parser<S, P>(
    &mut self,
    s: S,
    p: P,
  ) -> Result<SourceId, FrontErrorReportingContext>
  where
    S: Source,
    P: FnOnce(&str) -> Result<Vec<Item>, parser::ParserError>,
  {
    self.compile_source_with_parser_and_annotator(s, p, None::<fn(&mut Item)>)
  }

  pub fn compile_source_with_annotator<S, A>(
    &mut self,
    source: S,
    annotator: A,
  ) -> Result<SourceId, FrontErrorReportingContext>
  where
    S: Source,
    A: FnMut(&mut Item),
  {
    self.compile_source_with_parser_and_annotator(source, parser::str_to_items, Some(annotator))
  }

  pub fn compile_source_with_parser_and_annotator<S, P, A>(
    &mut self,
    source: S,
    parser: P,
    mut maybe_annotator: Option<A>,
  ) -> Result<SourceId, FrontErrorReportingContext>
  where
    S: Source,
    P: FnOnce(&str) -> Result<Vec<Item>, parser::ParserError>,
    A: FnMut(&mut Item),
  {
    // Make duplicate context since we want to stop when compile error
    let mut dup_ctx = self.clone();
    let mut error_ctx = FrontErrorReportingContext::new();

    // Parse the string
    let parsing_result = parser(source.content());
    let mut ast = match parsing_result {
      Ok(ast) => ast,
      Err(mut err) => {
        if let Some(name) = source.name() {
          err.set_source_name(name.to_string());
        }
        error_ctx.set_sources(&dup_ctx.sources);
        let source_id = error_ctx.sources.add(source);
        err.set_source_id(source_id);
        error_ctx.add(err);
        return Err(error_ctx);
      }
    };

    // Add & load imports
    dup_ctx.add_import(&source);
    dup_ctx.process_imports(&source, &ast)?;

    // Setup the location annotator
    let mut loc_annotator = LocationSpanAnnotator::new(&source);

    // Add the source to the context
    let source_id = dup_ctx.sources.add(source);
    error_ctx.set_sources(&dup_ctx.sources);

    // With source id (sid), setup source id annotator
    let mut source_id_annotator = SourceIdAnnotator::new(source_id);

    // Annotate it
    let node_id_annotator = &mut dup_ctx.node_id_annotator;
    let mut annotators = (
      node_id_annotator,
      &mut loc_annotator,
      &mut source_id_annotator,
    );
    annotators.walk_items(&mut ast);

    // Use external annotator to annotate each item
    if let Some(annotate) = &mut maybe_annotator {
      ast.iter_mut().for_each(|item| annotate(item));
    }

    // Front pre-transformaion analysis
    dup_ctx.analysis.modify(|analysis| {
      analysis.perform_pre_transformation_analysis(&ast);
      analysis.dump_errors(&mut error_ctx);
    });
    if error_ctx.has_error() {
      return Err(error_ctx);
    }

    // Front transformation; add new items into ast and re-annotate the ast
    apply_transformations(&mut ast);
    dup_ctx.node_id_annotator.walk_items(&mut ast);

    // Front analysis
    dup_ctx.analysis.modify(|analysis| {
      analysis.process_items(&ast);
      analysis.post_analysis();
      analysis.dump_errors(&mut error_ctx);
    });
    if error_ctx.has_error() {
      return Err(error_ctx);
    }

    // Update self if nothing goes wrong
    dup_ctx.items.extend(ast);
    *self = dup_ctx;

    // Pull out the last ast items
    Ok(SourceId(source_id))
  }

  fn process_imports<S: Source>(
    &mut self,
    s: &S,
    ast: &Vec<Item>,
  ) -> Result<(), FrontErrorReportingContext> {
    let mut error_ctx = FrontErrorReportingContext::new();
    for item in ast {
      if let Item::ImportDecl(id) = item {
        let f = s.resolve_import_file_path(id.input_file());
        if self.is_imported(&f) {
          error_ctx.add(CycleImportError { path: f });
          return Err(error_ctx);
        } else {
          match FileSource::new(&f) {
            Ok(s) => {
              self.compile_source(s)?;
            }
            Err(e) => {
              error_ctx.add(e);
              return Err(error_ctx);
            }
          }
        }
      }
    }
    return Ok(());
  }

  fn add_import<S: Source>(&mut self, s: &S) {
    if let Some(p) = s.absolute_file_path() {
      self.imported_files.insert(p);
    }
  }

  fn is_imported(&self, p: &PathBuf) -> bool {
    if let Ok(p) = fs::canonicalize(p) {
      self.imported_files.contains(&p)
    } else {
      false
    }
  }

  pub fn num_relations(&self) -> usize {
    self.type_inference().num_relations()
  }

  pub fn relations(&self) -> Vec<String> {
    self.type_inference().relations()
  }

  pub fn type_inference(&self) -> &TypeInference {
    &self.analysis.borrow().type_inference
  }

  pub fn items_of_source_id(&self, source_id: SourceId) -> impl Iterator<Item = &Item> {
    self
      .items
      .iter()
      .filter(move |item| item.location().source_id == source_id.0)
  }

  pub fn items_mut_of_source_id(&mut self, source_id: SourceId) -> impl Iterator<Item = &mut Item> {
    self
      .items
      .iter_mut()
      .filter(move |item| item.location().source_id == source_id.0)
  }

  pub fn rule_decl_of_source_id(&self, source_id: SourceId) -> Option<&RuleDecl> {
    self.items.iter().find_map(|i| match i {
      Item::RelationDecl(rd) => match &rd.node {
        RelationDeclNode::Rule(rd) => {
          if rd.source_id() == source_id.0 {
            Some(rd)
          } else {
            None
          }
        }
        _ => None,
      },
      _ => None,
    })
  }

  pub fn relation_type_decl_of_source_id(&self, source_id: SourceId) -> Option<&RelationTypeDecl> {
    self.items.iter().find_map(|i| match i {
      Item::TypeDecl(td) => match &td.node {
        TypeDeclNode::Relation(rtd) => {
          if rtd.source_id() == source_id.0 {
            Some(rtd)
          } else {
            None
          }
        }
        _ => None,
      },
      _ => None,
    })
  }

  pub fn iter_relation_decls(&self) -> impl Iterator<Item = &RelationDecl> {
    self.items.iter().filter_map(|item| match item {
      Item::RelationDecl(rd) => Some(rd),
      _ => None,
    })
  }

  pub fn iter_query_decls(&self) -> impl Iterator<Item = &QueryDecl> {
    self.items.iter().filter_map(|item| match item {
      Item::QueryDecl(qd) => Some(qd),
      _ => None,
    })
  }

  pub fn has_relation(&self, relation: &str) -> bool {
    self.type_inference().has_relation(relation)
  }

  pub fn relation_arg_types(&self, relation: &str) -> Option<Vec<ValueType>> {
    self.type_inference().relation_arg_types(relation)
  }

  pub fn relation_tuple_type(&self, relation: &str) -> Option<TupleType> {
    self.type_inference().relation_tuple_type(relation)
  }
}
