use std::collections::*;
use std::fs;
use std::path::PathBuf;

use super::analysis::*;
use super::analyzers::*;
use super::attribute::*;
use super::*;

use crate::common::foreign_aggregate::*;
use crate::common::foreign_function::*;
use crate::common::foreign_predicate::*;
use crate::common::tuple_type::*;
use crate::common::value_type::*;
use crate::utils::CopyOnWrite;

#[derive(Clone, Debug, Copy)]
pub struct SourceId(usize);

#[derive(Clone, Debug)]
pub struct FrontContext {
  /// All the compilation sources
  pub sources: Sources,

  /// The set of imported files (Paths)
  pub imported_files: HashSet<PathBuf>,

  /// All the compiled program items
  pub items: Items,

  /// Foreign function registry holding all foreign functions
  pub foreign_function_registry: ForeignFunctionRegistry,

  /// Foreign predicate registry holding all foreign predicates
  pub foreign_predicate_registry: ForeignPredicateRegistry,

  /// Foreign aggregate registry holding all foreign aggregates
  pub foreign_aggregate_registry: AggregateRegistry,

  /// Attribute processor registry holding all attribute processors
  pub attribute_processor_registry: AttributeProcessorRegistry,

  /// Node ID annotator for giving AST node IDs.
  pub node_id_annotator: NodeIdAnnotator,

  /// Front analysis which is Cow-ed, containing all the analyzed
  pub analysis: CopyOnWrite<Analysis>,
}

impl FrontContext {
  pub fn new() -> Self {
    let function_registry = ForeignFunctionRegistry::std();
    let predicate_registry = ForeignPredicateRegistry::std();
    let aggregate_registry = AggregateRegistry::std();
    let attribute_registry = AttributeProcessorRegistry::new();
    let analysis = Analysis::new(&function_registry, &predicate_registry, &aggregate_registry);
    Self {
      sources: Sources::new(),
      items: Vec::new(),
      foreign_function_registry: function_registry,
      foreign_predicate_registry: predicate_registry,
      foreign_aggregate_registry: aggregate_registry,
      attribute_processor_registry: attribute_registry,
      imported_files: HashSet::new(),
      node_id_annotator: NodeIdAnnotator::new(),
      analysis: CopyOnWrite::new(analysis),
    }
  }

  pub fn dump_ir(&self) {
    for item in &self.items {
      println!("{}", item);
    }
  }

  pub fn get_ir(&self) -> String {
    self
      .items
      .iter()
      .map(|i| format!("{}", i))
      .collect::<Vec<_>>()
      .join("\n")
  }

  pub fn register_foreign_function<F>(&mut self, f: F) -> Result<(), ForeignFunctionError>
  where
    F: ForeignFunction + Send + Sync + 'static,
  {
    // Prepare type inference data
    let func_name = f.name();
    let func_type = type_inference::FunctionType::from(&f);

    // First add the function to the foreign function registry
    // This process will make sure that the function is well formed and can be added
    self.foreign_function_registry.register(f)?;

    // If succeeded, we add it to the type inference module
    self.analysis.modify(|analysis| {
      analysis
        .type_inference
        .foreign_function_type_registry
        .add_function_type(func_name, func_type)
    })?;

    Ok(())
  }

  pub fn register_foreign_predicate<F>(&mut self, f: F) -> Result<(), ForeignPredicateError>
  where
    F: ForeignPredicate + Send + Sync + Clone + 'static,
  {
    // Check if the predicate name has already be defined before
    if self.type_inference().has_relation(&f.internal_name()) {
      return Err(ForeignPredicateError::AlreadyExisted { id: f.internal_name() });
    }

    // Add the predicate to the registry
    self.foreign_predicate_registry.register(f.clone())?;

    // If succeeded, we add it to the type inference module
    self.analysis.modify(|analysis| {
      // Update the type inference module
      analysis
        .type_inference
        .foreign_predicate_type_registry
        .add_foreign_predicate(&f);

      // Update the head analysis module
      analysis.head_relation_analysis.add_foreign_predicate(&f);

      // Update the boundness analysis module
      analysis.boundness_analysis.add_foreign_predicate(&f);
    });

    Ok(())
  }

  pub fn register_attribute_processor<P>(&mut self, p: P) -> Result<(), AttributeError>
  where
    P: AttributeProcessor + Send + Sync + Clone,
  {
    self.attribute_processor_registry.register(p)?;
    Ok(())
  }

  pub fn compile_string(&mut self, s: String) -> Result<SourceId, FrontCompileError> {
    let source = StringSource::new(s);
    self.compile_source_with_parser(source, parser::str_to_items)
  }

  pub fn compile_source<S: Source>(&mut self, s: S) -> Result<SourceId, FrontCompileError> {
    self.compile_source_with_parser(s, parser::str_to_items)
  }

  pub fn compile_relation<S>(&mut self, s: S) -> Result<SourceId, FrontCompileError>
  where
    S: Source,
  {
    self.compile_source_with_parser(s, parser::str_to_relation_type)
  }

  pub fn compile_relation_with_annotator<S, A>(&mut self, s: S, annotator: A) -> Result<SourceId, FrontCompileError>
  where
    S: Source,
    A: FnMut(&mut Item),
  {
    self.compile_source_with_parser_and_annotator(s, parser::str_to_relation_type, Some(annotator))
  }

  pub fn compile_rule<S>(&mut self, s: S) -> Result<SourceId, FrontCompileError>
  where
    S: Source,
  {
    self.compile_source_with_parser(s, parser::str_to_rule)
  }

  pub fn compile_rule_with_annotator<S, A>(&mut self, s: S, annotator: A) -> Result<SourceId, FrontCompileError>
  where
    S: Source,
    A: FnMut(&mut Item),
  {
    self.compile_source_with_parser_and_annotator(s, parser::str_to_rule, Some(annotator))
  }

  pub fn compile_query<S>(&mut self, s: S) -> Result<SourceId, FrontCompileError>
  where
    S: Source,
  {
    self.compile_source_with_parser(s, parser::str_to_query)
  }

  pub fn compile_query_with_annotator<S, A>(&mut self, s: S, annotator: A) -> Result<SourceId, FrontCompileError>
  where
    S: Source,
    A: FnMut(&mut Item),
  {
    self.compile_source_with_parser_and_annotator(s, parser::str_to_query, Some(annotator))
  }

  pub fn compile_source_with_parser<S, P>(&mut self, s: S, p: P) -> Result<SourceId, FrontCompileError>
  where
    S: Source,
    P: FnOnce(&str) -> Result<Vec<Item>, parser::ParserError>,
  {
    self.compile_source_with_parser_and_annotator(s, p, None::<fn(&mut Item)>)
  }

  pub fn compile_source_with_annotator<S, A>(&mut self, source: S, annotator: A) -> Result<SourceId, FrontCompileError>
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
  ) -> Result<SourceId, FrontCompileError>
  where
    S: Source,
    P: FnOnce(&str) -> Result<Vec<Item>, parser::ParserError>,
    A: FnMut(&mut Item),
  {
    // Make duplicate context since we want to stop when compile error
    let mut dup_ctx = self.clone();
    let mut error_ctx = FrontCompileError::new();

    // Parse the string
    let parsing_result = parser(source.content());
    let mut ast = match parsing_result {
      Ok(ast) => ast,
      Err(mut err) => {
        if let Some(name) = source.name() {
          err.set_source_name(name.to_string());
        }
        error_ctx.set_sources(&dup_ctx.sources);
        let source_id = error_ctx.add_source(source);
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
    ast.walk_mut(&mut (
      &mut dup_ctx.node_id_annotator,
      &mut loc_annotator,
      &mut source_id_annotator,
    ));

    // Use external annotator to annotate each item
    if let Some(annotate) = &mut maybe_annotator {
      ast.iter_mut().for_each(|item| annotate(item));
    }

    // Use foreign attribute registry to annotate item
    match self
      .attribute_processor_registry
      .analyze_and_process(&mut dup_ctx, &mut ast)
    {
      Ok(_) => {}
      Err(err) => {
        error_ctx.add(err);
        return Err(error_ctx);
      }
    };

    // Front pre-transformaion analysis
    dup_ctx.analysis.modify(|analysis| {
      analysis.perform_pre_transformation_analysis(&ast);
      analysis.dump_errors(&mut error_ctx);
    });
    if error_ctx.has_error() {
      return Err(error_ctx);
    }

    // Front transformation; add new items into ast and re-annotate the ast
    dup_ctx.analysis.modify_without_copy(|analysis| {
      apply_transformations(&mut ast, analysis);
    });
    ast.walk_mut(&mut dup_ctx.node_id_annotator);

    // Front analysis
    dup_ctx.analysis.modify(|analysis| {
      analysis.process_items(&ast);
      analysis.post_analysis();
      analysis.dump_errors(&mut error_ctx);
    });
    if error_ctx.has_error() {
      return Err(error_ctx);
    }

    // If there is no error, print the warnings
    if error_ctx.has_warning() {
      error_ctx.report_warnings();
    }

    // Update self if nothing goes wrong
    dup_ctx.items.extend(ast);
    *self = dup_ctx;

    // Pull out the last ast items
    Ok(SourceId(source_id))
  }

  fn process_imports<S: Source>(&mut self, s: &S, ast: &Vec<Item>) -> Result<(), FrontCompileError> {
    let mut error_ctx = FrontCompileError::new();
    for item in ast {
      if let Item::ImportDecl(id) = item {
        let f = s.resolve_import_file_path(id.import_file_path());
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
    self
      .type_inference()
      .relations()
      .into_iter()
      .filter(|r| !self.is_hidden_relation(r))
      .collect()
  }

  pub fn is_hidden_relation(&self, r: &str) -> bool {
    self.hidden_relation_analysis().contains(r)
  }

  pub fn type_inference(&self) -> &TypeInference {
    &self.analysis.borrow().type_inference
  }

  pub fn hidden_relation_analysis(&self) -> &HiddenRelationAnalysis {
    &self.analysis.borrow().hidden_analysis
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
      Item::RelationDecl(RelationDecl::Rule(rd)) if rd.location_source_id() == source_id.0 => Some(rd),
      _ => None,
    })
  }

  pub fn relation_type_decl_of_source_id(&self, source_id: SourceId) -> Option<&RelationTypeDecl> {
    self.items.iter().find_map(|i| match i {
      Item::TypeDecl(TypeDecl::Relation(rtd)) if rtd.location_source_id() == source_id.0 => Some(rtd),
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
