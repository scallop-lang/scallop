use std::collections::*;

use crate::common::adt_variant_registry::*;
use crate::common::expr::*;
use crate::common::foreign_aggregate::*;
use crate::common::foreign_function::*;
use crate::common::foreign_predicate::*;
use crate::common::input_file::InputFile;
use crate::common::input_tag::DynamicInputTag;
use crate::common::output_option::OutputOption;
use crate::common::tuple::{AsTuple, Tuple};
use crate::common::tuple_type::TupleType;
use crate::common::value::Value;

use crate::runtime::database::StorageMetadata;
use crate::runtime::env::Scheduler;

#[derive(Debug, Clone)]
pub struct Program {
  pub strata: Vec<Stratum>,
  pub function_registry: ForeignFunctionRegistry,
  pub predicate_registry: ForeignPredicateRegistry,
  pub aggregate_registry: AggregateRegistry,
  pub adt_variant_registry: ADTVariantRegistry,
  pub relation_to_stratum: HashMap<String, usize>,
}

impl Program {
  pub fn new() -> Self {
    Self {
      strata: Vec::new(),
      function_registry: ForeignFunctionRegistry::new(),
      predicate_registry: ForeignPredicateRegistry::new(),
      aggregate_registry: AggregateRegistry::new(),
      adt_variant_registry: ADTVariantRegistry::new(),
      relation_to_stratum: HashMap::new(),
    }
  }

  /// Get a relation by its name; returns `None` if the relation does not exist.
  pub fn relation(&self, name: &str) -> Option<&Relation> {
    self
      .relation_to_stratum
      .get(name)
      .and_then(|stratum_id| self.strata[*stratum_id].relations.get(name))
  }

  /// Get a relation by its name; panics if the relation does not exist.
  pub fn relation_unchecked(&self, name: &str) -> &Relation {
    &self.strata[self.relation_to_stratum[name]].relations[name]
  }

  /// Iterate over all relations in the program.
  pub fn relations(&self) -> impl Iterator<Item = &Relation> {
    self.strata.iter().flat_map(|s| s.relations.values())
  }

  /// Iterate over all relation (name, type) pairs in the program
  pub fn relation_types<'a>(&'a self) -> impl 'a + Iterator<Item = (String, TupleType)> {
    self
      .strata
      .iter()
      .flat_map(|s| s.relations.iter().map(|(p, r)| (p.clone(), r.tuple_type.clone())))
  }

  /// Get the tuple type of a relation by its name; returns `None` if the relation does not exist.
  pub fn relation_tuple_type(&self, predicate: &str) -> Option<TupleType> {
    if let Some(stratum_id) = self.relation_to_stratum.get(predicate) {
      Some(self.strata[*stratum_id].relations[predicate].tuple_type.clone())
    } else {
      None
    }
  }

  /// Get the input file option of a relation by its name, returns `None` if the relation does not exist.
  pub fn input_file(&self, predicate: &str) -> Option<&InputFile> {
    if let Some(stratum_id) = self.relation_to_stratum.get(predicate) {
      self.strata[*stratum_id].relations[predicate].input_file.as_ref()
    } else {
      None
    }
  }

  /// Iterate through all the input files in the program.
  pub fn input_files<'a>(&'a self) -> impl 'a + Iterator<Item = &'a InputFile> {
    self
      .relations()
      .filter_map(move |relation| relation.input_file.as_ref())
  }

  /// Set the target relation(s) to be output relations.
  pub fn set_output_relations(&mut self, target: Vec<&str>) {
    self.strata.iter_mut().for_each(|stratum| {
      stratum.relations.iter_mut().for_each(|(_, relation)| {
        if target.contains(&relation.predicate.as_str()) {
          relation.output = OutputOption::Default
        } else {
          relation.output = OutputOption::Hidden
        }
      })
    })
  }

  /// Get the output option of a relation by its name; returns `None` if the relation does not exist.
  pub fn output_option(&self, relation: &str) -> Option<&OutputOption> {
    self
      .relation_to_stratum
      .get(relation)
      .map(|stratum_id| &self.strata[*stratum_id].relations[relation].output)
  }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct Stratum {
  pub is_recursive: bool,
  pub relations: BTreeMap<String, Relation>,
  pub updates: Vec<Update>,
}

impl Stratum {
  pub fn relation(&self, r: &str) -> Option<&Relation> {
    self.relations.get(r)
  }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct Relation {
  /// The name of the relation
  pub predicate: String,

  /// The tuple type of the relation; will be always a one level tuple with no nested tuples
  pub tuple_type: TupleType,

  /// Whether there is a input file where this relation should be loaded from
  pub input_file: Option<InputFile>,

  /// Dynamic facts associated with this relation
  pub facts: Vec<Fact>,

  /// The output option; whether it is hidden or returned or piped to a file
  pub output: OutputOption,

  /// Whether the relation is a goal
  pub is_goal: bool,

  /// The scheduler for this specific scheduler
  pub scheduler: Option<Scheduler>,

  /// Storage of the relation
  pub storage: StorageMetadata,

  /// Whether the relation is immutable, i.e., not being populated by any rule
  pub immutable: bool,
}

impl Relation {
  /// Create a hidden temporary relation with predicate name and type
  pub fn hidden_relation(predicate: String, tuple_type: TupleType) -> Self {
    Self {
      predicate,
      tuple_type,
      input_file: None,
      facts: vec![],
      output: OutputOption::Hidden,
      is_goal: false,
      scheduler: None,
      storage: StorageMetadata::default(),
      immutable: false,
    }
  }

  pub fn populates_edb(&self) -> bool {
    !self.facts.is_empty() || self.input_file.is_some()
  }
}

impl std::cmp::Ord for Relation {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;

    // First compare predicate
    let pcmp = self.predicate.cmp(&other.predicate);
    if pcmp != Equal {
      return pcmp;
    };

    // Then compare tuple type
    let tcmp = self.tuple_type.cmp(&other.tuple_type);
    if tcmp != Equal {
      return tcmp;
    };

    // Then compare input file
    let icmp = self.input_file.cmp(&other.input_file);
    if icmp != Equal {
      return icmp;
    };

    // Finally compare facts
    self.facts.cmp(&other.facts)
  }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Fact {
  pub tag: DynamicInputTag,
  pub tuple: Tuple,
}

impl std::cmp::Eq for Fact {}

impl std::cmp::Ord for Fact {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    match self.partial_cmp(other) {
      Some(ord) => ord,
      _ => panic!("[Internal Error] No ordering found between facts"),
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Update {
  pub target: String,
  pub dataflow: Dataflow,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dataflow {
  // Base relation
  Unit(TupleType),
  UntaggedVec(Vec<Tuple>),
  Relation(String),

  // Unary operations
  Project(Box<Dataflow>, Expr),
  Filter(Box<Dataflow>, Expr),
  Find(Box<Dataflow>, Tuple),
  Sorted(Box<Dataflow>),

  // Binary operations
  Union(Box<Dataflow>, Box<Dataflow>),
  Join(Box<Dataflow>, Box<Dataflow>),
  Intersect(Box<Dataflow>, Box<Dataflow>),
  Product(Box<Dataflow>, Box<Dataflow>),
  Antijoin(Box<Dataflow>, Box<Dataflow>),
  Difference(Box<Dataflow>, Box<Dataflow>),

  // Specialized binary operations
  JoinIndexedVec(Box<Dataflow>, String),

  // Aggregation
  Reduce(Reduce),

  // Tag operations
  OverwriteOne(Box<Dataflow>),
  Exclusion(Box<Dataflow>, Box<Dataflow>),

  // Foreign predicates
  ForeignPredicateGround(String, Vec<Value>),
  ForeignPredicateConstraint(Box<Dataflow>, String, Vec<Expr>),
  ForeignPredicateJoin(Box<Dataflow>, String, Vec<Expr>),
}

impl Dataflow {
  /// Create a new unit dataflow given a tuple type
  pub fn unit(tuple_type: TupleType) -> Self {
    Self::Unit(tuple_type)
  }

  /// Create a union dataflow
  pub fn union(self, d2: Dataflow) -> Self {
    Self::Union(Box::new(self), Box::new(d2))
  }

  /// Create a join-ed dataflow from two dataflows
  pub fn join(self, d2: Dataflow) -> Self {
    Self::Join(Box::new(self), Box::new(d2))
  }

  /// Create a right indexed-vec join
  pub fn join_indexed_vec(self, d2: String) -> Self {
    Self::JoinIndexedVec(Box::new(self), d2)
  }

  /// Create an intersection dataflow from two dataflows
  pub fn intersect(self, d2: Dataflow) -> Self {
    Self::Intersect(Box::new(self), Box::new(d2))
  }

  pub fn product(self, d2: Dataflow) -> Self {
    Self::Product(Box::new(self), Box::new(d2))
  }

  pub fn antijoin(self, d2: Dataflow) -> Self {
    Self::Antijoin(Box::new(self), Box::new(d2))
  }

  pub fn difference(self, d2: Dataflow) -> Self {
    Self::Difference(Box::new(self), Box::new(d2))
  }

  pub fn project<E: Into<Expr>>(self, expr: E) -> Self {
    Self::Project(Box::new(self), expr.into())
  }

  pub fn filter<E: Into<Expr>>(self, expr: E) -> Self {
    Self::Filter(Box::new(self), expr.into())
  }

  pub fn find<T: AsTuple<Tuple>>(self, t: T) -> Self {
    Self::Find(Box::new(self), AsTuple::as_tuple(&t))
  }

  pub fn sorted(self) -> Self {
    Self::Sorted(Box::new(self))
  }

  pub fn overwrite_one(self) -> Self {
    Self::OverwriteOne(Box::new(self))
  }

  pub fn exclusion(self, right: Vec<Tuple>) -> Self {
    Self::Exclusion(Box::new(self), Box::new(Self::UntaggedVec(right)))
  }

  pub fn foreign_predicate_constraint(self, predicate: String, args: Vec<Expr>) -> Self {
    Self::ForeignPredicateConstraint(Box::new(self), predicate, args)
  }

  pub fn foreign_predicate_join(self, predicate: String, args: Vec<Expr>) -> Self {
    Self::ForeignPredicateJoin(Box::new(self), predicate, args)
  }

  pub fn reduce<S: ToString>(
    aggregator: String,
    aggregate_info: AggregateInfo,
    predicate: S,
    group_by: ReduceGroupByType,
  ) -> Self {
    Self::Reduce(Reduce {
      aggregator,
      aggregate_info,
      predicate: predicate.to_string(),
      group_by,
    })
  }

  pub fn relation<S: ToString>(r: S) -> Self {
    Self::Relation(r.to_string())
  }

  pub fn source_relations(&self) -> HashSet<&String> {
    match self {
      Self::Unit(_) => HashSet::new(),
      Self::Union(d1, d2)
      | Self::Join(d1, d2)
      | Self::Intersect(d1, d2)
      | Self::Product(d1, d2)
      | Self::Antijoin(d1, d2)
      | Self::Difference(d1, d2) => d1.source_relations().union(&d2.source_relations()).cloned().collect(),
      Self::Project(d, _)
      | Self::Filter(d, _)
      | Self::Find(d, _)
      | Self::Sorted(d)
      | Self::OverwriteOne(d)
      | Self::ForeignPredicateConstraint(d, _, _)
      | Self::ForeignPredicateJoin(d, _, _)
      | Self::Exclusion(d, _) => d.source_relations(),
      Self::JoinIndexedVec(d, s) => std::iter::once(s).chain(d.source_relations().into_iter()).collect(),
      Self::Reduce(r) => std::iter::once(r.source_relation()).collect(),
      Self::Relation(r) => std::iter::once(r).collect(),
      Self::ForeignPredicateGround(_, _) | Self::UntaggedVec(_) => HashSet::new(),
    }
  }

  pub fn substitute_relation(&mut self, source: &str, target: &str) -> bool {
    let mut has_update = false;
    match self {
      Self::Unit(_) => false,
      Self::Union(d1, d2)
      | Self::Join(d1, d2)
      | Self::Intersect(d1, d2)
      | Self::Product(d1, d2)
      | Self::Antijoin(d1, d2)
      | Self::Difference(d1, d2) => {
        has_update |= d1.substitute_relation(source, target);
        has_update |= d2.substitute_relation(source, target);
        has_update
      }
      Self::Project(d, _)
      | Self::Filter(d, _)
      | Self::Find(d, _)
      | Self::Sorted(d)
      | Self::OverwriteOne(d)
      | Self::ForeignPredicateConstraint(d, _, _)
      | Self::ForeignPredicateJoin(d, _, _)
      | Self::Exclusion(d, _) => {
        has_update |= d.substitute_relation(source, target);
        has_update
      }
      Self::JoinIndexedVec(d, s) => {
        has_update |= d.substitute_relation(source, target);
        if s == source {
          *s = target.to_string();
          has_update |= true;
        }
        has_update
      }
      Self::Reduce(r) => {
        has_update |= r.substitute_relation(source, target);
        has_update
      }
      Self::Relation(r) => {
        if r == source {
          *r = target.to_string();
          true
        } else {
          false
        }
      }
      Self::ForeignPredicateGround(_, _) | Self::UntaggedVec(_) => {
        false
      },
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReduceGroupByType {
  None,
  Implicit,
  Join(String),
}

impl ReduceGroupByType {
  pub fn none() -> Self {
    Self::None
  }

  pub fn implicit() -> Self {
    Self::Implicit
  }

  pub fn join<S: ToString>(s: S) -> Self {
    Self::Join(s.to_string())
  }

  pub fn substitute_relation(&mut self, source: &str, target: &str) -> bool {
    match self {
      Self::None | Self::Implicit => {
        false
      }
      Self::Join(s) => {
        if s == source {
          *s = target.to_string();
          true
        } else {
          false
        }
      }
    }
  }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reduce {
  pub aggregator: String,
  pub aggregate_info: AggregateInfo,
  pub predicate: String,
  pub group_by: ReduceGroupByType,
}

impl Reduce {
  pub fn source_relation(&self) -> &String {
    &self.predicate
  }

  pub fn substitute_relation(&mut self, source: &str, target: &str) -> bool {
    let mut has_update = false;
    if &self.predicate == source {
      self.predicate = target.to_string();
      has_update = true;
    }

    has_update |= self.group_by.substitute_relation(source, target);

    has_update
  }
}
