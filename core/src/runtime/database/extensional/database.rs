use std::collections::*;

use crate::common::input_tag::*;
use crate::common::tuple::*;
use crate::common::tuple_type::*;
use crate::compiler::ram;
use crate::runtime::dynamic::*;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;
use crate::runtime::statics::*;

use super::*;

#[derive(Clone, Debug)]
pub struct ExtensionalDatabase<Prov: Provenance> {
  /// Configuration of whether we perform type check when adding facts
  pub type_check: bool,

  /// Disjunction count; the count used for automatically generating mutual exclusion ID.
  pub disjunction_count: usize,

  /// Types of relations
  pub relation_types: HashMap<String, TupleType>,

  /// Extensional relations
  pub extensional_relations: HashMap<String, ExtensionalRelation<Prov>>,

  /// Flag for whether is internalized
  pub internalized: bool,
}

impl<Prov: Provenance> ExtensionalDatabase<Prov> {
  /// Create a new extensional database
  pub fn new() -> Self {
    Self {
      type_check: true,
      disjunction_count: 0,
      relation_types: HashMap::new(),
      extensional_relations: HashMap::new(),
      internalized: false,
    }
  }

  pub fn new_with_options(type_check: bool) -> Self {
    Self {
      type_check,
      disjunction_count: 0,
      relation_types: HashMap::new(),
      extensional_relations: HashMap::new(),
      internalized: false,
    }
  }

  pub fn clone_with_new_provenance<Prov2: Provenance>(&self) -> ExtensionalDatabase<Prov2>
  where
    Prov2::InputTag: ConvertFromInputTag<Prov::InputTag>,
  {
    ExtensionalDatabase {
      type_check: self.type_check,
      disjunction_count: self.disjunction_count,
      relation_types: self.relation_types.clone(),
      extensional_relations: self.extensional_relations.iter().map(|(pred, rel)| {
        let new_rel = rel.clone_with_new_provenance();
        (pred.clone(), new_rel)
      }).collect(),
      internalized: false,
    }
  }

  pub fn with_relation_types<I>(types: I) -> Self
  where
    I: Iterator<Item = (String, TupleType)>,
  {
    Self {
      type_check: true,
      disjunction_count: 0,
      relation_types: types.collect(),
      extensional_relations: HashMap::new(),
      internalized: false,
    }
  }

  pub fn with_relation_types_and_options<I>(types: I, type_check: bool) -> Self
  where
    I: Iterator<Item = (String, TupleType)>,
  {
    Self {
      type_check,
      disjunction_count: 0,
      relation_types: types.collect(),
      extensional_relations: HashMap::new(),
      internalized: false,
    }
  }

  pub fn type_of(&self, relation: &str) -> Option<TupleType> {
    self.relation_types.get(relation).cloned()
  }

  pub fn tap_relation(&mut self, relation: &str) {
    self.extensional_relations.entry(relation.to_string()).or_default();
  }

  pub fn has_relation(&self, relation: &str) -> bool {
    self.extensional_relations.contains_key(relation)
  }

  pub fn add_dynamic_input_facts<T>(&mut self, relation: &str, facts: Vec<(DynamicInputTag, T)>) -> Result<(), DatabaseError>
  where
    T: Into<Tuple>,
  {
    let facts: Vec<(_, Tuple)> = facts.into_iter().map(|(tag, tup)| (tag, tup.into())).collect();
    self.check_tuples_type(relation, facts.iter().map(|(_, tup)| tup))?;
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_dynamic_input_facts(facts);
    Ok(())
  }

  pub fn add_facts<T>(&mut self, relation: &str, facts: Vec<T>) -> Result<(), DatabaseError>
  where
    T: Into<Tuple>,
  {
    let facts: Vec<Tuple> = facts.into_iter().map(|tup| tup.into()).collect();
    self.check_tuples_type(relation, facts.iter())?;
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_facts(facts);
    Ok(())
  }

  pub fn add_static_input_facts(
    &mut self,
    relation: &str,
    facts: Vec<(Option<Prov::InputTag>, Tuple)>,
  ) -> Result<(), DatabaseError> {
    let facts: Vec<(_, Tuple)> = facts.into_iter().map(|(tag, tup)| (tag, tup.into())).collect();
    self.check_tuples_type(relation, facts.iter().map(|(_, tup)| tup))?;
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_static_input_facts(facts);
    Ok(())
  }

  pub fn add_static_input_facts_without_type_check(
    &mut self,
    relation: &str,
    facts: Vec<(Option<Prov::InputTag>, Tuple)>,
  ) -> Result<(), DatabaseError> {
    let facts: Vec<(_, Tuple)> = facts.into_iter().map(|(tag, tup)| (tag, tup.into())).collect();
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_static_input_facts(facts);
    Ok(())
  }

  pub fn get_dynamic_collection(&self, relation: &str) -> Option<&DynamicCollection<Prov>> {
    self.extensional_relations.get(relation).map(|r| &r.internal)
  }

  pub fn pop_dynamic_collection(&mut self, relation: &str) -> Option<DynamicCollection<Prov>> {
    self.extensional_relations.remove(relation).map(|r| r.internal)
  }

  pub fn load_into_static_relation<Tup>(&self, relation: &str, ctx: &Prov, rela: &StaticRelation<Tup, Prov>)
  where
    Tup: StaticTupleTrait,
    Tuple: AsTuple<Tup>,
  {
    if let Some(extensional_relation) = self.extensional_relations.get(relation) {
      if !extensional_relation.internal.is_empty() {
        rela.insert_dynamic_elements_ref(ctx, &extensional_relation.internal.elements);
      }
    }
  }

  pub fn populate_program_facts(&mut self, program: &ram::Program) -> Result<(), DatabaseError> {
    // Iterate through all relations declared in the program
    for relation in program.relations() {
      // Check if we need to load the relation facts
      if !relation.facts.is_empty() {
        let edb_relation = self
          .extensional_relations
          .entry(relation.predicate.clone())
          .or_default();
        if edb_relation.internalized_program_facts
          && edb_relation.has_program_facts()
          && edb_relation.num_program_facts() < relation.facts.len()
        {
          return Err(DatabaseError::NewProgramFacts {
            relation: relation.predicate.clone(),
          });
        }
        edb_relation.add_program_facts(relation.facts.iter().map(|f| (f.tag.clone(), f.tuple.clone())));
      }

      // Check if we need to load external facts (from files or databases)
      if relation.input_file.is_some() {
        unimplemented!("Cannot load external file yet");
      }
    }
    Ok(())
  }

  pub fn need_update_relations(&self) -> HashSet<String> {
    if self.internalized {
      self
        .extensional_relations
        .iter()
        .filter(|(_, r)| !r.internalized)
        .map(|(r, _)| r)
        .cloned()
        .collect()
    } else {
      HashSet::new()
    }
  }

  pub fn internalize(&mut self, ctx: &mut Prov) {
    for (_, relation) in &mut self.extensional_relations {
      relation.internalize(ctx);
    }
    self.internalized = true
  }

  pub fn internalize_with_monitor<M: Monitor<Prov>>(&mut self, ctx: &mut Prov, m: &M) {
    for (_, relation) in &mut self.extensional_relations {
      relation.internalize_with_monitor(ctx, m);
    }
    self.internalized = true
  }

  fn check_tuples_type<'a, I>(&self, relation: &str, iter: I) -> Result<(), DatabaseError>
  where
    I: Iterator<Item = &'a Tuple>,
  {
    // Check types
    if self.type_check {
      // Get the tuple type
      if let Some(tuple_type) = self.relation_types.get(relation) {
        // Iterate through all tuples
        for tuple in iter {
          if !tuple_type.matches(tuple) {
            return Err(DatabaseError::TypeError {
              relation: relation.to_string(),
              relation_type: tuple_type.clone(),
              tuple: tuple.clone(),
            });
          }
        }
        Ok(())
      } else {
        Err(DatabaseError::UnknownRelation {
          relation: relation.to_string(),
        })
      }
    } else {
      Ok(())
    }
  }
}

impl<Prov> ExtensionalDatabase<Prov>
where
  Prov: Provenance<InputTag = InputExclusiveProb>,
{
  pub fn add_exclusive_probabilistic_facts<T>(
    &mut self,
    relation: &str,
    facts: Vec<(f64, T)>,
  ) -> Result<(), DatabaseError>
  where
    T: Into<Tuple>,
  {
    let exclusion_id = self.disjunction_count;
    let facts: Vec<(_, Tuple)> = facts
      .into_iter()
      .map(|(prob, tup)| (Some(InputExclusiveProb::new(prob, Some(exclusion_id))), tup.into()))
      .collect();
    self.check_tuples_type(relation, facts.iter().map(|(_, tup)| tup))?;
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_static_input_facts(facts);
    self.disjunction_count += 1;
    Ok(())
  }

  pub fn add_probabilistic_facts<T>(&mut self, relation: &str, facts: Vec<(f64, T)>) -> Result<(), DatabaseError>
  where
    T: Into<Tuple>,
  {
    let facts: Vec<(_, Tuple)> = facts
      .into_iter()
      .map(|(prob, tup)| (Some(InputExclusiveProb::new(prob, None)), tup.into()))
      .collect();
    self.check_tuples_type(relation, facts.iter().map(|(_, tup)| tup))?;
    self
      .extensional_relations
      .entry(relation.to_string())
      .or_default()
      .add_static_input_facts(facts);
    Ok(())
  }
}
