use std::collections::*;

use crate::common::tuple::{AsTuple, Tuple};
use crate::common::tuple_type::TupleType;
use crate::runtime::provenance::ProvenanceContext;
use crate::runtime::statics::{StaticRelation, StaticTupleTrait};

use super::*;

pub struct EDB<C: ProvenanceContext> {
  types: BTreeMap<String, TupleType>,
  relations: BTreeMap<String, EDBRelation<C>>,
}

impl<C: ProvenanceContext> EDB<C> {
  pub fn new() -> Self {
    Self {
      types: BTreeMap::new(),
      relations: BTreeMap::new(),
    }
  }

  pub fn new_with_types<I>(types: I) -> Self
  where
    I: IntoIterator<Item = (String, TupleType)>,
  {
    Self {
      types: types.into_iter().collect(),
      relations: BTreeMap::new(),
    }
  }

  pub fn type_of(&self, relation: &str) -> Option<TupleType> {
    self.types.get(relation).cloned()
  }

  pub fn load_into_static_relation<Tup: StaticTupleTrait>(
    &mut self,
    relation: &str,
    ctx: &mut C,
    target: &StaticRelation<Tup, C::Tag>,
  ) where
    Tuple: AsTuple<Tup>,
  {
    if let Some(r) = self.relations.remove(relation) {
      target.insert_from_edb(ctx, r)
    }
  }

  fn process_facts<F, I>(&self, ty: &TupleType, facts: I) -> Result<Vec<EDBFact<C>>, Tuple>
  where
    F: Into<Tuple>,
    I: Iterator<Item = (Option<C::InputTag>, F)>,
  {
    facts
      .map(|(input_tag, f)| {
        let t: Tuple = f.into();
        if ty.matches(&t) {
          Ok(EDBFact::<C>::new(input_tag, t))
        } else {
          Err(t)
        }
      })
      .collect::<Result<Vec<_>, _>>()
  }

  pub fn add_facts<F>(&mut self, relation: &str, facts: Vec<(Option<C::InputTag>, F)>) -> Result<(), EDBError>
  where
    F: Clone + Into<Tuple>,
  {
    // If there is no fact, we do nothing
    if facts.len() == 0 {
      return Ok(());
    }

    // We get the type stored in the EDB. If there is no such type, we use the type of the first fact
    let mut newly_inserted = false;
    let ty = self.types.get(relation).cloned().unwrap_or_else(|| {
      newly_inserted = true;
      let t: Tuple = facts[0].1.clone().into();
      TupleType::type_of(&t)
    });

    // Turn facts into a set of EDBFact
    let edb_facts = self.process_facts(&ty, facts.into_iter()).map_err(|t| {
      if newly_inserted {
        EDBError::TypeMismatch {
          expected: ty.clone(),
          found: TupleType::type_of(&t),
          actual: t,
        }
      } else {
        EDBError::RelationTypeError {
          relation: relation.to_string(),
          expected: ty.clone(),
          found: TupleType::type_of(&t),
          actual: t,
        }
      }
    })?;

    // Get the relation, insert the facts
    self
      .relations
      .entry(relation.to_string())
      .or_default()
      .extend_facts(edb_facts);

    // If we are just creating the type, insert it too
    if newly_inserted {
      self.types.insert(relation.to_string(), ty);
    }

    // Success
    Ok(())
  }

  pub fn add_tagged_facts<F>(&mut self, relation: &str, facts: Vec<(C::InputTag, F)>) -> Result<(), EDBError>
  where
    F: Clone + Into<Tuple>,
  {
    // If there is no fact, we do nothing
    if facts.len() == 0 {
      return Ok(());
    }

    // We get the type stored in the EDB. If there is no such type, we use the type of the first fact
    let mut newly_inserted = false;
    let ty = self.types.get(relation).cloned().unwrap_or_else(|| {
      newly_inserted = true;
      let t: Tuple = facts[0].1.clone().into();
      TupleType::type_of(&t)
    });

    // Turn facts into a set of EDBFact
    let edb_facts = self
      .process_facts(&ty, facts.into_iter().map(|(tag, tup)| (Some(tag), tup)))
      .map_err(|t| {
        if newly_inserted {
          EDBError::TypeMismatch {
            expected: ty.clone(),
            found: TupleType::type_of(&t),
            actual: t,
          }
        } else {
          EDBError::RelationTypeError {
            relation: relation.to_string(),
            expected: ty.clone(),
            found: TupleType::type_of(&t),
            actual: t,
          }
        }
      })?;

    // Get the relation, insert the facts
    self
      .relations
      .entry(relation.to_string())
      .or_default()
      .extend_facts(edb_facts);

    // If we are just creating the type, insert it too
    if newly_inserted {
      self.types.insert(relation.to_string(), ty);
    }

    // Success
    Ok(())
  }

  pub fn add_untagged_facts<F>(&mut self, relation: &str, facts: Vec<F>) -> Result<(), EDBError>
  where
    F: Clone + Into<Tuple>,
  {
    // If there is no fact, we do nothing
    if facts.len() == 0 {
      return Ok(());
    }

    // We get the type stored in the EDB. If there is no such type, we use the type of the first fact
    let mut newly_inserted = false;
    let ty = self.types.get(relation).cloned().unwrap_or_else(|| {
      newly_inserted = true;
      let t: Tuple = facts[0].clone().into();
      TupleType::type_of(&t)
    });

    // Turn facts into a set of EDBFact
    let edb_facts = self
      .process_facts(&ty, facts.into_iter().map(|f| (None, f)))
      .map_err(|t| {
        if newly_inserted {
          EDBError::TypeMismatch {
            expected: ty.clone(),
            found: TupleType::type_of(&t),
            actual: t,
          }
        } else {
          EDBError::RelationTypeError {
            relation: relation.to_string(),
            expected: ty.clone(),
            found: TupleType::type_of(&t),
            actual: t,
          }
        }
      })?;

    // Get the relation, insert the facts
    self
      .relations
      .entry(relation.to_string())
      .or_default()
      .extend_facts(edb_facts);

    // If we are just creating the type, insert it too
    if newly_inserted {
      self.types.insert(relation.to_string(), ty);
    }

    // Success
    Ok(())
  }

  pub fn add_annotated_disjunction<F>(&mut self, relation: &str, facts: Vec<(C::InputTag, F)>) -> Result<(), EDBError>
  where
    F: Clone + Into<Tuple>,
  {
    // If there is no fact, we do nothing
    if facts.len() == 0 {
      return Ok(());
    }

    // We get the type stored in the EDB. If there is no such type, we use the type of the first fact
    let mut newly_inserted = false;
    let ty = self.types.get(relation).cloned().unwrap_or_else(|| {
      newly_inserted = true;
      let t: Tuple = facts[0].1.clone().into();
      TupleType::type_of(&t)
    });

    // Turn facts into a set of EDBFact
    let edb_facts = self
      .process_facts(&ty, facts.into_iter().map(|(tag, tup)| (Some(tag), tup)))
      .map_err(|t| {
        if newly_inserted {
          EDBError::TypeMismatch {
            expected: ty.clone(),
            found: TupleType::type_of(&t),
            actual: t,
          }
        } else {
          EDBError::RelationTypeError {
            relation: relation.to_string(),
            expected: ty.clone(),
            found: TupleType::type_of(&t),
            actual: t,
          }
        }
      })?;

    // Get the relation
    let rel = self.relations.entry(relation.to_string()).or_default();

    // First insert the disjunction
    rel.add_disjunction((rel.facts.len()..(rel.facts.len() + edb_facts.len())).collect::<Vec<_>>());

    // Then add the facts into the relation
    rel.extend_facts(edb_facts);

    // If we are just creating the type, insert it too
    if newly_inserted {
      self.types.insert(relation.to_string(), ty);
    }

    // Success
    Ok(())
  }
}
