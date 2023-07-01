use serde::*;

use crate::common::entity;

use super::*;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum EntityNode {
  Expr(Expr),
  Object(Object),
}

pub type Entity = AstNode<EntityNode>;

impl Entity {
  /// Create a new constant entity
  pub fn constant(c: Constant) -> Self {
    Self::default(EntityNode::Expr(Expr::Constant(c)))
  }

  /// Checks if the entity is a simple constant
  pub fn is_constant(&self) -> bool {
    match &self.node {
      EntityNode::Expr(e) => e.is_constant(),
      _ => false,
    }
  }

  /// Get the constant if the entity is a simple constant
  pub fn get_constant(&self) -> Option<&Constant> {
    match &self.node {
      EntityNode::Expr(e) => e.get_constant(),
      _ => None,
    }
  }

  /// Checks if the entity has variable inside
  pub fn has_variable(&self) -> bool {
    match &self.node {
      EntityNode::Expr(e) => e.has_variable(),
      EntityNode::Object(o) => o.has_variable(),
    }
  }

  /// Get the location of the first non-constant in the entity
  pub fn get_first_non_constant_location<F>(&self, is_constant: &F) -> Option<&AstNodeLocation>
  where
    F: Fn(&Variable) -> bool,
  {
    match &self.node {
      EntityNode::Expr(e) => e.get_first_non_constant_location(is_constant),
      EntityNode::Object(o) => o.get_first_non_constant_location(is_constant),
    }
  }

  pub fn to_facts(&self) -> (Vec<EntityFact>, Constant) {
    self.to_facts_with_constant_variables(|_| None)
  }

  pub fn to_facts_with_constant_variables<F>(&self, f: F) -> (Vec<EntityFact>, Constant)
  where
    F: Fn(&Variable) -> Option<Constant>,
  {
    fn helper<F>(entity: &Entity, facts: &mut Vec<EntityFact>, f: &F) -> Constant
    where
      F: Fn(&Variable) -> Option<Constant>,
    {
      // Check whether we need to recurse
      match &entity.node {
        EntityNode::Expr(e) => {
          if let Some(c) = e.get_constant() {
            c.clone()
          } else if let Some(v) = e.get_variable() {
            if let Some(c) = f(v) {
              c
            } else {
              panic!("[Internal Error] Found non-constant variable in ")
            }
          } else {
            panic!("[Internal Error] Should contain only constant or constant variables")
          }
        }
        EntityNode::Object(obj) => {
          let functor = obj.functor().clone_without_location_id();
          let args = obj.iter_args().map(|a| helper(a, facts, f)).collect::<Vec<_>>();

          // Create a hash value
          let raw_id = entity::encode_entity(functor.name(), args.iter().map(|a| &a.node));

          // Create a constant ID of the hash value
          let id = Constant {
            loc: obj.location().clone(),
            node: ConstantNode::Entity(raw_id),
          };

          // Create the entity fact and store it inside the storage
          let entity_fact = EntityFact {
            functor,
            id: id.clone(),
            args,
            loc: obj.location().clone(),
          };
          facts.push(entity_fact);

          // Return the ID
          id
        }
      }
    }

    let mut facts = Vec::new();
    let constant = helper(self, &mut facts, &f);
    (facts, constant)
  }
}

#[derive(Clone, Debug, Serialize)]
pub struct EntityFact {
  pub functor: Identifier,
  pub id: Constant,
  pub args: Vec<Constant>,
  pub loc: AstNodeLocation,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ObjectNode {
  pub functor: Identifier,
  pub args: Vec<Entity>,
}

pub type Object = AstNode<ObjectNode>;

impl Object {
  pub fn has_variable(&self) -> bool {
    self.node.args.iter().any(|a| a.has_variable())
  }

  pub fn functor(&self) -> &Identifier {
    &self.node.functor
  }

  pub fn functor_mut(&mut self) -> &mut Identifier {
    &mut self.node.functor
  }

  pub fn functor_name(&self) -> &str {
    self.node.functor.name()
  }

  pub fn iter_args(&self) -> impl Iterator<Item = &Entity> {
    self.node.args.iter()
  }

  pub fn iter_args_mut(&mut self) -> impl Iterator<Item = &mut Entity> {
    self.node.args.iter_mut()
  }

  pub fn get_first_non_constant_location<F>(&self, is_constant: &F) -> Option<&AstNodeLocation>
  where
    F: Fn(&Variable) -> bool,
  {
    for arg in self.iter_args() {
      if let Some(loc) = arg.get_first_non_constant_location(is_constant) {
        return Some(loc);
      }
    }
    None
  }
}
