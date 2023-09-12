use crate::common::entity;

use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub enum Entity {
  Expr(Expr),
  Object(Object),
}

impl Entity {
  /// Convert the entity to a constant if it is one
  pub fn as_constant(&self) -> Option<&Constant> {
    self.as_expr().and_then(Expr::as_constant)
  }

  /// Checks if the entity is a simple constant
  pub fn is_constant(&self) -> bool {
    self.as_expr().map(Expr::is_constant).unwrap_or(false)
  }

  /// Checks if the entity has variable inside
  pub fn has_variable(&self) -> bool {
    match self {
      Entity::Expr(e) => e.has_variable(),
      Entity::Object(o) => o.has_variable(),
    }
  }

  /// Get the location of the first non-constant in the entity
  pub fn get_first_non_constant_location<F>(&self, is_constant: &F) -> Option<&NodeLocation>
  where
    F: Fn(&Variable) -> bool,
  {
    match self {
      Entity::Expr(e) => e.get_first_non_constant_location(is_constant),
      Entity::Object(o) => o.get_first_non_constant_location(is_constant),
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
      match entity {
        Entity::Expr(e) => {
          if let Some(c) = e.as_constant() {
            c.clone()
          } else if let Some(v) = e.as_variable() {
            if let Some(c) = f(v) {
              c
            } else {
              panic!("[Internal Error] Found non-constant variable in ")
            }
          } else {
            panic!("[Internal Error] Should contain only constant or constant variables")
          }
        }
        Entity::Object(obj) => {
          let functor = obj.functor().clone_without_location_id();
          let args = obj.iter_args().map(|a| helper(a, facts, f)).collect::<Vec<_>>();

          // Create a hash value
          let raw_id = entity::encode_entity(functor.name(), args.iter().map(|a| a));

          // Create a constant ID of the hash value
          let entity = EntityLiteral::new_with_loc(raw_id, obj.location().clone());
          let id = Constant::entity(entity);

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
  pub loc: NodeLocation,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub struct _Object {
  pub functor: Identifier,
  pub args: Vec<Entity>,
}

impl Object {
  pub fn has_variable(&self) -> bool {
    self.args().iter().any(|a| a.has_variable())
  }

  pub fn functor_name(&self) -> &str {
    self.functor().name()
  }

  pub fn get_first_non_constant_location<F>(&self, is_constant: &F) -> Option<&NodeLocation>
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
