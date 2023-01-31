use std::collections::*;

use crate::common::foreign_function::*;
use crate::common::value_type::*;

use super::*;

/// Argument type of a function, which could be a generic type parameter or a type set (including base type)
#[derive(Clone, Debug)]
pub enum FunctionArgumentType {
  Generic(usize),
  TypeSet(TypeSet),
}

impl From<ForeignFunctionParameterType> for FunctionArgumentType {
  fn from(value: ForeignFunctionParameterType) -> Self {
    match value {
      ForeignFunctionParameterType::BaseType(ty) => Self::TypeSet(TypeSet::base(ty)),
      ForeignFunctionParameterType::TypeFamily(ty) => Self::TypeSet(TypeSet::from(ty)),
      ForeignFunctionParameterType::Generic(i) => Self::Generic(i),
    }
  }
}

impl FunctionArgumentType {
  pub fn is_generic(&self) -> bool {
    match self {
      Self::Generic(_) => false,
      _ => true,
    }
  }
}

/// Return type of a function, which need to be a generic type parameter or a base type (cannot be type set)
#[derive(Clone, Debug)]
pub enum FunctionReturnType {
  Generic(usize),
  BaseType(ValueType),
}

impl From<ForeignFunctionParameterType> for FunctionReturnType {
  fn from(value: ForeignFunctionParameterType) -> Self {
    match value {
      ForeignFunctionParameterType::BaseType(ty) => Self::BaseType(ty),
      ForeignFunctionParameterType::Generic(i) => Self::Generic(i),
      _ => panic!("Return type cannot be of type family"),
    }
  }
}

/// The function type
#[derive(Clone, Debug)]
pub struct FunctionType {
  /// Generic type parameters
  pub generic_type_parameters: Vec<TypeSet>,

  /// Static argument types
  pub static_argument_types: Vec<FunctionArgumentType>,

  /// Optional argument types
  pub optional_argument_types: Vec<FunctionArgumentType>,

  /// Variable argument type
  pub variable_argument_type: Option<FunctionArgumentType>,

  /// Return type
  pub return_type: FunctionReturnType,
}

impl<F: ForeignFunction> From<&F> for FunctionType {
  fn from(f: &F) -> Self {
    Self {
      generic_type_parameters: f.generic_type_parameters().into_iter().map(TypeSet::from).collect(),
      static_argument_types: f
        .static_argument_types()
        .into_iter()
        .map(FunctionArgumentType::from)
        .collect(),
      optional_argument_types: f
        .optional_argument_types()
        .into_iter()
        .map(FunctionArgumentType::from)
        .collect(),
      variable_argument_type: f.optional_variable_argument_type().map(FunctionArgumentType::from),
      return_type: FunctionReturnType::from(f.return_type()),
    }
  }
}

impl FunctionType {
  /// Get the number of static arguments
  pub fn num_static_arguments(&self) -> usize {
    self.static_argument_types.len()
  }

  /// Get the number of optional arguments
  pub fn num_optional_arguments(&self) -> usize {
    self.optional_argument_types.len()
  }

  /// Check whether this function has variable arguments
  pub fn has_variable_arguments(&self) -> bool {
    self.variable_argument_type.is_some()
  }

  /// Check if the given `num_args` is acceptable by the function type
  pub fn is_valid_num_args(&self, num_args: usize) -> bool {
    // First, there should be at least `len(static_argument_types)` arguments
    if num_args < self.num_static_arguments() {
      return false;
    }

    // Then, we compute if there is a right amount of optional arguments
    let num_optional_args = num_args - self.num_static_arguments();
    if !self.has_variable_arguments() {
      // If there is no variable arguments, then the #provided optional arguments should be <= #expected optional arguments
      if num_optional_args > self.num_optional_arguments() {
        return false;
      }
    }

    true
  }

  /// Get the type of i-th argument
  pub fn type_of_ith_argument(&self, i: usize) -> Option<FunctionArgumentType> {
    if i < self.num_static_arguments() {
      Some(self.static_argument_types[i].clone())
    } else {
      let optional_argument_id = i - self.num_static_arguments();
      if optional_argument_id < self.num_optional_arguments() {
        Some(self.optional_argument_types[optional_argument_id].clone())
      } else if let Some(var_arg_type) = &self.variable_argument_type {
        Some(var_arg_type.clone())
      } else {
        None
      }
    }
  }
}

#[derive(Clone, Debug)]
pub struct FunctionTypeRegistry {
  pub function_types: HashMap<String, FunctionType>,
}

impl FunctionTypeRegistry {
  pub fn empty() -> Self {
    Self {
      function_types: HashMap::new(),
    }
  }

  pub fn from_foreign_function_registry(foreign_function_registry: &ForeignFunctionRegistry) -> Self {
    let mut type_registry = Self::empty();
    for (_, ff) in foreign_function_registry {
      let name = ff.name();
      let func_type = FunctionType::from(ff);
      type_registry.add_function_type(name, func_type);
    }
    type_registry
  }

  pub fn add_function_type(&mut self, name: String, f: FunctionType) {
    self.function_types.insert(name, f);
  }

  pub fn get(&self, function_name: &str) -> Option<&FunctionType> {
    self.function_types.get(function_name)
  }
}
