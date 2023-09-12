//! # Foreign function interface
//!
//! Often times there are operations that Scallop alone cannot express.
//! We rely on foreign functions to perform such computations.
//! Foreign functions here are defined to be discrete functions that take
//! in an arbitrary amount of Scallop values and return one Scallop value.
//! The type signatures of foreign functions need to be declared up-front and
//! statically checked to be correct.
//! It is allowed for foreign functions to be partial (i.e. produce errors),
//! in which case the computation will be omitted and neglected.
//! All foreign functions need to be pure; no interior state is allowed and
//! given the same input, only one single output can be produced.
//!
//! ## Type declarations
//!
//! Scallop supports foreign functions defined in the following manner
//!
//! ``` scl
//! extern fn $FUNC(VAR: TY, ...) -> TY
//! ```
//!
//! One example is the function to get the length of a given string
//!
//! ``` scl
//! extern fn $string_length(String) -> usize
//! ```
//!
//! FFI supports generic types, which is denoted using angle brackets`<` `>`:
//!
//! ``` scl
//! extern fn $abs<T: Number>(x: T) -> T
//! ```
//!
//! Note that in the above example we have restricted the input type to `Number`
//! For other type families checkout the `TypeFamily` class.
//! If a type family is not provided, it is automatically assigned `Any`.
//! The argument types can be 1) any base type, 2) any type families, or 3) a generic type
//! parameter.
//! But the result type can only be a base type or a generic type parameter.
//!
//! FFI also supports variable arguments, using the following syntax
//!
//! ``` scl
//! extern fn $string_concat(s: String...) -> String
//! extern fn $hash(v: Any...) -> u64
//! ```
//!
//! In the above example, `$string_concat` function takes in 0 or more String
//! arguments.
//!
//! We can have optional arguments as well:
//!
//! ``` scl
//! extern fn $substring(s: String, begin: usize, end: usize?) -> String
//! ```
//!
//! The optional arguments need to be passed to the function in its defined order.
//! It has to be placed after
//! Note that optional arguments and variable arguments cannot co-exist.
//!
//! ## Implement a new built-in foreign function and add that to Scallop stdlib
//!
//! First, head to the module `ffs` and create a new foreign function.
//! Make sure it can be cloned, and then we can implement the `ForeignFunction` trait
//! for it.
//!
//! Make sure the type information and the function implementation is correctly
//! specified.
//! Then we add the function to the standard library.
//! In `ForeignFunctionRegistry::std`, we add the following line `registry.register(ffs::YOUR_FF);`.
//!
//! After this, the function will be available in the standard library

use std::collections::*;

use dyn_clone::DynClone;

use super::foreign_functions as ffs;
use super::type_family::*;
use super::value::*;
use super::value_type::*;

use crate::runtime::env::*;

/// A type used for defining a foreign function.
///
/// It could be generic or a specific base type.
#[derive(Clone, Debug)]
pub enum ForeignFunctionParameterType {
  /// A generic type parameter, referenced by its ID
  Generic(usize),

  /// A type family
  TypeFamily(TypeFamily),

  /// A base type
  BaseType(ValueType),
}

impl std::fmt::Display for ForeignFunctionParameterType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::BaseType(t) => t.fmt(f),
      Self::TypeFamily(tf) => tf.fmt(f),
      Self::Generic(i) => f.write_fmt(format_args!("T{}", i)),
    }
  }
}

#[derive(Clone, Debug)]
pub enum ArgumentKind {
  Static,
  Optional,
  Variable,
}

#[derive(Clone, Debug)]
pub enum ForeignFunctionError {
  AlreadyExisted { name: String },
  UnusedGenericType { id: usize },
  UnboundedReturnGenericType { id: usize },
  UnboundedReturnTypeFamily { family: TypeFamily },
  ConflictDefinition { name: String },
}

impl std::fmt::Display for ForeignFunctionError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::AlreadyExisted { name } => f.write_fmt(format_args!("Foreign function `{}` already exists", name)),
      Self::UnusedGenericType { id } => f.write_fmt(format_args!("Generic type #{} is unused", id)),
      Self::UnboundedReturnGenericType { id } => {
        f.write_fmt(format_args!("The returned generic type #{} is unbounded by input", id))
      }
      Self::UnboundedReturnTypeFamily { family } => {
        f.write_fmt(format_args!("Returning type family `{}` is disallowed", family))
      }
      Self::ConflictDefinition { name } => {
        f.write_fmt(format_args!("Conflicting definition for foreign function `${}`", name))
      }
    }
  }
}

/// The trait for defining a foreign function
pub trait ForeignFunction: DynClone {
  /// The name of the foreign function
  fn name(&self) -> String;

  /// Get number of generic types; generic types will be referenced by their ID
  fn num_generic_types(&self) -> usize {
    0
  }

  /// Get the type family of one generic type parameter
  #[allow(unused_variables)]
  fn generic_type_family(&self, i: usize) -> TypeFamily {
    panic!("There is no generic type parameter")
  }

  /// Return an iterator of all the generic type parameters
  fn generic_type_parameters(&self) -> Vec<TypeFamily> {
    (0..self.num_generic_types())
      .map(|i| self.generic_type_family(i))
      .collect()
  }

  /// Get the number of static (required) arguments
  fn num_static_arguments(&self) -> usize {
    0
  }

  /// Get the type of the i-th static argument; panic if there is no i-th argument
  #[allow(unused_variables)]
  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    panic!("There is no static argument")
  }

  /// Get the types of static arguments
  fn static_argument_types(&self) -> Vec<ForeignFunctionParameterType> {
    (0..self.num_static_arguments())
      .map(|i| self.static_argument_type(i))
      .collect()
  }

  /// Get the number of optional arguments
  fn num_optional_arguments(&self) -> usize {
    0
  }

  /// Get the type of the i-th optional argument. Note that the index for static and optional arguments are counted *separately*.
  /// Panic if there is no i-th optional argument
  #[allow(unused_variables)]
  fn optional_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    panic!("There is no optional argument")
  }

  /// Get the types of optional arguments
  fn optional_argument_types(&self) -> Vec<ForeignFunctionParameterType> {
    (0..self.num_optional_arguments())
      .map(|i| self.optional_argument_type(i))
      .collect()
  }

  /// Check if the function has variable arguments
  fn has_variable_arguments(&self) -> bool {
    false
  }

  /// Get the type of the variable argument; panic if there is no variable argument
  fn variable_argument_type(&self) -> ForeignFunctionParameterType {
    panic!("There is no variable argument")
  }

  /// Get the i-th argument type
  fn argument_type(&self, i: usize) -> Option<ForeignFunctionParameterType> {
    if i < self.num_static_arguments() {
      Some(self.static_argument_type(i).clone())
    } else {
      let optional_argument_id = i - self.num_static_arguments();
      if optional_argument_id < self.num_optional_arguments() {
        Some(self.optional_argument_type(optional_argument_id).clone())
      } else if let Some(var_arg_type) = &self.optional_variable_argument_type() {
        Some(var_arg_type.clone())
      } else {
        None
      }
    }
  }

  /// Get an `Option` containing the varibale argument type, if presented
  fn optional_variable_argument_type(&self) -> Option<ForeignFunctionParameterType> {
    if self.has_variable_arguments() {
      Some(self.variable_argument_type())
    } else {
      None
    }
  }

  /// Get the function return type
  fn return_type(&self) -> ForeignFunctionParameterType;

  /// Execute the function given arguments
  ///
  /// We assume that the given arguments obey the type declaration.
  /// In case error happens, we return `None` as the result.
  #[allow(unused_variables)]
  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    panic!(
      "[Internal Error] Missing execute function in the foreign function `{}`",
      self.name()
    )
  }

  /// Execute the function given arguments and a runtime environment
  #[allow(unused_variables)]
  fn execute_with_env(&self, env: &RuntimeEnvironment, args: Vec<Value>) -> Option<Value> {
    self.execute(args)
  }

  /// Get all the arguments
  fn arguments(&self) -> Vec<(ArgumentKind, ForeignFunctionParameterType)> {
    let mut args = vec![];

    // Add static arguments
    args.extend((0..self.num_static_arguments()).map(|i| (ArgumentKind::Static, self.static_argument_type(i))));

    // Add optional arguments
    args.extend((0..self.num_optional_arguments()).map(|i| (ArgumentKind::Optional, self.optional_argument_type(i))));

    // Add variable length argument
    if self.has_variable_arguments() {
      args.push((ArgumentKind::Variable, self.variable_argument_type()));
    }

    args
  }

  /// Infer the function return type from concrete input arguments
  fn infer_return_type(&self, input_args: &Vec<Value>) -> ValueType {
    // Get the exact return type from argument type
    match self.return_type() {
      ForeignFunctionParameterType::BaseType(ty) => ty,
      ForeignFunctionParameterType::Generic(generic_id) => {
        for i in 0..input_args.len() {
          let arg_ty = self.argument_type(i).unwrap();
          match arg_ty {
            ForeignFunctionParameterType::Generic(curr_generic_id) if generic_id == curr_generic_id => {
              return input_args[i].value_type()
            }
            _ => {}
          }
        }
        panic!("Should not happen; return type is not bounded by input type")
      }
      ForeignFunctionParameterType::TypeFamily(_) => {
        panic!("Should not happen; return type cannot be type family")
      }
    }
  }

  /// Check if the function's type is well-formed; if not, a `ForeignFunctionError` is returned
  fn check_type_well_formed(&self) -> Result<(), ForeignFunctionError> {
    let arg_types = self.arguments();
    let ret_type = self.return_type();
    let mut to_check_parameters: Vec<_> = arg_types.iter().map(|(_, t)| t).collect();
    to_check_parameters.push(&ret_type);

    // First check for each generic type, whether it is used
    for generic_type_id in 0..self.num_generic_types() {
      let mut used = false;
      for param in &to_check_parameters {
        match param {
          ForeignFunctionParameterType::Generic(i) if *i == generic_type_id => {
            used = true;
          }
          _ => {}
        }
      }
      if !used {
        return Err(ForeignFunctionError::UnusedGenericType { id: generic_type_id });
      }
    }

    // Then check the return type
    match self.return_type() {
      ForeignFunctionParameterType::BaseType(_) => {}
      ForeignFunctionParameterType::Generic(id) => {
        let mut bounded = false;
        for (_, arg_type) in arg_types {
          match arg_type {
            ForeignFunctionParameterType::Generic(arg_id) if arg_id == id => {
              bounded = true;
              break;
            }
            _ => {}
          }
        }
        if !bounded {
          return Err(ForeignFunctionError::UnboundedReturnGenericType { id });
        }
      }
      ForeignFunctionParameterType::TypeFamily(tf) => {
        return Err(ForeignFunctionError::UnboundedReturnTypeFamily { family: tf.clone() })
      }
    }

    Ok(())
  }
}

/// A dynamic foreign function that can hold any static foreign function
pub struct DynamicForeignFunction {
  ff: Box<dyn ForeignFunction + Send + Sync>,
}

impl DynamicForeignFunction {
  pub fn new<F: ForeignFunction + Send + Sync + 'static>(f: F) -> Self {
    Self { ff: Box::new(f) }
  }
}

impl std::fmt::Debug for DynamicForeignFunction {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("DynamicForeignFunction")
      .field("name", &self.name())
      .field("generic_type_parameters", &self.generic_type_parameters())
      .field("static_arg_types", &self.static_argument_types())
      .field("optional_arg_types", &self.optional_argument_types())
      .field("variable_arg_type", &self.optional_variable_argument_type())
      .field("return_type", &self.return_type())
      .finish()
  }
}

impl std::fmt::Display for DynamicForeignFunction {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("fn ${}", self.name()))?;

    // Write generic types
    if self.num_generic_types() > 0 {
      f.write_str("<")?;
      for i in 0..self.num_generic_types() {
        f.write_fmt(format_args!("T{}", i))?;
        let tf = self.generic_type_family(i);
        if tf != TypeFamily::Any {
          f.write_fmt(format_args!(": {}", tf))?;
        }
        if i < self.num_generic_types() - 1 {
          f.write_str(", ")?;
        }
      }
      f.write_str(">")?;
    }

    // Write left parenthesis
    f.write_str("(")?;

    // Write static variables
    for i in 0..self.num_static_arguments() {
      self.static_argument_type(i).fmt(f)?;
      if i < self.num_static_arguments() - 1 {
        f.write_str(", ")?;
      }
    }

    // Write optional variables
    if self.num_optional_arguments() > 0 {
      if self.num_static_arguments() > 0 {
        f.write_str(", ")?;
      }
      for i in 0..self.num_optional_arguments() {
        self.static_argument_type(i).fmt(f)?;
        f.write_str("?")?;
        if i < self.num_static_arguments() - 1 {
          f.write_str(", ")?;
        }
      }
    }

    // Write variable arguments
    if self.has_variable_arguments() {
      if self.num_static_arguments() > 0 {
        f.write_str(", ")?;
      }
      self.variable_argument_type().fmt(f)?;
      f.write_str("...")?;
    }

    // Write right parenthesis and result type
    f.write_fmt(format_args!(") -> {}", self.return_type()))?;

    Ok(())
  }
}

impl Clone for DynamicForeignFunction {
  fn clone(&self) -> Self {
    Self {
      ff: dyn_clone::clone_box(&*self.ff),
    }
  }
}

impl ForeignFunction for DynamicForeignFunction {
  fn name(&self) -> String {
    self.ff.name()
  }

  fn num_generic_types(&self) -> usize {
    self.ff.num_generic_types()
  }

  fn generic_type_family(&self, i: usize) -> TypeFamily {
    self.ff.generic_type_family(i)
  }

  fn num_static_arguments(&self) -> usize {
    self.ff.num_static_arguments()
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    self.ff.static_argument_type(i)
  }

  fn num_optional_arguments(&self) -> usize {
    self.ff.num_optional_arguments()
  }

  fn optional_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    self.ff.optional_argument_type(i)
  }

  fn has_variable_arguments(&self) -> bool {
    self.ff.has_variable_arguments()
  }

  fn variable_argument_type(&self) -> ForeignFunctionParameterType {
    self.ff.variable_argument_type()
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    self.ff.return_type()
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    self.ff.execute(args)
  }

  fn execute_with_env(&self, env: &RuntimeEnvironment, args: Vec<Value>) -> Option<Value> {
    self.ff.execute_with_env(env, args)
  }
}

/// Dynamic foreign function registry
///
/// A structure to hold all dynamic foreign functions
#[derive(Debug, Clone)]
pub struct ForeignFunctionRegistry {
  registry: HashMap<String, DynamicForeignFunction>,
}

impl ForeignFunctionRegistry {
  /// Create an empty FF registry
  pub fn new() -> Self {
    Self {
      registry: HashMap::new(),
    }
  }

  /// Create the standard foreign function library
  pub fn std() -> Self {
    let mut registry = Self::new();

    // Register all supported foreign functions
    // Note: unwrap is OK since
    // 1. we are starting from fresh registry;
    // 2. that all functions here have distinct names;
    // 3. all our functions are checked to be have correct types.

    // Arithmetic
    registry.register(ffs::Abs).unwrap();
    registry.register(ffs::Floor).unwrap();
    registry.register(ffs::Ceil).unwrap();
    registry.register(ffs::Exp).unwrap();
    registry.register(ffs::Exp2).unwrap();
    registry.register(ffs::Log).unwrap();
    registry.register(ffs::Log2).unwrap();
    registry.register(ffs::Pow).unwrap();
    registry.register(ffs::Powf).unwrap();
    registry.register(ffs::Sin).unwrap();
    registry.register(ffs::Cos).unwrap();
    registry.register(ffs::Tan).unwrap();
    registry.register(ffs::Asin).unwrap();
    registry.register(ffs::Acos).unwrap();
    registry.register(ffs::Atan).unwrap();
    registry.register(ffs::Atan2).unwrap();
    registry.register(ffs::Sign).unwrap();

    // Min/Max
    registry.register(ffs::Max).unwrap();
    registry.register(ffs::Min).unwrap();

    // String operations
    registry.register(ffs::StringConcat).unwrap();
    registry.register(ffs::StringLength).unwrap();
    registry.register(ffs::StringCharAt).unwrap();
    registry.register(ffs::Substring).unwrap();
    registry.register(ffs::Format).unwrap();
    registry.register(ffs::StringUpper).unwrap();
    registry.register(ffs::StringLower).unwrap();
    registry.register(ffs::StringIndexOf).unwrap();
    registry.register(ffs::StringReplace).unwrap();
    registry.register(ffs::StringTrim).unwrap();

    // DateTime operations
    registry.register(ffs::DateTimeDay).unwrap();
    registry.register(ffs::DateTimeMonth).unwrap();
    registry.register(ffs::DateTimeMonth0).unwrap();
    registry.register(ffs::DateTimeYear).unwrap();

    // Entity
    registry.register(ffs::ParseEntity).unwrap();

    // Hashing operation
    registry.register(ffs::Hash).unwrap();

    // Tensor operation
    registry.register(ffs::Dim).unwrap();
    registry.register(ffs::Dot).unwrap();

    registry
  }

  /// Register a new foreign function in this registry
  pub fn register<F: ForeignFunction + Send + Sync + 'static>(&mut self, f: F) -> Result<(), ForeignFunctionError> {
    let name = f.name();

    // Check if the registry already contains this function; we do not allow re-definition of foreign functions
    if self.contains(&name) {
      Err(ForeignFunctionError::AlreadyExisted { name })
    } else {
      // Check if the function is well formed
      f.check_type_well_formed()?;

      // Insert it into the registry
      self.registry.insert(name, DynamicForeignFunction::new(f));

      // Return ok
      Ok(())
    }
  }

  pub fn contains(&self, name: &str) -> bool {
    self.registry.contains_key(name)
  }

  /// Get a foreign function in this registry
  pub fn get(&self, name: &str) -> Option<&DynamicForeignFunction> {
    self.registry.get(name)
  }
}

impl<'a> IntoIterator for &'a ForeignFunctionRegistry {
  type IntoIter = std::collections::hash_map::Iter<'a, String, DynamicForeignFunction>;

  type Item = (&'a String, &'a DynamicForeignFunction);

  fn into_iter(self) -> Self::IntoIter {
    self.registry.iter()
  }
}

/// Floating point function
pub trait UnaryFloatFunction: Clone {
  fn name(&self) -> String;

  #[allow(unused_variables)]
  fn execute_f32(&self, arg: f32) -> f32 {
    0.0
  }

  fn execute_f32_partial(&self, arg: f32) -> Option<f32> {
    Some(self.execute_f32(arg))
  }

  #[allow(unused_variables)]
  fn execute_f64(&self, arg: f64) -> f64 {
    0.0
  }

  fn execute_f64_partial(&self, arg: f64) -> Option<f64> {
    Some(self.execute_f64(arg))
  }
}

impl<F: UnaryFloatFunction> ForeignFunction for F {
  fn name(&self) -> String {
    UnaryFloatFunction::name(self)
  }

  fn num_generic_types(&self) -> usize {
    1
  }

  fn generic_type_family(&self, i: usize) -> TypeFamily {
    assert_eq!(i, 0);
    TypeFamily::Float
  }

  fn num_static_arguments(&self) -> usize {
    1
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert_eq!(i, 0);
    ForeignFunctionParameterType::Generic(0)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::Generic(0)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match args[0] {
      Value::F32(f) => self.execute_f32_partial(f).map(Value::F32),
      Value::F64(f) => self.execute_f64_partial(f).map(Value::F64),
      _ => panic!("Expect floating point input"),
    }
  }
}
