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
//! ``` ignore
//! #[derive(Clone)]
//! pub struct YOUR_FF;
//!
//! impl ForeignFunction for YOUR_FF {
//!   ...
//! }
//! ```
//!
//! Make sure the type information and the function implementation is correctly
//! specified.
//! Then we add the function to the standard library.
//! In `ForeignFunctionRegistry::std`, we add the following line
//!
//! ``` ignore
//! registry.register(ffs::YOUR_FF);
//! ```
//!
//! After this, the function will be available in the standard library

use std::collections::*;
use std::convert::*;

use dyn_clone::DynClone;

use super::type_family::*;
use super::value::*;
use super::value_type::*;

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
  fn execute(&self, args: Vec<Value>) -> Option<Value>;

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
    registry.register(ffs::Abs).unwrap();
    registry.register(ffs::Sin).unwrap();
    registry.register(ffs::Cos).unwrap();
    registry.register(ffs::Tan).unwrap();
    registry.register(ffs::Max).unwrap();
    registry.register(ffs::Min).unwrap();
    registry.register(ffs::StringConcat).unwrap();
    registry.register(ffs::StringLength).unwrap();
    registry.register(ffs::StringCharAt).unwrap();
    registry.register(ffs::Substring).unwrap();
    registry.register(ffs::Hash).unwrap();

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

/// A library of pre-implemented foreign functions
pub mod ffs {
  use super::*;

  /// Absolute value foreign function
  ///
  /// ``` scl
  /// extern fn $abs<T: Number>(x: T) -> T
  /// ```
  #[derive(Clone)]
  pub struct Abs;

  impl ForeignFunction for Abs {
    fn name(&self) -> String {
      "abs".to_string()
    }

    fn num_generic_types(&self) -> usize {
      1
    }

    fn generic_type_family(&self, i: usize) -> TypeFamily {
      assert_eq!(i, 0);
      TypeFamily::Number
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
        // Signed integers, take absolute
        Value::I8(f) => Some(Value::I8(f.abs())),
        Value::I16(f) => Some(Value::I16(f.abs())),
        Value::I32(f) => Some(Value::I32(f.abs())),
        Value::I64(f) => Some(Value::I64(f.abs())),
        Value::I128(f) => Some(Value::I128(f.abs())),
        Value::ISize(f) => Some(Value::ISize(f.abs())),

        // Unsigned integers, directly return
        Value::U8(f) => Some(Value::U8(f)),
        Value::U16(f) => Some(Value::U16(f)),
        Value::U32(f) => Some(Value::U32(f)),
        Value::U64(f) => Some(Value::U64(f)),
        Value::U128(f) => Some(Value::U128(f)),
        Value::USize(f) => Some(Value::USize(f)),

        // Floating points, take absolute
        Value::F32(f) => Some(Value::F32(f.abs())),
        Value::F64(f) => Some(Value::F64(f.abs())),
        _ => panic!("should not happen; input variable to abs should be a number"),
      }
    }
  }

  /// Floating point function
  pub trait UnaryFloatFunction: Clone {
    fn name(&self) -> String;

    fn execute_f32(&self, arg: f32) -> f32;

    fn execute_f64(&self, arg: f64) -> f64;
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
        Value::F32(f) => Some(Value::F32(self.execute_f32(f))),
        Value::F64(f) => Some(Value::F64(self.execute_f64(f))),
        _ => panic!("Expect floating point input"),
      }
    }
  }

  /// Sin value foreign function
  ///
  /// ``` scl
  /// extern fn $sin<T: Float>(x: T) -> T
  /// ```
  #[derive(Clone)]
  pub struct Sin;

  impl UnaryFloatFunction for Sin {
    fn name(&self) -> String {
      "sin".to_string()
    }

    fn execute_f32(&self, arg: f32) -> f32 {
      arg.sin()
    }

    fn execute_f64(&self, arg: f64) -> f64 {
      arg.sin()
    }
  }

  /// Cos value foreign function
  ///
  /// ``` scl
  /// extern fn $cos<T: Float>(x: T) -> T
  /// ```
  #[derive(Clone)]
  pub struct Cos;

  impl UnaryFloatFunction for Cos {
    fn name(&self) -> String {
      "cos".to_string()
    }

    fn execute_f32(&self, arg: f32) -> f32 {
      arg.cos()
    }

    fn execute_f64(&self, arg: f64) -> f64 {
      arg.cos()
    }
  }

  /// Tan value foreign function
  ///
  /// ``` scl
  /// extern fn $tan<T: Float>(x: T) -> T
  /// ```
  #[derive(Clone)]
  pub struct Tan;

  impl UnaryFloatFunction for Tan {
    fn name(&self) -> String {
      "tan".to_string()
    }

    fn execute_f32(&self, arg: f32) -> f32 {
      arg.tan()
    }

    fn execute_f64(&self, arg: f64) -> f64 {
      arg.tan()
    }
  }

  /// Substring
  ///
  /// ``` scl
  /// extern fn $substring(s: String, begin: usize, end: usize?) -> String
  /// ```
  #[derive(Clone)]
  pub struct Substring;

  impl ForeignFunction for Substring {
    fn name(&self) -> String {
      "substring".to_string()
    }

    fn num_static_arguments(&self) -> usize {
      2
    }

    fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
      match i {
        0 => ForeignFunctionParameterType::BaseType(ValueType::String),
        1 => ForeignFunctionParameterType::BaseType(ValueType::USize),
        _ => panic!("No argument {}", i),
      }
    }

    fn num_optional_arguments(&self) -> usize {
      1
    }

    fn optional_argument_type(&self, _: usize) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::BaseType(ValueType::USize)
    }

    fn return_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::BaseType(ValueType::String)
    }

    fn execute(&self, args: Vec<Value>) -> Option<Value> {
      if args.len() == 2 {
        match (&args[0], &args[1]) {
          (Value::String(s), Value::USize(i)) => Some(Value::String(s[*i..].to_string())),
          _ => panic!("Invalid arguments"),
        }
      } else {
        match (&args[0], &args[1], &args[2]) {
          (Value::String(s), Value::USize(i), Value::USize(j)) => Some(Value::String(s[*i..*j].to_string())),
          _ => panic!("Invalid arguments"),
        }
      }
    }
  }

  /// String concat
  ///
  /// ``` scl
  /// extern fn $string_concat(s: String...) -> String
  /// ```
  #[derive(Clone)]
  pub struct StringConcat;

  impl ForeignFunction for StringConcat {
    fn name(&self) -> String {
      "string_concat".to_string()
    }

    fn has_variable_arguments(&self) -> bool {
      true
    }

    fn variable_argument_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::BaseType(ValueType::String)
    }

    fn return_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::BaseType(ValueType::String)
    }

    fn execute(&self, args: Vec<Value>) -> Option<Value> {
      let mut result = "".to_string();
      for arg in args {
        match arg {
          Value::String(s) => {
            result += &s;
          }
          _ => panic!("Argument is not string"),
        }
      }
      Some(Value::String(result))
    }
  }

  /// String length
  ///
  /// ``` scl
  /// extern fn $string_length(s: String) -> usize
  /// ```
  #[derive(Clone)]
  pub struct StringLength;

  impl ForeignFunction for StringLength {
    fn name(&self) -> String {
      "string_length".to_string()
    }

    fn num_static_arguments(&self) -> usize {
      1
    }

    fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
      assert_eq!(i, 0);
      ForeignFunctionParameterType::BaseType(ValueType::String)
    }

    fn return_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::BaseType(ValueType::USize)
    }

    fn execute(&self, args: Vec<Value>) -> Option<Value> {
      match &args[0] {
        Value::String(s) => Some(Value::USize(s.len())),
        Value::Str(s) => Some(Value::USize(s.len())),
        _ => None,
      }
    }
  }

  /// String char at
  ///
  /// ``` scl
  /// extern fn $string_chat_at(s: String, i: usize) -> char
  /// ```
  #[derive(Clone)]
  pub struct StringCharAt;

  impl ForeignFunction for StringCharAt {
    fn name(&self) -> String {
      "string_char_at".to_string()
    }

    fn num_static_arguments(&self) -> usize {
      2
    }

    fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
      match i {
        0 => ForeignFunctionParameterType::BaseType(ValueType::String),
        1 => ForeignFunctionParameterType::BaseType(ValueType::USize),
        _ => panic!("Invalid {}-th argument", i),
      }
    }

    fn return_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::BaseType(ValueType::Char)
    }

    fn execute(&self, args: Vec<Value>) -> Option<Value> {
      match (&args[0], &args[1]) {
        (Value::String(s), Value::USize(i)) => s.chars().skip(*i).next().map(Value::Char),
        (Value::Str(s), Value::USize(i)) => s.chars().skip(*i).next().map(Value::Char),
        _ => None,
      }
    }
  }

  /// Hash
  ///
  /// ``` scl
  /// extern fn $hash(x: Any...) -> u64
  /// ```
  #[derive(Clone)]
  pub struct Hash;

  impl ForeignFunction for Hash {
    fn name(&self) -> String {
      "hash".to_string()
    }

    fn has_variable_arguments(&self) -> bool {
      true
    }

    fn variable_argument_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::TypeFamily(TypeFamily::Any)
    }

    fn return_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::BaseType(ValueType::U64)
    }

    fn execute(&self, args: Vec<Value>) -> Option<Value> {
      use std::collections::hash_map::DefaultHasher;
      use std::hash::{Hash, Hasher};
      let mut s = DefaultHasher::new();
      args.hash(&mut s);
      Some(s.finish().into())
    }
  }

  /// Max
  ///
  /// ``` scl
  /// extern fn $max<T: Number>(x: T...) -> T
  /// ```
  #[derive(Clone)]
  pub struct Max;

  impl Max {
    fn dyn_max<T: PartialOrd>(args: Vec<Value>) -> Option<T> where Value: TryInto<T> {
      let mut iter = args.into_iter();
      let mut curr_max: T = iter.next()?.try_into().ok()?;
      while let Some(next_elem) = iter.next() {
        let next_elem = next_elem.try_into().ok()?;
        if next_elem > curr_max {
          curr_max = next_elem;
        }
      }
      Some(curr_max)
    }
  }

  impl ForeignFunction for Max {
    fn name(&self) -> String {
      "max".to_string()
    }

    fn num_generic_types(&self) -> usize {
      1
    }

    fn generic_type_family(&self, i: usize) -> TypeFamily {
      assert_eq!(i, 0);
      TypeFamily::Number
    }

    fn has_variable_arguments(&self) -> bool {
      true
    }

    fn variable_argument_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::Generic(0)
    }

    fn return_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::Generic(0)
    }

    fn execute(&self, args: Vec<Value>) -> Option<Value> {
      let rt = self.infer_return_type(&args);
      match rt {
        ValueType::I8 => Self::dyn_max(args).map(Value::I8),
        ValueType::I16 => Self::dyn_max(args).map(Value::I16),
        ValueType::I32 => Self::dyn_max(args).map(Value::I32),
        ValueType::I64 => Self::dyn_max(args).map(Value::I64),
        ValueType::I128 => Self::dyn_max(args).map(Value::I128),
        ValueType::ISize => Self::dyn_max(args).map(Value::ISize),
        ValueType::U8 => Self::dyn_max(args).map(Value::U8),
        ValueType::U16 => Self::dyn_max(args).map(Value::U16),
        ValueType::U32 => Self::dyn_max(args).map(Value::U32),
        ValueType::U64 => Self::dyn_max(args).map(Value::U64),
        ValueType::U128 => Self::dyn_max(args).map(Value::U128),
        ValueType::USize => Self::dyn_max(args).map(Value::USize),
        ValueType::F32 => Self::dyn_max(args).map(Value::F32),
        ValueType::F64 => Self::dyn_max(args).map(Value::F64),
        _ => None,
      }
    }
  }

  /// Min
  ///
  /// ``` scl
  /// extern fn $min<T: Number>(x: T...) -> T
  /// ```
  #[derive(Clone)]
  pub struct Min;

  impl Min {
    fn dyn_min<T: PartialOrd>(args: Vec<Value>) -> Option<T> where Value: TryInto<T> {
      let mut iter = args.into_iter();
      let mut curr_min: T = iter.next()?.try_into().ok()?;
      while let Some(next_elem) = iter.next() {
        let next_elem = next_elem.try_into().ok()?;
        if next_elem < curr_min {
          curr_min = next_elem;
        }
      }
      Some(curr_min)
    }
  }

  impl ForeignFunction for Min {
    fn name(&self) -> String {
      "min".to_string()
    }

    fn num_generic_types(&self) -> usize {
      1
    }

    fn generic_type_family(&self, i: usize) -> TypeFamily {
      assert_eq!(i, 0);
      TypeFamily::Number
    }

    fn has_variable_arguments(&self) -> bool {
      true
    }

    fn variable_argument_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::Generic(0)
    }

    fn return_type(&self) -> ForeignFunctionParameterType {
      ForeignFunctionParameterType::Generic(0)
    }

    fn execute(&self, args: Vec<Value>) -> Option<Value> {
      let rt = self.infer_return_type(&args);
      match rt {
        ValueType::I8 => Self::dyn_min(args).map(Value::I8),
        ValueType::I16 => Self::dyn_min(args).map(Value::I16),
        ValueType::I32 => Self::dyn_min(args).map(Value::I32),
        ValueType::I64 => Self::dyn_min(args).map(Value::I64),
        ValueType::I128 => Self::dyn_min(args).map(Value::I128),
        ValueType::ISize => Self::dyn_min(args).map(Value::ISize),
        ValueType::U8 => Self::dyn_min(args).map(Value::U8),
        ValueType::U16 => Self::dyn_min(args).map(Value::U16),
        ValueType::U32 => Self::dyn_min(args).map(Value::U32),
        ValueType::U64 => Self::dyn_min(args).map(Value::U64),
        ValueType::U128 => Self::dyn_min(args).map(Value::U128),
        ValueType::USize => Self::dyn_min(args).map(Value::USize),
        ValueType::F32 => Self::dyn_min(args).map(Value::F32),
        ValueType::F64 => Self::dyn_min(args).map(Value::F64),
        _ => None,
      }
    }
  }
}
