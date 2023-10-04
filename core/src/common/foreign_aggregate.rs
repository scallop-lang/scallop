use std::collections::*;

use crate::runtime::dynamic::*;
use crate::runtime::env::RuntimeEnvironment;
use crate::runtime::provenance::*;

use super::type_family::*;
use super::value::*;
use super::value_type::*;

use super::foreign_aggregates::*;

#[derive(Clone, Debug)]
pub enum GenericTypeFamily {
  /// A unit tuple or a given type family
  UnitOr(Box<GenericTypeFamily>),

  /// An arbitrary tuple, can have variable lengths > 0
  NonEmptyTuple,

  /// A tuple, with elements length > 0
  NonEmptyTupleWithElements(Vec<GenericTypeFamily>),

  /// A base value type family
  TypeFamily(TypeFamily),
}

impl GenericTypeFamily {
  pub fn unit_or(g: Self) -> Self {
    Self::UnitOr(Box::new(g))
  }

  pub fn type_family(t: TypeFamily) -> Self {
    Self::TypeFamily(t)
  }

  pub fn non_empty_tuple() -> Self {
    Self::NonEmptyTuple
  }

  pub fn possibly_empty_tuple() -> Self {
    Self::unit_or(Self::non_empty_tuple())
  }

  pub fn non_empty_tuple_with_elements(elems: Vec<Self>) -> Self {
    assert!(elems.len() > 0, "elements must be non-empty");
    Self::NonEmptyTupleWithElements(elems)
  }

  pub fn possibly_empty_tuple_with_elements(elems: Vec<Self>) -> Self {
    Self::unit_or(Self::non_empty_tuple_with_elements(elems))
  }

  pub fn is_type_family(&self) -> bool {
    match self {
      Self::TypeFamily(_) => true,
      _ => false,
    }
  }

  pub fn as_type_family(&self) -> Option<&TypeFamily> {
    match self {
      Self::TypeFamily(tf) => Some(tf),
      _ => None,
    }
  }
}

#[derive(Clone, Debug)]
pub enum BindingTypes {
  /// A tuple type; arity-0 means unit
  TupleType(Vec<BindingType>),

  /// Depending on whether the generic type is evaluated to be unit, choose
  /// between the `then_type` or the `else_type`
  IfNotUnit {
    generic_type: String,
    then_type: Box<BindingTypes>,
    else_type: Box<BindingTypes>,
  },
}

impl BindingTypes {
  pub fn unit() -> Self {
    Self::TupleType(vec![])
  }

  pub fn generic(s: &str) -> Self {
    Self::TupleType(vec![BindingType::generic(s)])
  }

  pub fn value_type(v: ValueType) -> Self {
    Self::TupleType(vec![BindingType::value_type(v)])
  }

  pub fn if_not_unit(t: &str, t1: Self, t2: Self) -> Self {
    Self::IfNotUnit {
      generic_type: t.to_string(),
      then_type: Box::new(t1),
      else_type: Box::new(t2),
    }
  }

  pub fn empty_tuple() -> Self {
    Self::tuple(vec![])
  }

  pub fn tuple(elems: Vec<BindingType>) -> Self {
    Self::TupleType(elems)
  }
}

#[derive(Clone, Debug)]
pub enum BindingType {
  /// A generic type
  Generic(String),

  /// A single value type
  ValueType(ValueType),
}

impl BindingType {
  pub fn generic(s: &str) -> Self {
    Self::Generic(s.to_string())
  }

  pub fn value_type(value_type: ValueType) -> Self {
    Self::ValueType(value_type)
  }

  pub fn is_generic(&self) -> bool {
    match self {
      Self::Generic(_) => false,
      _ => true,
    }
  }

  pub fn as_value_type(&self) -> Option<&ValueType> {
    match self {
      Self::ValueType(vt) => Some(vt),
      _ => None,
    }
  }
}

#[derive(Clone, Debug)]
pub enum ParamType {
  Mandatory(ValueType),
  Optional(ValueType),
}

/// The type of an aggregator
///
/// ``` ignore
/// OUTPUT_TYPE := AGGREGATE<PARAM: FAMILY, ...>[ARG_TYPE](INPUT_TYPE)
/// ```
#[derive(Clone, Debug)]
pub struct AggregateType {
  pub generics: HashMap<String, GenericTypeFamily>,
  pub param_types: Vec<ParamType>,
  pub arg_type: BindingTypes,
  pub input_type: BindingTypes,
  pub output_type: BindingTypes,
  pub allow_exclamation_mark: bool,
}

pub trait Aggregate: Into<DynamicAggregate> {
  /// The concrete aggregator that this aggregate is instantiated into
  type Aggregator<P: Provenance>: Aggregator<P>;

  /// The name of the aggregate
  fn name(&self) -> String;

  /// The type of the aggregate
  fn aggregate_type(&self) -> AggregateType;

  /// Instantiate the aggregate into an aggregator with the given parameters
  fn instantiate<P: Provenance>(
    &self,
    params: Vec<Value>,
    has_exclamation_mark: bool,
    arg_types: Vec<ValueType>,
    input_types: Vec<ValueType>,
  ) -> Self::Aggregator<P>;
}

/// A dynamic aggregate kind
#[derive(Clone)]
pub enum DynamicAggregate {
  Avg(AvgAggregate),
  Count(CountAggregate),
  Exists(ExistsAggregate),
  MinMax(MinMaxAggregate),
  Sampler(DynamicSampleAggregate),
  StringJoin(StringJoinAggregate),
  SumProd(SumProdAggregate),
  WeightedSumAvg(WeightedSumAvgAggregate),
}

macro_rules! match_aggregate {
  ($a: expr, $v:ident, $e:expr) => {
    match $a {
      DynamicAggregate::Avg($v) => $e,
      DynamicAggregate::Count($v) => $e,
      DynamicAggregate::MinMax($v) => $e,
      DynamicAggregate::SumProd($v) => $e,
      DynamicAggregate::StringJoin($v) => $e,
      DynamicAggregate::WeightedSumAvg($v) => $e,
      DynamicAggregate::Exists($v) => $e,
      DynamicAggregate::Sampler($v) => $e,
    }
  };
}

impl DynamicAggregate {
  pub fn name(&self) -> String {
    match_aggregate!(self, a, a.name())
  }

  pub fn aggregate_type(&self) -> AggregateType {
    match_aggregate!(self, a, a.aggregate_type())
  }
}

impl std::fmt::Debug for DynamicAggregate {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match_aggregate!(self, a, f.write_str(&a.name()))
  }
}

/// The registry of aggregates
#[derive(Clone, Debug)]
pub struct AggregateRegistry {
  registry: HashMap<String, DynamicAggregate>,
}

impl AggregateRegistry {
  /// Create an empty registry
  pub fn new() -> Self {
    Self {
      registry: HashMap::new(),
    }
  }

  pub fn std() -> Self {
    let mut registry = Self::new();

    // Register
    registry.register(CountAggregate);
    registry.register(MinMaxAggregate::min());
    registry.register(MinMaxAggregate::max());
    registry.register(MinMaxAggregate::argmin());
    registry.register(MinMaxAggregate::argmax());
    registry.register(SumProdAggregate::sum());
    registry.register(SumProdAggregate::prod());
    registry.register(AvgAggregate::new());
    registry.register(WeightedSumAvgAggregate::weighted_sum());
    registry.register(WeightedSumAvgAggregate::weighted_avg());
    registry.register(ExistsAggregate::new());
    registry.register(StringJoinAggregate::new());
    registry.register(DynamicSampleAggregate::new(TopKSamplerAggregate::top()));
    registry.register(DynamicSampleAggregate::new(TopKSamplerAggregate::unique()));
    registry.register(DynamicSampleAggregate::new(CategoricalAggregate::new()));
    registry.register(DynamicSampleAggregate::new(UniformAggregate::new()));

    // Return
    registry
  }

  /// Register an aggregate into the registry
  pub fn register<A: Aggregate>(&mut self, agg: A) {
    self.registry.entry(agg.name()).or_insert(agg.into());
  }

  pub fn iter(&self) -> impl Iterator<Item = (&String, &DynamicAggregate)> {
    self.registry.iter()
  }

  pub fn instantiate_aggregator<P: Provenance>(
    &self,
    name: &str,
    params: Vec<Value>,
    has_exclamation_mark: bool,
    arg_types: Vec<ValueType>,
    input_types: Vec<ValueType>,
  ) -> Option<DynamicAggregator<P>> {
    if let Some(aggregate) = self.registry.get(name) {
      match_aggregate!(aggregate, a, {
        let instantiated = a.instantiate::<P>(params, has_exclamation_mark, arg_types, input_types);
        Some(DynamicAggregator(Box::new(instantiated)))
      })
    } else {
      None
    }
  }
}

pub trait Aggregator<P: Provenance>: dyn_clone::DynClone + 'static {
  fn aggregate(&self, p: &P, env: &RuntimeEnvironment, elems: DynamicElements<P>) -> DynamicElements<P>;
}

pub struct DynamicAggregator<P: Provenance>(Box<dyn Aggregator<P>>);

impl<P: Provenance> Clone for DynamicAggregator<P> {
  fn clone(&self) -> Self {
    Self(dyn_clone::clone_box(&*self.0))
  }
}

impl<Prov: Provenance> DynamicAggregator<Prov> {
  pub fn aggregate(
    &self,
    prov: &Prov,
    env: &RuntimeEnvironment,
    elems: DynamicElements<Prov>,
  ) -> DynamicElements<Prov> {
    self.0.aggregate(prov, env, elems)
  }
}
