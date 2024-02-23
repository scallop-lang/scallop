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

impl Default for BindingTypes {
  fn default() -> Self {
    Self::unit()
  }
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

impl ParamType {
  pub fn is_mandatory(&self) -> bool {
    match self {
      Self::Mandatory(_) => true,
      _ => false,
    }
  }

  pub fn value_type(&self) -> &ValueType {
    match self {
      Self::Mandatory(vt) => vt,
      Self::Optional(vt) => vt,
    }
  }
}

/// The type of an aggregator
///
/// ``` ignore
/// OUTPUT_TYPE := AGGREGATE<PARAM: FAMILY, ...>[ARG_TYPE](INPUT_TYPE)
/// ```
#[derive(Clone, Debug, Default)]
pub struct AggregateType {
  pub generics: HashMap<String, GenericTypeFamily>,
  pub param_types: Vec<ParamType>,
  pub named_param_types: HashMap<String, ParamType>,
  pub arg_type: BindingTypes,
  pub input_type: BindingTypes,
  pub output_type: BindingTypes,
  pub allow_exclamation_mark: bool,
}

impl AggregateType {
  pub fn infer_output_arity(&self, arg_arity: usize, input_arity: usize) -> Result<usize, String> {
    let mut grounded_generic_arity = HashMap::new();
    self.ground_input_aggregate_binding("argument", &self.arg_type, arg_arity, &mut grounded_generic_arity)?;
    self.ground_input_aggregate_binding("input", &self.input_type, input_arity, &mut grounded_generic_arity)?;
    self.solve_output_binding_arity(&self.output_type, &grounded_generic_arity)
  }

  fn ground_input_aggregate_binding(
    &self,
    kind: &str,
    binding_types: &BindingTypes,
    num_variables: usize,
    grounded_generic_arity: &mut HashMap<String, usize>,
  ) -> Result<(), String> {
    match binding_types {
      BindingTypes::IfNotUnit { .. } => Err(format!("cannot have if-not-unit binding type in aggregate input")),
      BindingTypes::TupleType(elems) => {
        if elems.len() == 0 {
          // If elems.len() is 0, it means that there should be no variable for this part of aggregation.
          // We throw error if there is at least 1 variable.
          // Otherwise, the type checking is done as there is no variable that needs to be unified for type
          if num_variables != 0 {
            Err(format!("expected 0 {kind} variables, found {num_variables}"))
          } else {
            Ok(())
          }
        } else if elems.len() == 1 {
          // If elems.len() is 1, we could have that exact element to be a generic type variable or a concrete value type
          match &elems[0] {
            BindingType::Generic(g) => {
              if let Some(grounded_type_arity) = grounded_generic_arity.get(g) {
                if *grounded_type_arity != num_variables {
                  Err(format!("the generic type `{g}` is grounded to have {grounded_type_arity} variables, but found {num_variables}"))
                } else {
                  Ok(())
                }
              } else if let Some(generic_type_family) = self.generics.get(g) {
                let arity = self.solve_generic_type(kind, g, generic_type_family, num_variables)?;
                grounded_generic_arity.insert(g.to_string(), arity);
                Ok(())
              } else {
                Err(format!("unknown generic type parameter `{g}`"))
              }
            }
            BindingType::ValueType(_) => {
              if num_variables == 1 {
                Ok(())
              } else {
                // Arity mismatch
                Err(format!("expected one {kind} variable; found {num_variables}"))
              }
            }
          }
        } else {
          if elems.iter().any(|e| e.is_generic()) {
            Err(format!(
              "cannot have generic in the {kind} of aggregate of more than 1 elements"
            ))
          } else if elems.len() != num_variables {
            Err(format!(
              "expected {} {kind} variables, found {num_variables}",
              elems.len()
            ))
          } else {
            Ok(())
          }
        }
      }
    }
  }

  fn solve_generic_type(
    &self,
    kind: &str,
    generic_type_name: &str,
    generic_type_family: &GenericTypeFamily,
    num_variables: usize,
  ) -> Result<usize, String> {
    match generic_type_family {
      GenericTypeFamily::NonEmptyTuple => {
        if num_variables == 0 {
          Err(format!(
            "arity mismatch. Expected non-empty {kind} variables, but found 0"
          ))
        } else {
          Ok(num_variables)
        }
      }
      GenericTypeFamily::NonEmptyTupleWithElements(elem_type_families) => {
        if elem_type_families.iter().any(|tf| !tf.is_type_family()) {
          Err(format!(
            "generic type family `{generic_type_name}` contains unsupported nested tuple"
          ))
        } else if num_variables != elem_type_families.len() {
          Err(format!(
            "arity mismatch. Expected {} {kind} variables, but found 0",
            elem_type_families.len()
          ))
        } else {
          Ok(num_variables)
        }
      }
      GenericTypeFamily::UnitOr(child_generic_type_family) => {
        if num_variables == 0 {
          Ok(0)
        } else {
          self.solve_generic_type(kind, generic_type_name, &*child_generic_type_family, num_variables)
        }
      }
      GenericTypeFamily::TypeFamily(_) => {
        if num_variables != 1 {
          Err(format!("arity mismatch. Expected 1 {kind} variables, but found 0"))
        } else {
          Ok(1)
        }
      }
    }
  }

  fn solve_output_binding_arity(
    &self,
    binding_types: &BindingTypes,
    grounded_generic_arity: &HashMap<String, usize>,
  ) -> Result<usize, String> {
    match binding_types {
      BindingTypes::IfNotUnit {
        generic_type,
        then_type,
        else_type,
      } => {
        if let Some(arity) = grounded_generic_arity.get(generic_type) {
          if *arity > 0 {
            self.solve_output_binding_arity(then_type, grounded_generic_arity)
          } else {
            self.solve_output_binding_arity(else_type, grounded_generic_arity)
          }
        } else {
          Err(format!(
            "error grounding output type: unknown generic type `{generic_type}`"
          ))
        }
      }
      BindingTypes::TupleType(elems) => Ok(
        elems
          .iter()
          .map(|elem| match elem {
            BindingType::Generic(g) => {
              if let Some(arity) = grounded_generic_arity.get(g) {
                Ok(*arity)
              } else {
                Err(format!("error grounding output type: unknown generic type `{g}`"))
              }
            }
            BindingType::ValueType(_) => Ok(1),
          })
          .collect::<Result<Vec<_>, _>>()?
          .into_iter()
          .fold(0, |acc, a| acc + a),
      ),
    }
  }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct AggregateInfo {
  pub pos_params: Vec<Value>,
  pub named_params: BTreeMap<String, Value>,
  pub has_exclamation_mark: bool,
  pub arg_var_types: Vec<ValueType>,
  pub input_var_types: Vec<ValueType>,
}

impl AggregateInfo {
  pub fn with_arg_var_types(mut self, arg_var_types: Vec<ValueType>) -> Self {
    self.arg_var_types = arg_var_types;
    self
  }

  pub fn with_input_var_types(mut self, input_var_types: Vec<ValueType>) -> Self {
    self.input_var_types = input_var_types;
    self
  }
}

pub trait Aggregate: Into<DynamicAggregate> {
  /// The concrete aggregator that this aggregate is instantiated into
  type Aggregator<P: Provenance>: Aggregator<P>;

  /// The name of the aggregate
  fn name(&self) -> String;

  /// The type of the aggregate
  fn aggregate_type(&self) -> AggregateType;

  /// Instantiate the aggregate into an aggregator with the given parameters
  fn instantiate<P: Provenance>(&self, aggregate_info: AggregateInfo) -> Self::Aggregator<P>;
}

/// A dynamic aggregate kind
#[derive(Clone)]
pub enum DynamicAggregate {
  Avg(AvgAggregate),
  Count(CountAggregate),
  Disjunct(DisjunctAggregate),
  Enumerate(EnumerateAggregate),
  Exists(ExistsAggregate),
  MinMax(MinMaxAggregate),
  Normalize(NormalizeAggregate),
  Rank(RankAggregate),
  Sampler(DynamicSampleAggregate),
  Sort(SortAggregate),
  StringJoin(StringJoinAggregate),
  SumProd(SumProdAggregate),
  WeightedSumAvg(WeightedSumAvgAggregate),
}

macro_rules! match_aggregate {
  ($a: expr, $v:ident, $e:expr) => {
    match $a {
      DynamicAggregate::Avg($v) => $e,
      DynamicAggregate::Count($v) => $e,
      DynamicAggregate::Disjunct($v) => $e,
      DynamicAggregate::Enumerate($v) => $e,
      DynamicAggregate::MinMax($v) => $e,
      DynamicAggregate::Normalize($v) => $e,
      DynamicAggregate::Rank($v) => $e,
      DynamicAggregate::Sampler($v) => $e,
      DynamicAggregate::Sort($v) => $e,
      DynamicAggregate::StringJoin($v) => $e,
      DynamicAggregate::SumProd($v) => $e,
      DynamicAggregate::WeightedSumAvg($v) => $e,
      DynamicAggregate::Exists($v) => $e,
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
    registry.register(DisjunctAggregate);
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
    registry.register(NormalizeAggregate::normalize());
    registry.register(NormalizeAggregate::softmax());
    registry.register(StringJoinAggregate::new());
    registry.register(EnumerateAggregate::new());
    registry.register(RankAggregate::new());
    registry.register(SortAggregate::sort());
    registry.register(SortAggregate::argsort());
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

  pub fn instantiate_aggregator<P: Provenance>(&self, name: &str, info: AggregateInfo) -> Option<DynamicAggregator<P>> {
    if let Some(aggregate) = self.registry.get(name) {
      match_aggregate!(aggregate, a, {
        let instantiated = a.instantiate::<P>(info);
        Some(DynamicAggregator(Box::new(instantiated)))
      })
    } else {
      None
    }
  }

  pub fn create_type_registry(&self) -> HashMap<String, AggregateType> {
    self
      .registry
      .iter()
      .map(|(name, agg)| (name.clone(), agg.aggregate_type()))
      .collect()
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
