use crate::common::tuple::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone)]
pub struct RankAggregate;

impl RankAggregate {
  pub fn new() -> Self {
    Self
  }
}

impl Into<DynamicAggregate> for RankAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Rank(self)
  }
}

impl Aggregate for RankAggregate {
  type Aggregator<P: Provenance> = RankAggregator;

  fn name(&self) -> String {
    "rank".to_string()
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: vec![("T".to_string(), GenericTypeFamily::non_empty_tuple())]
        .into_iter()
        .collect(),
      named_param_types: vec![
        ("asc".to_string(), ParamType::Optional(ValueType::Bool)),
        ("desc".to_string(), ParamType::Optional(ValueType::Bool)),
      ]
      .into_iter()
      .collect(),
      input_type: BindingTypes::generic("T"),
      output_type: BindingTypes::tuple(vec![
        BindingType::value_type(ValueType::USize),
        BindingType::generic("T"),
      ]),
      ..Default::default()
    }
  }

  fn instantiate<P: Provenance>(&self, info: AggregateInfo) -> Self::Aggregator<P> {
    let asc = info.named_params.get("asc").map(|v| v.as_bool());
    let not_desc = info.named_params.get("desc").map(|v| !v.as_bool());
    RankAggregator {
      ascending: asc.unwrap_or(not_desc.unwrap_or(false)),
    }
  }
}

#[derive(Clone)]
pub struct RankAggregator {
  pub ascending: bool,
}

impl<Prov: Provenance> Aggregator<Prov> for RankAggregator {
  default fn aggregate(
    &self,
    prov: &Prov,
    _env: &RuntimeEnvironment,
    mut batch: DynamicElements<Prov>,
  ) -> DynamicElements<Prov> {
    // Helper struct to sort f64
    #[derive(PartialEq, PartialOrd)]
    struct OrdF64(f64);
    impl std::cmp::Eq for OrdF64 {}
    impl std::cmp::Ord for OrdF64 {
      fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
      }
    }

    // Sort the batch
    if self.ascending {
      batch.sort_by_cached_key(|elem| OrdF64(prov.weight(&elem.tag)));
    } else {
      batch.sort_by_cached_key(|elem| OrdF64(-prov.weight(&elem.tag)));
    }

    // Return
    batch
      .into_iter()
      .enumerate()
      .map(|(i, elem)| {
        let tup = Tuple::from_values(std::iter::once(i.into()).chain(elem.tuple.to_values().into_iter()));
        DynamicElement::new(tup, elem.tag)
      })
      .collect()
  }
}
