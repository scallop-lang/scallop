use crate::common::input_tag::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::provenance::*;
use crate::utils::*;

use super::*;

#[derive(Clone)]
pub struct NormalizeAggregate {
  pub soft: bool,
}

impl NormalizeAggregate {
  pub fn normalize() -> Self {
    Self { soft: false }
  }

  pub fn softmax() -> Self {
    Self { soft: true }
  }
}

impl Into<DynamicAggregate> for NormalizeAggregate {
  fn into(self) -> DynamicAggregate {
    DynamicAggregate::Normalize(self)
  }
}

impl Aggregate for NormalizeAggregate {
  type Aggregator<P: Provenance> = NormalizeAggregator;

  fn name(&self) -> String {
    if self.soft {
      "softmax".to_string()
    } else {
      "normalize".to_string()
    }
  }

  fn aggregate_type(&self) -> AggregateType {
    AggregateType {
      generics: std::iter::once(("T".to_string(), GenericTypeFamily::non_empty_tuple())).collect(),
      input_type: BindingTypes::generic("T"),
      output_type: BindingTypes::generic("T"),
      ..Default::default()
    }
  }

  fn instantiate<P: Provenance>(&self, _: AggregateInfo) -> Self::Aggregator<P> {
    NormalizeAggregator { soft: self.soft }
  }
}

#[derive(Clone)]
pub struct NormalizeAggregator {
  soft: bool,
}

impl<Prov: Provenance> Aggregator<Prov> for NormalizeAggregator {
  default fn aggregate(
    &self,
    _prov: &Prov,
    _env: &RuntimeEnvironment,
    batch: DynamicElements<Prov>,
  ) -> DynamicElements<Prov> {
    batch
  }
}

macro_rules! impl_softmax_for_prob_prov {
  ($module:ident :: $prov:ident) => {
    impl_softmax_for_prob_prov!($module::$prov < >);
  };
  ($module:ident :: $prov:ident < $( $N:ident $(: $b0:ident $(+$b:ident)* )? ),* >) => {
    impl<$( $N $(: $b0 $(+$b)* )? ),*> Aggregator<$module::$prov<$( $N ),*>> for NormalizeAggregator {
      fn aggregate(
        &self,
        p: &$module::$prov< $( $N ),* >,
        env: &RuntimeEnvironment,
        elems: DynamicElements<$module::$prov<$( $N ),*>>,
      ) -> DynamicElements<$module::$prov<$( $N ),*>> {
        let weights = elems.iter().map(|e| p.weight(&e.tag)).collect::<Vec<_>>();
        let normalized_weights = if self.soft {
          softmax(&weights)
        } else {
          normalize(&weights)
        };
        let exclusion_id = env.exclusion_id_allocator.alloc();
        elems
          .iter()
          .zip(normalized_weights)
          .filter_map(|(e, np)| {
            let dyn_tag = DynamicInputTag::ExclusiveFloat(np, exclusion_id);
            if let Some(input_tag) = <$module::$prov< $( $N ),* > as Provenance>::InputTag::from_dynamic_input_tag(&dyn_tag) {
              let tag = <$module::$prov< $( $N ),* > as Provenance>::tagging_fn(p, input_tag);
              Some(DynamicElement::new(e.tuple.clone(), tag))
            } else {
              None
            }
          })
          .collect()
      }
    }
  };
}

impl_softmax_for_prob_prov!(min_max_prob::MinMaxProbProvenance);
impl_softmax_for_prob_prov!(add_mult_prob::AddMultProbProvenance);
impl_softmax_for_prob_prov!(top_k_proofs::TopKProofsProvenance<P: PointerFamily>);
impl_softmax_for_prob_prov!(top_bottom_k_clauses::TopBottomKClausesProvenance<P: PointerFamily>);

fn softmax(vec: &[f64]) -> Vec<f64> {
  let sum_of_exp = vec.iter().fold(0.0, |acc, v| acc + v.exp());
  vec.iter().map(|v| v.exp() / sum_of_exp).collect()
}

fn normalize(vec: &[f64]) -> Vec<f64> {
  let sum_of_prob = vec.iter().fold(0.0, |acc, v| acc + v);
  vec.iter().map(|v| v / sum_of_prob).collect()
}
