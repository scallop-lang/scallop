use crate::common::tuple::Tuple;
use crate::runtime::provenance::*;

pub trait Monitor<Prov: Provenance> {
  /// Observe stratum iteration
  #[allow(unused_variables)]
  fn observe_executing_stratum(&self, stratum_id: usize) {}

  /// Observe stratum iteration
  #[allow(unused_variables)]
  fn observe_stratum_iteration(&self, iteration_count: usize) {}

  /// Observe hitting iteration limit
  #[allow(unused_variables)]
  fn observe_hitting_iteration_limit(&self) {}

  /// Observe converging
  #[allow(unused_variables)]
  fn observe_converging(&self) {}

  /// Observe loading a relation
  #[allow(unused_variables)]
  fn observe_loading_relation(&self, relation: &str) {}

  /// Observe loading a relation
  #[allow(unused_variables)]
  fn observe_loading_relation_from_edb(&self, relation: &str) {}

  /// Observe loading a relation
  #[allow(unused_variables)]
  fn observe_loading_relation_from_idb(&self, relation: &str) {}

  /// Observe a call on tagging function
  #[allow(unused_variables)]
  fn observe_tagging(&self, tup: &Tuple, input_tag: &Option<Prov::InputTag>, tag: &Prov::Tag) {}

  /// Observe recovering output tags of a relation
  #[allow(unused_variables)]
  fn observe_recovering_relation(&self, relation: &str) {}

  /// Observe a call on recover function
  #[allow(unused_variables)]
  fn observe_recover(&self, tup: &Tuple, tag: &Prov::Tag, output_tag: &Prov::OutputTag) {}
}

impl<Prov: Provenance> Monitor<Prov> for () {}

macro_rules! monitor_observe_event {
  ($func:ident, ($($arg:ident),*), $elem:ident) => {
    $elem.$func( $($arg),* );
  };
  ($func:ident, ($($arg:ident),*), $elem:ident, $($rest:ident),* ) => {
    $elem.$func( $($arg),* );
    monitor_observe_event!( $func, ($($arg),*), $($rest),* );
  };
  ($func:ident, ($($elem:ident),*), ($($arg:ident: $ty:ty),*)) => {
    fn $func(&self, $($arg : $ty,)*) {
      #[allow(non_snake_case)]
      let ($( $elem,)*) = self;
      monitor_observe_event!( $func, ($($arg),*), $($elem),* );
    }
  };
}

macro_rules! impl_monitor {
  ( $($elem:ident),* ) => {
    impl<$($elem),*, Prov> Monitor<Prov> for ($($elem,)*)
    where
      $($elem: Monitor<Prov>,)*
      Prov: Provenance,
    {
      monitor_observe_event!(observe_executing_stratum, ($($elem),*), (stratum_id: usize));
      monitor_observe_event!(observe_stratum_iteration, ($($elem),*), (iteration_count: usize));
      monitor_observe_event!(observe_hitting_iteration_limit, ($($elem),*), ());
      monitor_observe_event!(observe_converging, ($($elem),*), ());
      monitor_observe_event!(observe_loading_relation, ($($elem),*), (relation: &str));
      monitor_observe_event!(observe_loading_relation_from_edb, ($($elem),*), (relation: &str));
      monitor_observe_event!(observe_loading_relation_from_idb, ($($elem),*), (relation: &str));
      monitor_observe_event!(observe_tagging, ($($elem),*), (tup: &Tuple, input_tag: &Option<Prov::InputTag>, tag: &Prov::Tag));
      monitor_observe_event!(observe_recovering_relation, ($($elem),*), (relation: &str));
      monitor_observe_event!(observe_recover, ($($elem),*), (tup: &Tuple, tag: &Prov::Tag, output_tag: &Prov::OutputTag));
    }
  }
}

impl_monitor!(M1);
impl_monitor!(M1, M2);
impl_monitor!(M1, M2, M3);
impl_monitor!(M1, M2, M3, M4);
impl_monitor!(M1, M2, M3, M4, M5);
