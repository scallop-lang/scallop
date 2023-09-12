use crate::common::tuple::*;

use super::*;

pub struct DynamicAggregationImplicitGroupDataflow<'a, Prov: Provenance> {
  pub agg: DynamicAggregator,
  pub d: DynamicDataflow<'a, Prov>,
  pub ctx: &'a Prov,
  pub runtime: &'a RuntimeEnvironment,
}

impl<'a, Prov: Provenance> Clone for DynamicAggregationImplicitGroupDataflow<'a, Prov> {
  fn clone(&self) -> Self {
    Self {
      agg: self.agg.clone(),
      d: self.d.clone(),
      ctx: self.ctx,
      runtime: self.runtime,
    }
  }
}

impl<'a, Prov: Provenance> DynamicAggregationImplicitGroupDataflow<'a, Prov> {
  pub fn new(agg: DynamicAggregator, d: DynamicDataflow<'a, Prov>, ctx: &'a Prov, runtime: &'a RuntimeEnvironment) -> Self {
    Self { agg, d, ctx, runtime }
  }
}

impl<'a, Prov: Provenance> Dataflow<'a, Prov> for DynamicAggregationImplicitGroupDataflow<'a, Prov> {
  fn iter_recent(&self) -> DynamicBatches<'a, Prov> {
    // Sanitize input relation
    let mut batch = if let Some(b) = self.d.iter_recent().next_batch() {
      b
    } else {
      return DynamicBatches::empty();
    };

    // Temporary function to aggregate the group and populate the result
    let consolidate_group =
      |result: &mut DynamicElements<Prov>, agg_key: Tuple, agg_group| {
        let agg_results = self.agg.aggregate(agg_group, self.ctx, self.runtime);
        let joined_results = agg_results
          .into_iter()
          .map(|agg_result| DynamicElement::new((agg_key.clone(), agg_result.tuple.clone()), agg_result.tag));
        result.extend(joined_results);
      };

    // Get the first element from the batch; otherwise, return empty
    let first_elem = if let Some(e) = batch.next_elem() {
      e
    } else {
      return DynamicBatches::empty();
    };

    // Internal states
    let mut result = vec![];
    let mut agg_key = first_elem.tuple[0].clone();
    let mut agg_group = vec![DynamicElement::new(first_elem.tuple[1].clone(), first_elem.tag.clone())];

    // Enter main loop
    while let Some(curr_elem) = batch.next_elem() {
      let curr_key = curr_elem.tuple[0].clone();
      if curr_key == agg_key {
        // If the key is the same, add this element to the same batch
        agg_group.push(DynamicElement::new(curr_elem.tuple[1].clone(), curr_elem.tag.clone()));
      } else {
        // Add the group into the results
        let mut new_agg_key = curr_elem.tuple[0].clone();
        std::mem::swap(&mut new_agg_key, &mut agg_key);
        let mut new_agg_group = vec![DynamicElement::new(curr_elem.tuple[1].clone(), curr_elem.tag.clone())];
        std::mem::swap(&mut new_agg_group, &mut agg_group);
        consolidate_group(&mut result, new_agg_key, new_agg_group);
      }
    }

    // Make sure we handle the last group
    consolidate_group(&mut result, agg_key, agg_group);

    // Return the result as a single batch
    DynamicBatches::single(ElementsBatch::new(result))
  }
}
