use crate::common::input_tag::*;
use crate::common::tuple::*;
use crate::runtime::dynamic::*;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;

#[derive(Clone, Debug)]
pub struct ExtensionalRelation<Prov: Provenance> {
  /// The facts from the program
  program_facts: Vec<(DynamicInputTag, Tuple)>,

  /// Whether we have internalized the program facts; we only allow a single
  /// round of internalization of program facts
  pub internalized_program_facts: bool,

  /// Dynamically tagged input facts
  dynamic_input: Vec<(DynamicInputTag, Tuple)>,

  /// Statically tagged input facts
  static_input: Vec<(Option<Prov::InputTag>, Tuple)>,

  /// Internalized facts
  pub internal: DynamicCollection<Prov>,

  /// Internalized flag
  pub internalized: bool,
}

impl<Prov: Provenance> Default for ExtensionalRelation<Prov> {
  fn default() -> Self {
    Self::new()
  }
}

impl<Prov: Provenance> ExtensionalRelation<Prov> {
  pub fn new() -> Self {
    Self {
      program_facts: vec![],
      internalized_program_facts: false,
      dynamic_input: vec![],
      static_input: vec![],
      internal: DynamicCollection::empty(),
      internalized: false,
    }
  }

  pub fn has_program_facts(&self) -> bool {
    !self.program_facts.is_empty()
  }

  pub fn num_program_facts(&self) -> usize {
    self.program_facts.len()
  }

  pub fn add_program_facts<I>(&mut self, i: I)
  where
    I: Iterator<Item = (DynamicInputTag, Tuple)>,
  {
    self.program_facts.extend(i)
  }

  pub fn add_facts(&mut self, facts: Vec<Tuple>) {
    if !facts.is_empty() {
      self.internalized = false;
    }

    self.static_input.extend(facts.into_iter().map(|tup| (None, tup)))
  }

  pub fn add_dynamic_input_facts(&mut self, facts: Vec<(DynamicInputTag, Tuple)>) {
    if !facts.is_empty() {
      self.internalized = false;
    }

    self.dynamic_input.extend(facts)
  }

  pub fn add_static_input_facts(&mut self, facts: Vec<(Option<Prov::InputTag>, Tuple)>) {
    if !facts.is_empty() {
      self.internalized = false;
    }

    self.static_input.extend(facts)
  }

  pub fn internalize(&mut self, ctx: &mut Prov) {
    let mut elems: Vec<DynamicElement<Prov>> = Vec::new();

    // First internalize program facts, only if there is program facts
    if !self.program_facts.is_empty() {
      // Iterate (not drain) the program facts
      elems.extend(self.program_facts.iter().map(|(tag, tup)| {
        let maybe_input_tag = StaticInputTag::from_dynamic_input_tag(&tag);
        let tag = ctx.tagging_optional_fn(maybe_input_tag);
        DynamicElement::new(tup.clone(), tag)
      }));

      // Set the internalization to `true`
      self.internalized_program_facts = true;
    }

    // First internalize dynamic input facts
    elems.extend(self.dynamic_input.drain(..).map(|(tag, tup)| {
      let maybe_input_tag = StaticInputTag::from_dynamic_input_tag(&tag);
      let tag = ctx.tagging_optional_fn(maybe_input_tag);
      DynamicElement::new(tup, tag)
    }));

    // Then internalize static input facts
    elems.extend(self.static_input.drain(..).map(|(tag, tup)| {
      let tag = ctx.tagging_optional_fn(tag);
      DynamicElement::new(tup, tag)
    }));

    // Add existed facts
    elems.extend(self.internal.elements.drain(..));

    // Finally sort the internal facts; note that we need to merge possibly duplicated tags
    self.internal = DynamicCollection::from_vec(elems, ctx);
    self.internalized = true;
  }

  pub fn internalize_with_monitor<M: Monitor<Prov>>(&mut self, ctx: &mut Prov, m: &M) {
    let mut elems: Vec<DynamicElement<Prov>> = Vec::new();

    // First internalize program facts, only if there is program facts
    if !self.program_facts.is_empty() {
      // Iterate (not drain) the program facts
      elems.extend(self.program_facts.iter().map(|(tag, tup)| {
        let maybe_input_tag = StaticInputTag::from_dynamic_input_tag(&tag);
        let tag = ctx.tagging_optional_fn(maybe_input_tag.clone());

        // !SPECIAL MONITORING!
        m.observe_tagging(tup, &maybe_input_tag, &tag);

        DynamicElement::new(tup.clone(), tag)
      }));

      // Set the internalization to `true`
      self.internalized_program_facts = true;
    }

    // First internalize dynamic input facts
    elems.extend(self.dynamic_input.drain(..).map(|(tag, tup)| {
      let maybe_input_tag = StaticInputTag::from_dynamic_input_tag(&tag);
      let tag = ctx.tagging_optional_fn(maybe_input_tag.clone());

      // !SPECIAL MONITORING!
      m.observe_tagging(&tup, &maybe_input_tag, &tag);

      DynamicElement::new(tup, tag)
    }));

    // Then internalize static input facts
    elems.extend(self.static_input.drain(..).map(|(input_tag, tup)| {
      let tag = ctx.tagging_optional_fn(input_tag.clone());

      // !SPECIAL MONITORING!
      m.observe_tagging(&tup, &input_tag, &tag);

      DynamicElement::new(tup, tag)
    }));

    // Add existed facts
    elems.extend(self.internal.elements.drain(..));

    // Finally sort the internal facts; note that we need to merge possibly duplicated tags
    self.internal = DynamicCollection::from_vec(elems, ctx);
    self.internalized = true;
  }
}
