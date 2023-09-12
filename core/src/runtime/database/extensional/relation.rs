use std::collections::*;
use std::path::*;

use crate::common::input_file::*;
use crate::common::input_tag::*;
use crate::common::tuple::*;
use crate::common::tuple_type::*;
use crate::common::value::*;
use crate::common::value_type::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::error::*;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;

use super::*;

#[derive(Clone, Debug)]
pub struct ExtensionalRelation<Prov: Provenance> {
  /// The facts from the program
  program_facts: Vec<(DynamicInputTag, Tuple)>,

  /// Whether we have internalized the program facts; we only allow a single
  /// round of internalization of program facts
  pub internalized_program_facts: bool,

  /// Loaded files
  pub loaded_files: BTreeSet<PathBuf>,

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
      loaded_files: BTreeSet::new(),
      dynamic_input: vec![],
      static_input: vec![],
      internal: DynamicCollection::empty(),
      internalized: false,
    }
  }

  pub fn clone_with_new_provenance<Prov2: Provenance>(&self) -> ExtensionalRelation<Prov2>
  where
    Prov2::InputTag: ConvertFromInputTag<Prov::InputTag>,
  {
    ExtensionalRelation {
      program_facts: self.program_facts.clone(),
      internalized_program_facts: false,
      loaded_files: BTreeSet::new(),
      dynamic_input: self.dynamic_input.clone(),
      static_input: self
        .static_input
        .iter()
        .map(|(tag, tuple)| {
          let new_tag = tag
            .as_ref()
            .and_then(|tag| ConvertFromInputTag::from_input_tag(tag.clone()));
          (new_tag, tuple.clone())
        })
        .collect(),
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

  pub fn load_from_file(
    &mut self,
    env: &RuntimeEnvironment,
    file: &InputFile,
    loaded_file_content: &io::InputFileContent,
    types: &TupleType,
  ) -> Result<(), DatabaseError> {
    // Do not load again
    if self.loaded_files.contains(file.file_path()) {
      return Ok(());
    }

    // Load the file into the dynamic input
    let result = load_from_file(env, file, loaded_file_content, types).map_err(DatabaseError::IO)?;
    self.dynamic_input.extend(result);

    // If succeeded, add the file into the file_path
    self.loaded_files.insert(file.file_path().clone());

    // Return ok
    Ok(())
  }

  pub fn internalize(&mut self, env: &RuntimeEnvironment, ctx: &Prov) {
    let mut elems: Vec<DynamicElement<Prov>> = Vec::new();

    // First internalize program facts, only if there is program facts
    if !self.program_facts.is_empty() {
      // Iterate (not drain) the program facts
      elems.extend(self.program_facts.iter().filter_map(|(tag, tup)| {
        let int_tup = env.internalize_tuple(tup)?;
        let maybe_input_tag = StaticInputTag::from_dynamic_input_tag(&tag);
        let tag = ctx.tagging_optional_fn(maybe_input_tag);
        Some(DynamicElement::new(int_tup, tag))
      }));

      // Set the internalization to `true`
      self.internalized_program_facts = true;
    }

    // First internalize dynamic input facts
    elems.extend(self.dynamic_input.drain(..).filter_map(|(tag, tup)| {
      let int_tup = env.internalize_tuple(&tup)?;
      let maybe_input_tag = StaticInputTag::from_dynamic_input_tag(&tag);
      let tag = ctx.tagging_optional_fn(maybe_input_tag);
      Some(DynamicElement::new(int_tup, tag))
    }));

    // Then internalize static input facts
    elems.extend(self.static_input.drain(..).filter_map(|(tag, tup)| {
      let int_tup = env.internalize_tuple(&tup)?;
      let tag = ctx.tagging_optional_fn(tag);
      Some(DynamicElement::new(int_tup, tag))
    }));

    // Add existed facts
    elems.extend(self.internal.elements.drain(..));

    // Finally sort the internal facts; note that we need to merge possibly duplicated tags
    self.internal = DynamicCollection::from_vec(elems, ctx);
    self.internalized = true;
  }

  pub fn internalize_with_monitor<M: Monitor<Prov>>(&mut self, env: &RuntimeEnvironment, ctx: &Prov, m: &M) {
    let mut elems: Vec<DynamicElement<Prov>> = Vec::new();

    // First internalize program facts, only if there is program facts
    if !self.program_facts.is_empty() {
      // Iterate (not drain) the program facts
      elems.extend(self.program_facts.iter().filter_map(|(tag, tup)| {
        let int_tup = env.internalize_tuple(tup)?;
        let maybe_input_tag = StaticInputTag::from_dynamic_input_tag(&tag);
        let tag = ctx.tagging_optional_fn(maybe_input_tag.clone());

        // !SPECIAL MONITORING!
        m.observe_tagging(&int_tup, &maybe_input_tag, &tag);

        Some(DynamicElement::new(int_tup, tag))
      }));

      // Set the internalization to `true`
      self.internalized_program_facts = true;
    }

    // First internalize dynamic input facts
    elems.extend(self.dynamic_input.drain(..).filter_map(|(tag, tup)| {
      let int_tup = env.internalize_tuple(&tup)?;
      let maybe_input_tag = StaticInputTag::from_dynamic_input_tag(&tag);
      let tag = ctx.tagging_optional_fn(maybe_input_tag.clone());

      // !SPECIAL MONITORING!
      m.observe_tagging(&tup, &maybe_input_tag, &tag);

      Some(DynamicElement::new(int_tup, tag))
    }));

    // Then internalize static input facts
    elems.extend(self.static_input.drain(..).filter_map(|(input_tag, tup)| {
      let int_tup = env.internalize_tuple(&tup)?;
      let tag = ctx.tagging_optional_fn(input_tag.clone());

      // !SPECIAL MONITORING!
      m.observe_tagging(&tup, &input_tag, &tag);

      Some(DynamicElement::new(int_tup, tag))
    }));

    // Add existed facts
    elems.extend(self.internal.elements.drain(..));

    // Finally sort the internal facts; note that we need to merge possibly duplicated tags
    self.internal = DynamicCollection::from_vec(elems, ctx);
    self.internalized = true;
  }
}

fn load_from_file(
  env: &RuntimeEnvironment,
  input_file: &InputFile,
  loaded_file_content: &io::InputFileContent,
  types: &TupleType,
) -> Result<Vec<(DynamicInputTag, Tuple)>, IOError> {
  match (input_file, loaded_file_content) {
    (
      InputFile::Csv {
        keys,
        fields,
        has_probability,
        ..
      },
      io::InputFileContent::CSV(content),
    ) => load_from_csv(env, keys, fields, *has_probability, content, types),
  }
}

fn load_from_csv(
  env: &RuntimeEnvironment,
  keys: &Option<Vec<String>>,
  fields: &Option<Vec<String>>,
  has_probability: bool,
  loaded_file_content: &io::CSVFileContent,
  types: &TupleType,
) -> Result<Vec<(DynamicInputTag, Tuple)>, IOError> {
  match (keys, fields) {
    (Some(keys), Some(fields)) => {
      load_from_csv_with_keys_and_fields(env, keys, fields, has_probability, loaded_file_content, types)
    }
    (Some(keys), None) => load_from_csv_with_keys(env, keys, has_probability, loaded_file_content, types),
    (None, Some(fields)) => load_from_csv_with_fields(fields, has_probability, loaded_file_content, types),
    (None, None) => load_from_csv_raw(has_probability, loaded_file_content, types),
  }
}

fn load_from_csv_with_keys_and_fields(
  env: &RuntimeEnvironment,
  keys: &Vec<String>,
  fields: &Vec<String>,
  has_probability: bool,
  loaded_file_content: &io::CSVFileContent,
  types: &TupleType,
) -> Result<Vec<(DynamicInputTag, Tuple)>, IOError> {
  // Get the value types and the probability offset
  let value_types = get_value_types(types)?;
  let probability_offset = if has_probability { 1 } else { 0 };

  // Check arity first
  if value_types.len() != keys.len() + 2 {
    return Err(IOError::ArityMismatch {
      expected: keys.len() + 2,
      found: value_types.len(),
    });
  }

  // Check types
  match (value_types[value_types.len() - 2], value_types[value_types.len() - 1]) {
    (ValueType::Symbol, ValueType::String) => { /* GOOD */ }
    (ValueType::Symbol, value_type) => {
      return Err(IOError::ExpectStringType {
        actual: value_type.clone(),
      })
    }
    (field_type, _) => {
      return Err(IOError::ExpectSymbolType {
        actual: field_type.clone(),
      })
    }
  }

  // Get the indices of the keys
  let key_value_ids = keys
    .iter()
    .zip(value_types.iter())
    .map(|(k, t)| {
      let id = loaded_file_content
        .get_header_id(k)
        .ok_or(IOError::CannotFindField { field: k.clone() })?;
      Ok((id, *t))
    })
    .collect::<Result<Vec<_>, _>>()?;

  // Get the set of fields to include
  let fields_id_set = fields
    .iter()
    .map(|f| {
      loaded_file_content
        .get_header_id(f)
        .ok_or(IOError::CannotFindField { field: f.clone() })
    })
    .collect::<Result<HashSet<_>, _>>()?;

  // Field values
  let field_symbols: Vec<Value> = loaded_file_content
    .iter_headers()
    .map(|h| Value::Symbol(env.symbol_registry.register(h.clone())))
    .collect();

  // Cache the results and process each row
  let mut result = vec![];
  for row in loaded_file_content.get_rows() {
    // TODO: this could generate multiple facts with the duplicated same tag
    // Get the tag
    let tag = if has_probability {
      let s = row.get(0).ok_or(IOError::IndexOutOfBounds { index: 0 })?;
      s.parse::<DynamicInputTag>()
        .map_err(|_| IOError::CannotParseProbability { value: s.to_string() })?
    } else {
      DynamicInputTag::None
    };

    // Get the keys
    let keys = key_value_ids
      .iter()
      .map(|(i, t)| {
        let s = row.get(*i).ok_or(IOError::IndexOutOfBounds { index: *i })?;
        t.parse(s).map_err(|e| IOError::ValueParseError { error: e })
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Get all other values
    let field_values = row
      .iter()
      .enumerate()
      .skip(probability_offset) // we skip the tag in the front
      .filter(|(i, _)| {
        // We want to skip the fields that are keys
        key_value_ids.iter().find(|(j, _)| i == j).is_none() && fields_id_set.contains(i)
      })
      .map(|(i, s)| {
        let field = field_symbols
          .get(i)
          .ok_or(IOError::IndexOutOfBounds { index: i })?
          .clone();
        let value = Value::String(s.to_string());
        Ok((field, value))
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Create the results
    let curr_results = field_values
      .into_iter()
      .map(|(field, value)| {
        let tuple = Tuple::from_values(keys.iter().cloned().chain(vec![field, value]));
        let element = (tag.clone(), tuple);
        element
      })
      .collect::<Vec<_>>();

    // Extend the results
    result.extend(curr_results);
  }

  // Return
  Ok(result)
}

fn load_from_csv_with_keys(
  env: &RuntimeEnvironment,
  keys: &Vec<String>,
  has_probability: bool,
  loaded_file_content: &io::CSVFileContent,
  types: &TupleType,
) -> Result<Vec<(DynamicInputTag, Tuple)>, IOError> {
  // Get the value types and the probability offset
  let value_types = get_value_types(types)?;
  let probability_offset = if has_probability { 1 } else { 0 };

  // Check arity first
  if value_types.len() != keys.len() + 2 {
    return Err(IOError::ArityMismatch {
      expected: keys.len() + 2,
      found: value_types.len(),
    });
  }

  // Check types
  match (value_types[value_types.len() - 2], value_types[value_types.len() - 1]) {
    (ValueType::Symbol, ValueType::String) => { /* GOOD */ }
    (ValueType::Symbol, value_type) => {
      return Err(IOError::ExpectStringType {
        actual: value_type.clone(),
      })
    }
    (field_type, _) => {
      return Err(IOError::ExpectSymbolType {
        actual: field_type.clone(),
      })
    }
  }

  // Get the indices of the keys
  let key_value_ids = keys
    .iter()
    .zip(value_types.iter())
    .map(|(k, t)| {
      let id = loaded_file_content
        .get_header_id(k)
        .ok_or(IOError::CannotFindField { field: k.clone() })?;
      Ok((id, *t))
    })
    .collect::<Result<Vec<_>, _>>()?;

  // Field values
  let field_symbols: Vec<Value> = loaded_file_content
    .iter_headers()
    .map(|h| Value::Symbol(env.symbol_registry.register(h.clone())))
    .collect();

  // Cache the results and process each row
  let mut result = vec![];
  for row in loaded_file_content.get_rows() {
    // TODO: this could generate multiple facts with the duplicated same tag
    // Get the tag
    let tag = if has_probability {
      let s = row.get(0).ok_or(IOError::IndexOutOfBounds { index: 0 })?;
      s.parse::<DynamicInputTag>()
        .map_err(|_| IOError::CannotParseProbability { value: s.to_string() })?
    } else {
      DynamicInputTag::None
    };

    // Get the keys
    let keys = key_value_ids
      .iter()
      .map(|(i, t)| {
        let s = row.get(*i).ok_or(IOError::IndexOutOfBounds { index: *i })?;
        t.parse(s).map_err(|e| IOError::ValueParseError { error: e })
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Get all other values
    let field_values = row
      .iter()
      .enumerate()
      .skip(probability_offset) // we skip the tag in the front
      .filter(|(i, _)| {
        // We want to skip the fields that are keys
        key_value_ids.iter().find(|(j, _)| i == j).is_none()
      })
      .map(|(i, s)| {
        let field = field_symbols
          .get(i)
          .ok_or(IOError::IndexOutOfBounds { index: i })?
          .clone();
        let value = Value::String(s.to_string());
        Ok((field, value))
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Create the results
    let curr_results = field_values
      .into_iter()
      .map(|(field, value)| {
        let tuple = Tuple::from_values(keys.iter().cloned().chain(vec![field, value]));
        let element = (tag.clone(), tuple);
        element
      })
      .collect::<Vec<_>>();

    // Extend the results
    result.extend(curr_results);
  }

  // Return
  Ok(result)
}

fn load_from_csv_with_fields(
  fields: &Vec<String>,
  has_probability: bool,
  loaded_file_content: &io::CSVFileContent,
  types: &TupleType,
) -> Result<Vec<(DynamicInputTag, Tuple)>, IOError> {
  // Get the value types and the probability offset
  let value_types = get_value_types(types)?;

  // Check arity first
  if value_types.len() != fields.len() {
    return Err(IOError::ArityMismatch {
      expected: fields.len(),
      found: value_types.len(),
    });
  }

  // Get the indices of the keys
  let id_type_pairs = fields
    .iter()
    .zip(value_types.iter())
    .map(|(f, t)| {
      let id = loaded_file_content
        .get_header_id(f)
        .ok_or(IOError::CannotFindField { field: f.clone() })?;
      Ok((id, *t))
    })
    .collect::<Result<Vec<_>, _>>()?;

  // Cache the results and process each row
  let mut result = vec![];
  for row in loaded_file_content.get_rows() {
    // Get the tag
    let tag = if has_probability {
      let s = row.get(0).ok_or(IOError::IndexOutOfBounds { index: 0 })?;
      s.parse::<DynamicInputTag>()
        .map_err(|_| IOError::CannotParseProbability { value: s.to_string() })?
    } else {
      DynamicInputTag::None
    };

    // Get the tuple
    let values = id_type_pairs
      .iter()
      .map(|(id, t)| {
        let value = row.get(*id).ok_or(IOError::IndexOutOfBounds { index: *id })?;
        t.parse(value).map_err(|e| IOError::ValueParseError { error: e })
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Create the tagged-tuple
    let tagged_tuple = (tag, Tuple::from(values));
    result.push(tagged_tuple);
  }

  Ok(result)
}

fn load_from_csv_raw(
  has_probability: bool,
  loaded_file_content: &io::CSVFileContent,
  types: &TupleType,
) -> Result<Vec<(DynamicInputTag, Tuple)>, IOError> {
  // Get the value types and the probability offset
  let value_types = get_value_types(types)?;
  let probability_offset = if has_probability { 1 } else { 0 };

  // Cache the results and process each row
  let mut result = vec![];
  for row in loaded_file_content.get_rows() {
    // Check arity
    if row.len() != value_types.len() + probability_offset {
      return Err(IOError::ArityMismatch {
        expected: row.len(),
        found: value_types.len() + probability_offset,
      });
    }

    // Get the tag
    let tag = if has_probability {
      let s = row.get(0).ok_or(IOError::IndexOutOfBounds { index: 0 })?;
      s.parse::<DynamicInputTag>()
        .map_err(|_| IOError::CannotParseProbability { value: s.to_string() })?
    } else {
      DynamicInputTag::None
    };

    // Get the tuple
    let values = row
      .iter()
      .skip(probability_offset)
      .zip(value_types.iter())
      .map(|(r, t)| t.parse(r).map_err(|e| IOError::ValueParseError { error: e }))
      .collect::<Result<Vec<_>, _>>()?;

    // Create the tagged-tuple
    let tagged_tuple = (tag, Tuple::from(values));
    result.push(tagged_tuple);
  }

  Ok(result)
}

fn get_value_types(types: &TupleType) -> Result<Vec<&ValueType>, IOError> {
  match types {
    TupleType::Tuple(ts) => ts
      .iter()
      .map(|t| match t {
        TupleType::Value(v) => Some(v),
        _ => None,
      })
      .collect::<Option<Vec<_>>>()
      .ok_or(IOError::InvalidType { types: types.clone() }),
    TupleType::Value(_) => Err(IOError::InvalidType { types: types.clone() }),
  }
}
