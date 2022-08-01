use csv::{ReaderBuilder, WriterBuilder};
use std::fs::File;
use std::path::PathBuf;

use crate::common::input_file::InputFile;
use crate::common::input_tag::InputTag;
use crate::common::output_option::OutputFile;
use crate::common::tuple::Tuple;
use crate::common::tuple_type::TupleType;
use crate::common::value_type::{ValueParseError, ValueType};

pub fn load(input_file: &InputFile, types: &TupleType) -> Result<Vec<(InputTag, Tuple)>, IOError> {
  match input_file {
    InputFile::Csv {
      file_path,
      deliminator,
      has_header,
      has_probability,
    } => load_csv(file_path, *deliminator, *has_header, *has_probability, types),
    InputFile::Txt(_) => unimplemented!(),
  }
}

pub fn load_csv(
  file_path: &PathBuf,
  deliminator: u8,
  has_header: bool,
  has_probability: bool,
  types: &TupleType,
) -> Result<Vec<(InputTag, Tuple)>, IOError> {
  // First parse the value types
  let value_types = get_value_types(types)?;

  // Setup probability offset
  let probability_offset = if has_probability { 1 } else { 0 };

  // Then load the file
  let file = File::open(file_path).map_err(|e| IOError::CannotOpenFile {
    file_path: file_path.clone(),
    error: format!("{}", e),
  })?;

  let mut result = vec![];
  let mut csv_rdr = ReaderBuilder::new()
    .delimiter(deliminator)
    .has_headers(has_header)
    .from_reader(file);

  for row in csv_rdr.records() {
    let record = row.map_err(|e| IOError::CannotParseCSV { error: e.to_string() })?;

    if record.len() - probability_offset != value_types.len() {
      return Err(IOError::ArityMismatch {
        expected: value_types.len(),
        found: record.len(),
      });
    }

    let tag = if has_probability {
      let s = record.get(0).unwrap();
      s.parse::<InputTag>()
        .map_err(|_| IOError::CannotParseProbability { value: s.to_string() })?
    } else {
      InputTag::None
    };

    let values = record
      .into_iter()
      .skip(probability_offset)
      .zip(value_types.iter())
      .map(|(r, t)| t.parse(r).map_err(|e| IOError::ValueParseError { error: e }))
      .collect::<Result<Vec<_>, _>>()?;

    let tuple = Tuple::from_primitives(values);
    result.push((tag, tuple));
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

pub fn store<'a, I>(output_file: &OutputFile, tuples: I) -> Result<(), IOError>
where
  I: Iterator<Item = &'a Tuple>,
{
  match output_file {
    OutputFile::CSV(f) => store_csv(&f.file_path, f.deliminator, tuples),
  }
}

pub fn store_csv<'a, I>(file_path: &PathBuf, deliminator: u8, tuples: I) -> Result<(), IOError>
where
  I: Iterator<Item = &'a Tuple>,
{
  // Then load the file
  let file = File::create(file_path).map_err(|e| IOError::CannotOpenFile {
    file_path: file_path.clone(),
    error: format!("{}", e),
  })?;

  // Write the tuples
  let mut wtr = WriterBuilder::new().delimiter(deliminator).from_writer(file);
  for tuple in tuples {
    let record = tuple.as_ref_values().into_iter().map(|v| format!("{}", v));
    wtr
      .write_record(record)
      .map_err(|e| IOError::CannotWriteRecord { error: e.to_string() })?;
  }

  Ok(())
}

#[derive(Clone, Debug)]
pub enum IOError {
  CannotOpenFile { file_path: PathBuf, error: String },
  CannotReadFile { error: String },
  CannotParseCSV { error: String },
  InvalidType { types: TupleType },
  ValueParseError { error: ValueParseError },
  CannotParseProbability { value: String },
  ArityMismatch { expected: usize, found: usize },
  CannotWriteRecord { error: String },
}

impl std::fmt::Display for IOError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::CannotOpenFile { file_path, error } => f.write_fmt(format_args!(
        "IO: Cannot open file `{}`: {}",
        file_path.as_os_str().to_string_lossy(),
        error
      )),
      Self::CannotReadFile { error } => f.write_fmt(format_args!("IO: Cannot read file: {}", error)),
      Self::CannotParseCSV { error } => f.write_fmt(format_args!("IO: Cannot parse CSV: {}", error)),
      Self::InvalidType { types } => f.write_fmt(format_args!("IO: Invalid tuple type: `{}`", types)),
      Self::ValueParseError { error } => std::fmt::Display::fmt(error, f),
      Self::CannotParseProbability { value } => f.write_fmt(format_args!("IO: Cannot parse probability `{}`", value)),
      Self::ArityMismatch { expected, found } => f.write_fmt(format_args!(
        "IO: Arity mismatch; expected {}, found {}",
        expected, found
      )),
      Self::CannotWriteRecord { error } => f.write_fmt(format_args!("IO: Cannot write record: {}", error)),
    }
  }
}
