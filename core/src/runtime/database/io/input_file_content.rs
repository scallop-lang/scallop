use std::collections::*;
use std::path::*;

use crate::common::input_file::InputFile;
use crate::runtime::error::IOError;

#[derive(Debug, Clone)]
pub enum InputFileContent {
  CSV(CSVFileContent),
}

impl InputFileContent {
  pub fn load(input_file: &InputFile) -> Result<Self, IOError> {
    match input_file {
      InputFile::Csv {
        file_path,
        deliminator,
        has_header,
        ..
      } => CSVFileContent::from_file(file_path.clone(), *deliminator, *has_header).map(InputFileContent::CSV),
    }
  }
}

#[derive(Debug, Clone)]
pub struct CSVFileContent {
  fields: Vec<String>,
  field_id_map: BTreeMap<String, usize>,
  rows: Vec<Vec<String>>,
}

impl CSVFileContent {
  pub fn from_file(file_path: PathBuf, deliminator: u8, has_header: bool) -> Result<Self, IOError> {
    let mut rdr = csv::ReaderBuilder::new()
      .delimiter(deliminator)
      .has_headers(has_header)
      .from_path(file_path.clone())
      .map_err(|e| IOError::CannotOpenFile {
        file_path,
        error: format!("{}", e),
      })?;

    // Generate fields
    let (fields, field_id_map) = if has_header {
      let fields = rdr
        .headers()
        .map_err(|e| IOError::CannotReadHeader {
          error: format!("{}", e),
        })?
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
      let field_id_map = fields.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
      (fields, field_id_map)
    } else {
      (vec![], BTreeMap::new())
    };

    // Generate rows
    let mut rows = Vec::new();
    for result in rdr.records() {
      let record = result.map_err(|e| IOError::CannotParseCSV {
        error: format!("{}", e),
      })?;
      rows.push(record.iter().map(|s| s.to_string()).collect());
    }

    Ok(Self {
      fields,
      field_id_map,
      rows,
    })
  }

  pub fn num_columns(&self) -> usize {
    self.fields.len()
  }

  pub fn num_rows(&self) -> usize {
    self.rows.len()
  }

  pub fn get_header_id(&self, header: &str) -> Option<usize> {
    self.field_id_map.get(header).copied()
  }

  pub fn get_ith_header(&self, i: usize) -> Option<&String> {
    self.fields.get(i)
  }

  pub fn headers(&self) -> &Vec<String> {
    &self.fields
  }

  pub fn iter_headers(&self) -> impl Iterator<Item = &String> {
    self.fields.iter()
  }

  pub fn get_rows(&self) -> impl Iterator<Item = &Vec<String>> {
    self.rows.iter()
  }
}
