use std::fs::File;
use std::path::*;

use csv::WriterBuilder;

use crate::common::output_option::*;
use crate::runtime::error::*;
use crate::runtime::provenance::*;
use crate::utils::*;

use super::super::*;

pub fn store_file<Prov: Provenance, Ptr: PointerFamily>(
  output_file: &OutputFile,
  idb_relation: &intentional::IntentionalRelation<Prov, Ptr>,
) -> Result<(), IOError> {
  match output_file {
    OutputFile::CSV(f) => store_csv_file(&f.file_path, f.deliminator, idb_relation),
  }
}

pub fn store_csv_file<Prov: Provenance, Ptr: PointerFamily>(
  file_path: &PathBuf,
  deliminator: u8,
  idb_relation: &intentional::IntentionalRelation<Prov, Ptr>,
) -> Result<(), IOError> {
  // Then load the file
  let file = File::create(file_path).map_err(|e| IOError::CannotOpenFile {
    file_path: file_path.clone(),
    error: format!("{}", e),
  })?;

  // Write the tuples to the file
  let mut wtr = WriterBuilder::new().delimiter(deliminator).from_writer(file);
  for (_, tuple) in Ptr::get_rc(&idb_relation.recovered_facts).iter() {
    let record = tuple.as_ref_values().into_iter().map(|v| format!("{}", v));
    wtr
      .write_record(record)
      .map_err(|e| IOError::CannotWriteRecord { error: e.to_string() })?;
  }

  Ok(())
}
