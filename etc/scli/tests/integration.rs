use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

#[test]
fn file_doesnt_exist_1() -> Result<(), Box<dyn std::error::Error>> {
  let mut cmd = Command::cargo_bin("scli")?;

  cmd.arg("test/file/doesnt/exist");
  cmd.assert()
    .failure()
    .stderr(predicate::str::contains("Cannot open file test/file/doesnt/exist"));

  Ok(())
}
