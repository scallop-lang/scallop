extern crate lalrpop;

fn main() {
  lalrpop::Configuration::new()
    .set_in_dir("src")
    .set_out_dir("src")
    .process()
    .unwrap();
}
