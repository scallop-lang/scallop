use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn encode_entity<V: Hash, I: Iterator<Item = V>>(functor: &str, args: I) -> u64 {
  let mut s = DefaultHasher::new();
  functor.hash(&mut s);
  for arg in args {
    arg.hash(&mut s);
  }
  s.finish()
}
