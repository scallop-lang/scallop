mod aggregation;
mod antijoin;
pub mod batching;
mod collection;
mod difference;
mod filter;
mod find;
mod intersection;
mod join;
mod product;
mod project;
mod relation;
mod union;
mod unit;
mod utils;
mod vec;

pub use aggregation::*;
pub use antijoin::*;
pub use collection::*;
pub use difference::*;
pub use filter::*;
pub use find::*;
pub use intersection::*;
pub use join::*;
pub use product::*;
pub use project::*;
pub use relation::*;
pub use union::*;
pub use unit::*;

// Module specific
use super::*;
use crate::runtime::provenance::*;
use batching::*;

/// A dataflow trait
///
/// A dataflow will be divided into stable parts and recent parts.
/// Each part returns a sequence of batches (as defined by `Batches`).
/// One batch will further be iterated and collected into individual
/// elements.
///
/// The sequence of recent batches represent the new elements that
/// we want to add into the system. This is as opposed to the sequence
/// of stable batches, which represents the elements that are already
/// inside of the system.
///
/// Inside a statically compiled dataflow, stable batches and recent
/// batches can have separate types. Any type that instantiates this
/// dataflow trait must provide a `Stable` type and a `Recent` type.
/// Henceforth, the `iter_stable` function will return the sequence
/// of stable batches, and the `iter_recent` function will return the
/// sequence of recent batches.
pub trait Dataflow<Tup, Prov>: Sized + Clone
where
  Tup: StaticTupleTrait,
  Prov: Provenance,
{
  type Stable: Batches<Tup, Prov>;

  type Recent: Batches<Tup, Prov>;

  fn iter_stable(&self) -> Self::Stable;

  fn iter_recent(self) -> Self::Recent;
}
