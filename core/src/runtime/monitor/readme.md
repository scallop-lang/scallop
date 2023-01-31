# Monitors

Monitor in Scallop is a general structure for monitoring Scallop runtime system operations.
Such operations include tagging, loading, executing stratum, fixed-point iteration, recovering, and so on.
There are many implemented monitors in Scallop, taking responsibility of logging events, computing running time, tracking tags and computation graph, and so on.
Advanced usage of monitors involve exporting system diagnostics as HTML or JSON files.

Note that monitors can only passively observe system events but not actively affecting it.
Also monitors themselves are required to be immutable.
Any mutable monitors need to possess internal mutability through use of `RefCell` or `Mutex`.
This is to make sure that a uniform interface is maintained for monitors involving parallel computing and so on.
