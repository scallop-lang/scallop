# Static Runtime

The static runtime will be used in a static compilation environment.
The type for each relation will be determined during compile time, and
all the types are templated.
The compilation will be delegated to Rust compiler for best optimization
opportunities.
Once the `ram` updates are compiled into static dataflow insertions,
they cannot be removed or changed since they are part of the statically
compiled code.
However, it is possible to interface the static program with dynamic programs--
dynamic dataflows can take static relations as input and carry the dynamic
execution forward.

Ideally, one should maximize the statically compiled/executed portion of
any Scallop code, and add light-weight dynamic execution on the top.
