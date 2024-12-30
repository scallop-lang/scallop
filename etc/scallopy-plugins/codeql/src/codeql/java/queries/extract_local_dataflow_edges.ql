import java
import semmle.code.java.dataflow.DataFlow

from
  DataFlow::Node source,
  DataFlow::Node sink
where
  DataFlow::localFlow(source, sink)
select
  source.toString() as source_node,
  source.getLocation().getStartLine() as source_start_line,
  source.getLocation().getStartColumn() as source_start_column,
  sink.toString() as sink_node,
  sink.getLocation().getStartLine() as sink_start_line,
  sink.getLocation().getStartColumn() as sink_start_column,
  source.getLocation().getFile().getRelativePath() as file_name,
  source.getEnclosingCallable().(Method).getName() as method,
  source.getEnclosingCallable().(Method).getLocation().getStartLine() as method_start_line,
  source.getEnclosingCallable().(Method).getLocation().getStartColumn() as method_start_column
