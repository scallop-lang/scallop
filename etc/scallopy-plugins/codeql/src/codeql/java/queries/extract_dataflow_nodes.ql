import java
import semmle.code.java.dataflow.DataFlow

from
  DataFlow::Node n
select
  n.toString() as node,
  n.getLocation().getFile().getRelativePath() as file_name,
  n.getLocation().getStartLine() as start_line,
  n.getLocation().getStartColumn() as start_column,
  n.getEnclosingCallable().getName() as enclosing_method,
  n.getEnclosingCallable().getLocation().getStartLine() as enclosing_method_start_line,
  n.getEnclosingCallable().getLocation().getStartColumn() as enclosing_method_start_column
