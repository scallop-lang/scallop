import java

from
  Method c
where
  c.fromSource() and
  c.getName() != ""
select
  c.getName() as method_name,
  c.getDeclaringType() as class_name,
  c.getFile().getRelativePath() as file_name,
  c.getLocation().getStartLine() as start_line,
  c.getLocation().getStartColumn() as start_column
