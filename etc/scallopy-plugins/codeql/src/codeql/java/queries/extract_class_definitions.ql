import java

from
  RefType c
where
  c.fromSource() and
  c.getName() != ""
select
  c.getName() as name,
  c.getFile().getRelativePath() as file,
  c.getLocation().getStartLine() as start_line,
  c.getLocation().getEndLine() + c.getTotalNumberOfLines() as end_line
