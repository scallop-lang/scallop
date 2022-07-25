from typing import List, Dict

class HistoryAction:
  def __init__(self, func_name: str, pos_args: List, kw_args: Dict):
    self.func_name = func_name
    self.pos_args = pos_args
    self.kw_args = kw_args

  def __repr__(self):
    pos_args_str = ", ".join([repr(a) for a in self.pos_args])
    if len(self.kw_args) == 0:
      return f"{self.func_name}({pos_args_str})"
    else:
      kw_args_str = ", ".join([f"{k}={v}" for (k, v) in self.kw_args.items()])
      return f"{self.func_name}({pos_args_str}, {kw_args_str})"


# A decorator for recording the history
def record_history(f):
  def wrapper(this, *pos_args, **kw_args):
    action = HistoryAction(f.__name__, pos_args, kw_args)
    this._history_actions.append(action)
    return f(this, *pos_args, **kw_args)
  return wrapper
