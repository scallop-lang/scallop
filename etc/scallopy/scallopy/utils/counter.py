class Counter:
  def __init__(self):
    self.count = 0

  def get_and_increment(self) -> int:
    result = self.count
    self.count += 1
    return result
