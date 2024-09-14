class NodeLocation:
  def __init__(self, internal_location):
    self._offset_span = (internal_location["offset_span"]["start"], internal_location["offset_span"]["end"])
    self._id = internal_location["id"]
    self._souce_id = internal_location["source_id"]
    if internal_location["loc_span"]:
      self._loc_span = (
        (internal_location["loc_span"]["start"]["row"], internal_location["loc_span"]["start"]["col"]),
        (internal_location["loc_span"]["end"]["row"], internal_location["loc_span"]["end"]["col"]))
    else:
      self._loc_span = None

  def __repr__(self):
    if self._id is not None:
      if self._loc_span is not None:
        return f"[#{self._id} {self._loc_span[0][0]}:{self._loc_span[0][1]} - {self._loc_span[1][0]}:{self._loc_span[1][1]}]"
      else:
        return f"[#{self._id} {self._offset_span[0]}-{self._offset_span[1]}]"
    else:
      if self._loc_span is not None:
        return f"[{self._loc_span[0][0]}:{self._loc_span[0][1]} - {self._loc_span[1][0]}:{self._loc_span[1][1]}]"
      else:
        return f"[{self._offset_span[0]}-{self._offset_span[1]}]"
