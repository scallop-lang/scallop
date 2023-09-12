import scallopy

from .face_detection import face_detection

def load_into_context(ctx: scallopy.Context):
  ctx.register_foreign_attribute(face_detection)
