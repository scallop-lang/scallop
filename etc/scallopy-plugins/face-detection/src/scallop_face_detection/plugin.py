import scallopy

from PIL import Image
import face_detection as face_detection_lib
from scallop_gpu import get_device

class ScallopFaceDetectionPlugin(scallopy.Plugin):
  def __init__(self, dump_image_path = ".tmp/scallop-face"):
    super().__init__()

    self._dump_image_path = dump_image_path
    self._face_model = None

  def dump_image_path(self) -> str:
    return self._dump_image_path

  def get_face_model(self):
    if self._face_model is None:
      try:
        self._face_model = face_detection_lib.build_detector(
          "DSFDDetector",
          confidence_threshold=0.5,
          nms_iou_threshold=0.3,
          device=get_device(),
        )
      except:
        return None

    return self._face_model

  def load_into_ctx(self, ctx: scallopy.ScallopContext):
    from .face_detection import get_face_detection
    ctx.register_foreign_attribute(get_face_detection(self))
