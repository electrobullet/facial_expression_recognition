import cv2 as cv
import numpy as np

from ..FaceDetector import FaceDetector
from . import core


class FaceDetectionRetail0044(FaceDetector):
    def __init__(self, model_path: str, device: str = 'CPU') -> None:
        self._model = core.compile_model(core.read_model(model_path), device)

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        res = cv.resize(image, (300, 300))
        res = res.transpose([2, 0, 1])
        res = np.expand_dims(res, 0)

        return res.astype(np.float32)

    def predict(self, image: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        result = []

        h, w = image.shape[:2]

        predictions = self._model.infer_new_request({0: FaceDetectionRetail0044.preprocess(image)})
        predictions = next(iter(predictions.values())).reshape(-1, 7)
        predictions = predictions[predictions[:, 2] > threshold]

        for _, _, _, x_min, y_min, x_max, y_max in predictions:
            result.append([
                np.clip(x_min * w, 0, w),
                np.clip(y_min * h, 0, h),
                np.clip(x_max * w, 0, w),
                np.clip(y_max * h, 0, h),
            ])

        return np.array(result, dtype=np.int16)
