from typing import List, Tuple

import cv2 as cv
import numpy as np

from ..EmotionRecognizer import EmotionRecognizer
from . import core


class EfficientFER(EmotionRecognizer):
    def __init__(self, model_path: str, input_size: Tuple[int, int],
                 emotions: List[str] = None, device: str = 'CPU') -> None:  # type: ignore
        self._model = core.compile_model(core.read_model(model_path), device)
        self._input_size = input_size
        self._emotions = emotions if emotions else sorted(EmotionRecognizer.COLORS.keys())
        self._colors = [EmotionRecognizer.COLORS[label] for label in self._emotions]

    @property
    def emotions(self) -> List[str]:
        return self._emotions

    @property
    def colors(self) -> List[Tuple[int, int, int]]:
        return self._colors

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        res = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        res = cv.resize(res, self._input_size, interpolation=cv.INTER_LINEAR)
        res = np.expand_dims(res, (0, -1))

        return res.astype(np.float32)

    def predict(self, face_crop: np.ndarray) -> np.ndarray:
        predictions = self._model.infer_new_request({0: self.preprocess(face_crop)})
        predictions = next(iter(predictions.values()))

        return predictions.reshape(len(self._emotions))
