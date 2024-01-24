import cv2
from importlib.resources import files
from keras.models import load_model, Model
import numpy as np
from typing import Any

MODEL_NAME: str = "mind.keras"

class Predictor:
    def __init__(self):
        self._model: Model | Any = load_model(files("mind.model").joinpath(MODEL_NAME))

    def predict(self, image: np.ndarray, threshold: float) -> np.ndarray:
        if image.shape != (256, 256, 3):
            raise ValueError(image.shape, (256, 256 ,3))
        prediction = self._model.predict(np.expand_dims(self._preprocess(image), axis=0))
        return self._blend(image, ((prediction.squeeze() > threshold) * 255).astype(np.uint8), 0.5)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        processed_image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) # type: ignore
        return processed_image
    
    def _blend(self, image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
        return cv2.addWeighted(image, 1 - alpha, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), alpha, 0)