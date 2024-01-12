import cv2
from typing import Any
from keras.models import load_model, Model
import numpy as np

MODEL_DIR: str = "data/model/"
MODEL_NAME: str = "mind.keras"

class Predictor:
    def __init__(self):
        self._model: Model | Any = load_model(MODEL_DIR + MODEL_NAME)

    def predict(self, image: np.ndarray) -> np.ndarray:
        if image.shape != (256, 256, 3):
            raise ValueError(image.shape, (256, 256 ,3))
        prediction = self._model.predict(self._preprocess(image))
        return self._blend(image, prediction.squeeze() > 0.95, 0.5)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        processed_image = np.zeros(image.shape)
        cv2.normalize(image, processed_image, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return processed_image
    
    def _blend(self, image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
        return cv2.addWeighted(image, 1 - alpha, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), alpha, 0)