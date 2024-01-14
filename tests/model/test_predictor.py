import numpy as np
import unittest
from unittest.mock import Mock, patch

from mind.model.predictor import Predictor
from mind.model.predictor import Model

class TestPredictor(unittest.TestCase):
    
    @patch("mind.model.predictor.load_model")
    def setUp(self, mock_load_model):
        mock_load_model.return_value = Model()
        self._rng = np.random.default_rng(42)
        self._predictor = Predictor()
        self._image_shape = (256, 256, 3)
    
    
    @patch.object(Model, "predict")
    @patch.object(Predictor, "_blend")
    def test_predict_correct(self, mock_predict, mock_blend):
        mock_blend.return_value = mock_predict.return_value = np.zeros(shape=self._image_shape)
        result = self._predictor.predict(self._rng.integers(0, 256, size=self._image_shape, dtype=np.uint8), 0)
        self.assertTrue(np.array_equal(result, np.zeros(shape=self._image_shape)))

    def test_predict_incorrect(self):
        with self.assertRaises(ValueError):
            self._predictor.predict(np.zeros((1, 1)), 0)

    def test_preprocess(self):
        preprocessed_img = self._predictor._preprocess(self._rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8))
        self.assertTrue(np.all((preprocessed_img >= 0) & (preprocessed_img <= 1)))
    
    @patch("mind.model.predictor.cv2.addWeighted")
    @patch("mind.model.predictor.cv2.cvtColor")
    def test_blend(self, mock_cvt_color, mock_add_weighted):
        random_image = self._rng.integers(0, 256, size=(256, 256, 3))
        mock_cvt_color.return_value = random_image
        mock_add_weighted.return_value = np.zeros(shape=self._image_shape)
        self.assertTrue(np.array_equal(self._predictor._blend(random_image, random_image, 0), np.zeros(shape=self._image_shape)))
