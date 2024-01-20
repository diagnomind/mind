import asyncio
import unittest
from unittest.mock import MagicMock, patch
from urllib import response
from fastapi import UploadFile
import numpy as np

from mind.model.predictor import Predictor, Model

patch("mind.model.predictor.load_model", new=MagicMock(spec=Model())).start()

from mind.api.model_controller import predict, simulation


class TestPredictEndpoint(unittest.TestCase):

    @patch("cv2.imdecode")
    @patch.object(Predictor, "predict")
    @patch("cv2.imencode")
    def test_predict_endpoint_correct(self, mock_imencode, mock_predict, mock_imdecode):
        mock_imdecode.return_value = MagicMock()
        mock_predict.return_value = MagicMock()
        mock_imencode.return_value = (None, np.frombuffer(b"mocked_encoded_image", dtype=np.uint8))
        fake_uploaded_img = MagicMock(spec=UploadFile)

        with patch.object(fake_uploaded_img, "read") as mock_read:
            mock_read.return_value = b"mocked_image_content"
            response = asyncio.run(predict(fake_uploaded_img))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "image/png")
        self.assertEqual(response.headers["Content-Disposition"], "inline; filename=prediction.png")
        self.assertEqual(response.body, b"mocked_encoded_image")

    def test_predict_endpoint_incorrect(self):
        fake_uploaded_img = MagicMock(spec=UploadFile)
        with patch.object(fake_uploaded_img, "read", side_effect=Exception("Simulated reading error")) as mock_read:
            mock_read.return_value = MagicMock()
            response = asyncio.run(predict(fake_uploaded_img))

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.body, b'Error: Simulated reading error')
    
    @patch("cv2.imdecode")
    @patch.object(Predictor, "predict")
    @patch("cv2.imencode")
    def test_simulation_endpoint_correct(self, mock_imencode, mock_predict, mock_imdecode):
        mock_imdecode.return_value = MagicMock()
        mock_predict.return_value = MagicMock()
        mock_imencode.return_value = (None, np.frombuffer(b"mocked_encoded_image", dtype=np.uint8))

        response = asyncio.run(simulation())

        self.assertEqual(response.status_code, 200)

 
