import cv2
from fastapi import FastAPI, File, Response, UploadFile
import numpy as np
from mind.model.predictor import Predictor

_app = FastAPI()
_predictor = Predictor()

@_app.post("/predict")
async def predict(uploaded_img: UploadFile = File(...)) -> Response:
    try:
        raw_img = cv2.imdecode(np.frombuffer(await uploaded_img.read(), np.uint8), cv2.IMREAD_COLOR)
        predicted =  _predictor.predict(raw_img, 0.01)
        _, encoded_image = cv2.imencode(".png", predicted)
        headers = {
            "Content-Type": "image/png",
            "Content-Disposition": "inline; filename=prediction.png",
        }
        return Response(content=encoded_image.tobytes(), media_type="image/png", headers=headers)
    except Exception as e:
        return Response(content=f"Error: {str(e)}", status_code=500)
    
@_app.post("/simulation")
async def simulation(uploaded_img: UploadFile = File(...)) -> Response:
    try:
        raw_img = cv2.imdecode(np.frombuffer(await uploaded_img.read(), np.uint8), cv2.IMREAD_COLOR)
        _, encoded_image = cv2.imencode(".png", raw_img)
        headers = {
            "Content-Type": "image/png",
            "Content-Disposition": "inline; filename=prediction.png",
        }
        return Response(content=encoded_image.tobytes(), media_type="image/png", headers=headers)
    except Exception as e:
        return Response(content=f"Error: {str(e)}", status_code=500)