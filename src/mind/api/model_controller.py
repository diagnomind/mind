import time
import cv2
from fastapi import FastAPI, File, Response, UploadFile
import numpy as np
from mind.model.predictor import Predictor

CONTENT_TYPE = "image/png"

_app = FastAPI()
_predictor = Predictor()

@_app.post("/predict")
async def predict(uploaded_img: UploadFile = File(...)) -> Response:
    try:
        raw_img = cv2.imdecode(np.frombuffer(await uploaded_img.read(), np.uint8), cv2.IMREAD_COLOR)
        predicted =  _predictor.predict(raw_img, 0.01)
        _, encoded_image = cv2.imencode(".png", predicted)
        headers = {
            "Content-Type": CONTENT_TYPE,
            "Content-Disposition": "inline; filename=prediction.png",
        }
        return Response(content=encoded_image.tobytes(), media_type=CONTENT_TYPE, headers=headers)
    except Exception as e:
        return Response(content=f"Error: {str(e)}", status_code=500)
    
@_app.get("/simulation")
async def simulation() -> Response:
    noise_image = np.random.default_rng(int(time.time())).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    _ , encoded_image = cv2.imencode(".png", noise_image)
    headers = {
        "Content-Type": CONTENT_TYPE,
        "Content-Disposition": "inline; filename=prediction.png",
    }
    return Response(content=encoded_image.tobytes(), media_type=CONTENT_TYPE, headers=headers)
