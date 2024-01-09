from fastapi import FastAPI, File, Response, UploadFile


class ModelController:
    _app: FastAPI = FastAPI()

    @_app.post("/predict")
    async def predict(self, uploaded_img: UploadFile = File(...)) -> Response:
        raise NotImplementedError()
    
    @_app.post("/train")
    async def train(self) -> Response:
        raise NotImplementedError()