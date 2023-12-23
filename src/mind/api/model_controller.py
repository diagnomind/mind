from fastapi import FastAPI, File, Response, UploadFile


class ModelController:
    __app: FastAPI = FastAPI()

    @__app.post("/predict")
    async def predict(self, uploaded_img: UploadFile = File(...)) -> Response:
        raise NotImplementedError()
    
    @__app.post("/train")
    async def train(self) -> Response:
        raise NotImplementedError()