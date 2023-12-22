from fastapi import FastAPI, File, Response, UploadFile, status


class ModelController:
    __app: FastAPI = FastAPI()

    @__app.post("/predict")
    async def predict(self, uploaded_img: UploadFile = File(...)) -> Response:
        content: bytes = await uploaded_img.read() # This content will be sent to the model.
        # prediction: bytes = predict()
        # return StreamingResponse(content, media_type="image/png")
        raise NotImplementedError()
        return Response(status_code=status.HTTP_204_NO_CONTENT)