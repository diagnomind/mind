import uvicorn

if __name__ == "__main__":
    uvicorn.run("mind.api.model_controller:_app", app_dir="src/", host="127.0.0.1", port=8080, log_level="info", reload=True)

