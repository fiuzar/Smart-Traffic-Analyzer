import uvicorn
from fastapi import FastAPI
from model_class import GetOd

app = FastAPI()

@app.get("/")
def index() :
    return { "message": "returned success good" }

@app.post("/")
def postIndex():
    return { "message": "returned post" }

@app.post("/video/{id}")
def get_video(id):
    name = GetOd.get_name(id)
    return { 'message': name }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=True)