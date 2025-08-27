import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def index() :
    return { "message": "returned success good" }

@app.post("/")
def postIndex():
    return { "message": "returned post" }



if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)