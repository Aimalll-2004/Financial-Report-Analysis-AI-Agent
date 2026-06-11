from fastapi import FastAPI
from pydantic import BaseModel

from graph import app as graph_app

app = FastAPI()

class Query(BaseModel):
    question: str

#API endpoint
@app.post("/ask")
def question(query: Query):

    result = graph_app.invoke({
        "question": query.question,
        "route": "",
        "result": "",
        "answer": ""
    })

    return {
        "question": query.question,
        "route": result["route"],
        "result": result["result"],
        "answer": result["answer"]
    }