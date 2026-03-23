from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.post("/task")
async def handle_task(request: dict):
    question = request["question"]
    answer = call_my_llm(question)   # your agent logic here
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
