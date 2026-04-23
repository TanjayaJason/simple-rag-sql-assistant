from fastapi import FastAPI
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.post("/chat")
def chat(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": False
        }
    )
    
    result = response.json()
    return {"response": result["response"]}