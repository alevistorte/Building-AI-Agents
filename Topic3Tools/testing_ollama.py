import requests
import json


def call_ollama(prompt, model="llama3.2"):
    response = requests.post('http://localhost:11434/api/generate',
                             json={
                                 "model": model,
                                 "prompt": prompt,
                                 "stream": False
                             })
    return response.json()['response']


# Use it
result = call_ollama("Explain transformers in one sentence")
print(result)
