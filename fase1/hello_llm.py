import requests

def perguntar(prompt):
    resposta = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    return resposta.json()["response"]

if __name__ == "__main__":
    pergunta = "Explique o que é inferência em LLMs em 2 frases simples."
    print(perguntar(pergunta))
