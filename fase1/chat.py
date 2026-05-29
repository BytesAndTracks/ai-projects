import ollama

historico = [
    {
        "role": "system",
        "content": (
            "Você é um assistente especialista em produtos financeiros do Itaú. "
            "Responda sempre em português, de forma clara e objetiva. "
            "Nunca invente informações sobre taxas ou produtos — se não souber, diga que não sabe."
        )
    }
]

print("Chat local com LLaMA 3.2 — digite 'sair' para encerrar\n")

while True:
    entrada = input("Você: ")
    
    if entrada.lower() == "sair":
        print("Encerrando chat.")
        break

    historico.append({
        "role": "user",
        "content": entrada
    })

    resposta = ollama.chat(
        model="llama3.2",
        messages=historico
    )

    mensagem = resposta["message"]["content"]
    
    historico.append({
        "role": "assistant",
        "content": mensagem
    })

    print(f"\nModelo: {mensagem}\n")