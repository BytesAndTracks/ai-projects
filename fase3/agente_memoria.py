import ollama
import json
import math
from datetime import datetime
from memoria import (
    carregar_memoria, adicionar_conversa,
    salvar_fato, resumo_memoria
)

def calcular(expressao: str) -> str:
    try:
        expressao = expressao.replace(",", ".")
        resultado = eval(expressao, {"__builtins__": {}}, {"math": math})
        return str(resultado)
    except Exception as e:
        return f"Erro: {e}"

def data_atual() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M")

def buscar_web(termo: str) -> str:
    return f"Busca simulada para '{termo}' — será conectada na próxima etapa."

def lembrar_fato(texto: str) -> str:
    # espera formato "chave: valor"
    if ":" in texto:
        chave, valor = texto.split(":", 1)
        memoria = carregar_memoria()
        salvar_fato(memoria, chave.strip(), valor.strip())
        return f"Memorizado: {chave.strip()} = {valor.strip()}"
    return "Formato inválido. Use 'chave: valor'"

def ver_memoria() -> str:
    memoria = carregar_memoria()
    return resumo_memoria(memoria)

FERRAMENTAS = {
    "calcular":    {"fn": calcular,    "desc": "Calcula expressões matemáticas"},
    "data_atual":  {"fn": lambda _: data_atual(), "desc": "Retorna data e hora atual"},
    "buscar_web":  {"fn": buscar_web,  "desc": "Busca informações na web"},
    "lembrar_fato":{"fn": lembrar_fato,"desc": "Salva um fato. Use 'chave: valor'"},
    "ver_memoria": {"fn": lambda _: ver_memoria(), "desc": "Mostra o que foi memorizado"},
}

def montar_system_prompt():
    memoria = carregar_memoria()
    resumo = resumo_memoria(memoria)

    return f"""Você é um assistente pessoal com memória persistente.

MEMÓRIA ATUAL:
{resumo}

Ferramentas disponíveis:
- calcular(expressao): cálculos matemáticos
- data_atual(): data e hora atual  
- buscar_web(termo): busca na web
- lembrar_fato(chave: valor): salva informações importantes sobre o usuário
- ver_memoria(): mostra tudo que foi memorizado

Quando identificar informações importantes sobre o usuário (nome, preferências,
trabalho, metas), use lembrar_fato automaticamente para guardar.

Responda SEMPRE em JSON:
{{"acao": "nome_ferramenta", "parametro": "valor"}}
ou
{{"acao": "resposta_final", "parametro": "sua resposta"}}"""

def rodar_agente(pergunta: str):
    historico = [
        {"role": "system", "content": montar_system_prompt()},
        {"role": "user",   "content": pergunta}
    ]

    print(f"\nPergunta: {pergunta}")
    print("-" * 50)

    resposta_final = None

    for passo in range(10):
        resposta = ollama.chat(
            model="llama3.2",
            messages=historico,
            options={"temperature": 0.1}
        )

        conteudo = resposta["message"]["content"].strip()

        try:
            limpo = conteudo.replace("```json", "").replace("```", "").strip()
            acao = json.loads(limpo)
            nome = acao.get("acao", "")
            parametro = acao.get("parametro", "")

            if nome == "resposta_final":
                print(f"\nResposta: {parametro}")
                resposta_final = parametro
                break

            elif nome in FERRAMENTAS:
                print(f"[passo {passo+1}] {nome}({parametro!r})")
                resultado = FERRAMENTAS[nome]["fn"](parametro)
                print(f"           → {resultado}")
                historico.append({"role": "assistant", "content": conteudo})
                historico.append({"role": "user", "content": f"Resultado: {resultado}"})

            else:
                print(f"[passo {passo+1}] ação desconhecida: {nome}")
                break

        except json.JSONDecodeError:
            print(f"\nResposta: {conteudo}")
            resposta_final = conteudo
            break

    if resposta_final:
        memoria = carregar_memoria()
        adicionar_conversa(memoria, pergunta, resposta_final)

if __name__ == "__main__":
    print("Assistente com memória persistente")
    print("Tente: 'meu nome é Rubens', depois feche e abra de novo\n")

    while True:
        pergunta = input("Você: ").strip()
        if pergunta.lower() == "sair":
            break
        if pergunta:
            rodar_agente(pergunta)
            print()
