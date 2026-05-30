import json
import os
from datetime import datetime

ARQUIVO_MEMORIA = "memoria.json"

def carregar_memoria() -> dict:
    if os.path.exists(ARQUIVO_MEMORIA):
        with open(ARQUIVO_MEMORIA, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"conversas": [], "fatos": {}}

def salvar_memoria(memoria: dict):
    with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=2)

def adicionar_conversa(memoria: dict, pergunta: str, resposta: str):
    memoria["conversas"].append({
        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "pergunta": pergunta,
        "resposta": resposta
    })
    # mantém só as últimas 20 conversas
    memoria["conversas"] = memoria["conversas"][-20:]
    salvar_memoria(memoria)

def salvar_fato(memoria: dict, chave: str, valor: str):
    memoria["fatos"][chave] = {
        "valor": valor,
        "salvo_em": datetime.now().strftime("%d/%m/%Y %H:%M")
    }
    salvar_memoria(memoria)

def resumo_memoria(memoria: dict) -> str:
    if not memoria["fatos"] and not memoria["conversas"]:
        return "Nenhuma memória ainda."

    partes = []

    if memoria["fatos"]:
        partes.append("Fatos que eu sei sobre você:")
        for chave, dados in memoria["fatos"].items():
            partes.append(f"  - {chave}: {dados['valor']}")

    if memoria["conversas"]:
        ultimas = memoria["conversas"][-3:]
        partes.append("\nÚltimas conversas:")
        for c in ultimas:
            partes.append(f"  [{c['data']}] {c['pergunta'][:60]}...")

    return "\n".join(partes)
