import ollama
import json
import math
from datetime import datetime

# --- Ferramentas disponíveis ---

def calcular(expressao: str) -> str:
    try:
        # corrige notação brasileira
        expressao = expressao.replace(",", ".")
        resultado = eval(expressao, {"__builtins__": {}}, {"math": math})
        return str(resultado)
    except Exception as e:
        return f"Erro ao calcular: {e}"

def data_atual() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M")

def buscar_web(termo: str) -> str:
    # Simulado por enquanto — na fase 4 conectamos uma API real
    return f"Resultado simulado para '{termo}': esta ferramenta será conectada a uma API de busca real na próxima fase."

FERRAMENTAS = {
    "calcular": {
        "fn": calcular,
        "descricao": "Calcula expressões matemáticas. Ex: '2 + 2', '10 * 3.5', 'math.sqrt(144)'"
    },
    "data_atual": {
        "fn": lambda: data_atual(),
        "descricao": "Retorna a data e hora atual."
    },
    "buscar_web": {
        "fn": buscar_web,
        "descricao": "Busca informações na web sobre um termo."
    }
}

# --- System prompt do agente ---

SYSTEM_PROMPT = """Você é um agente inteligente que resolve problemas usando ferramentas.

Ferramentas disponíveis:
- calcular(expressao): calcula expressões matemáticas
- data_atual(): retorna a data e hora atual
- buscar_web(termo): busca informações na web

Quando precisar usar uma ferramenta, responda EXATAMENTE neste formato JSON:
{"acao": "nome_da_ferramenta", "parametro": "valor"}

Quando tiver a resposta final, responda EXATAMENTE neste formato JSON:
{"acao": "resposta_final", "parametro": "sua resposta aqui"}

IMPORTANTE: decomponha problemas complexos em múltiplos passos.
Para problemas com várias etapas, use a ferramenta calcular múltiplas vezes.
Exemplo: 'quanto sobra por ano' = calcular mensal → depois multiplicar por 12.

Pense passo a passo. Use ferramentas quantas vezes precisar antes de responder.

NUNCA faça cálculos matemáticos de cabeça. Sempre use a ferramenta calcular,
inclusive para operações simples. Se precisar de múltiplos cálculos, chame
a ferramenta múltiplas vezes, uma operação por vez."""

# --- Loop do agente ---

def rodar_agente(pergunta: str, verbose: bool = True):
    historico = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pergunta}
    ]

    print(f"\nPergunta: {pergunta}")
    print("-" * 50)

    for passo in range(10):  # máximo 10 iterações
        resposta = ollama.chat(
            model="llama3.2",
            messages=historico,
            options={"temperature": 0.1}  # baixa temperatura = mais determinístico
        )

        conteudo = resposta["message"]["content"].strip()

        # tenta parsear como JSON
        try:
            # remove possíveis blocos de código markdown
            conteudo_limpo = conteudo.replace("```json", "").replace("```", "").strip()
            acao = json.loads(conteudo_limpo)

            nome = acao.get("acao", "")
            parametro = acao.get("parametro", "")

            if nome == "resposta_final":
                print(f"\nResposta final: {parametro}")
                return parametro

            elif nome in FERRAMENTAS:
                if verbose:
                    print(f"[passo {passo+1}] usando ferramenta: {nome}({parametro!r})")

                ferramenta = FERRAMENTAS[nome]["fn"]
                if parametro:
                    resultado = ferramenta(parametro)
                else:
                    resultado = ferramenta()

                if verbose:
                    print(f"           resultado: {resultado}")

                # devolve o resultado pro modelo continuar raciocínando
                historico.append({"role": "assistant", "content": conteudo})
                historico.append({"role": "user", "content": f"Resultado da ferramenta: {resultado}"})

            else:
                if verbose:
                    print(f"[passo {passo+1}] ação desconhecida: {nome}")
                break

        except json.JSONDecodeError:
            # modelo respondeu em texto livre — trata como resposta final
            print(f"\nResposta: {conteudo}")
            return conteudo

    return "Agente não conseguiu concluir."

# --- Interface de chat ---

if __name__ == "__main__":
    print("Agente ReAct local — digite 'sair' para encerrar\n")
    print("Ferramentas disponíveis:")
    for nome, info in FERRAMENTAS.items():
        print(f"  - {nome}: {info['descricao']}")
    print()

    while True:
        pergunta = input("Você: ").strip()
        if pergunta.lower() == "sair":
            break
        if not pergunta:
            continue
        rodar_agente(pergunta)
        print()
