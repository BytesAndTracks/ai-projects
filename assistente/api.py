import os
import json
import math
import fitz
import ollama
import chromadb
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from chromadb.utils import embedding_functions
from ddgs import DDGS
from memoria import (
    carregar_memoria, adicionar_conversa,
    salvar_fato, resumo_memoria
)

DOCS_DIR = "./docs"
MODELO = "llama3.2"
os.makedirs(DOCS_DIR, exist_ok=True)

app = FastAPI()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
chroma_cliente = chromadb.Client()
colecao = chroma_cliente.get_or_create_collection(
    "documentos", embedding_function=embedding_fn
)

# --- Ferramentas ---

def calcular(expressao: str) -> str:
    try:
        expressao = expressao.replace(",", ".")
        resultado = eval(expressao, {"__builtins__": {}}, {"math": math})
        return str(resultado)
    except Exception as e:
        return f"Erro: {e}"

def data_atual(_=None) -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M")

def buscar_web(termo: str) -> str:
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(termo, max_results=3))
        if not resultados:
            return "Nenhum resultado encontrado."
        partes = []
        for r in resultados:
            partes.append(f"Título: {r['title']}\nResumo: {r['body']}")
        return "\n\n".join(partes)
    except Exception as e:
        return f"Erro na busca: {e}"

def buscar_documentos(pergunta: str) -> str:
    try:
        total = colecao.count()
        if total == 0:
            return "Nenhum documento indexado ainda."
        resultados = colecao.query(query_texts=[pergunta], n_results=3)
        chunks = resultados["documents"][0]
        fontes = [m["fonte"] for m in resultados["metadatas"][0]]
        distancias = resultados["distances"][0]
        partes = []
        for chunk, fonte, dist in zip(chunks, fontes, distancias):
            sim = round((1 - dist) * 100)
            partes.append(f"[{fonte} — {sim}% similar]\n{chunk[:400]}")
        return "\n\n".join(partes)
    except Exception as e:
        return f"Erro ao buscar documentos: {e}"

def lembrar_fato(texto: str) -> str:
    try:
        if ":" in texto:
            chave, valor = texto.split(":", 1)
            memoria = carregar_memoria()
            salvar_fato(memoria, chave.strip(), valor.strip())
            return f"Memorizado: {chave.strip()} = {valor.strip()}"
        return "Use o formato 'chave: valor'"
    except Exception as e:
        return f"Erro: {e}"

def ver_memoria(_=None) -> str:
    return resumo_memoria(carregar_memoria())

FERRAMENTAS = {
    "calcular":          calcular,
    "data_atual":        data_atual,
    "buscar_web":        buscar_web,
    "buscar_documentos": buscar_documentos,
    "lembrar_fato":      lembrar_fato,
    "ver_memoria":       ver_memoria,
}

def montar_system_prompt() -> str:
    resumo = resumo_memoria(carregar_memoria())
    return f"""Você é um assistente pessoal inteligente com memória e acesso a ferramentas.

MEMÓRIA ATUAL:
{resumo}

Ferramentas disponíveis:
- calcular(expressao): cálculos matemáticos. Use ponto como decimal.
- data_atual(): data e hora atual
- buscar_web(termo): busca informações atuais na internet
- buscar_documentos(pergunta): busca nos documentos PDF enviados pelo usuário
- lembrar_fato(chave: valor): salva informações importantes. Ex: "nome: Rubens"
- ver_memoria(): mostra tudo memorizado

REGRAS:
1. Responda SEMPRE em JSON com exatamente um desses formatos:
   {{"acao": "nome_ferramenta", "parametro": "valor"}}
   {{"acao": "resposta_final", "parametro": "sua resposta aqui"}}
2. Se a pergunta mencionar "arquivo", "documento", "PDF" ou for sobre algo pessoal do usuário, use buscar_documentos PRIMEIRO, antes de qualquer outra ferramenta
3. Se buscar_documentos não retornar a informação, diga explicitamente "não encontrei essa informação no documento"
4. Só use buscar_web para informações gerais ou notícias — NUNCA para responder sobre documentos enviados
5. Salve automaticamente informações importantes com lembrar_fato
6. Nunca faça cálculos de cabeça — sempre use calcular
7. Para resposta_final, escreva em português de forma natural e completa"""

# --- Loop do agente ---

def rodar_agente(pergunta: str):
    # sempre busca nos documentos primeiro, independente do modelo
    contexto_docs = buscar_documentos(pergunta)
    
    contexto_extra = ""
    if "Nenhum documento" not in contexto_docs:
        contexto_extra = f"\nTRECHOS RELEVANTES DOS DOCUMENTOS DO USUÁRIO:\n{contexto_docs}\n"

    historico = [
        {"role": "system", "content": montar_system_prompt() + contexto_extra},
        {"role": "user",   "content": pergunta}
    ]
    passos_log = []

    if contexto_extra:
        passos_log.append({
            "ferramenta": "buscar_documentos",
            "parametro": pergunta,
            "resultado": contexto_docs[:200]
        })

    for _ in range(8):
        resposta = ollama.chat(
            model=MODELO,
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
                return parametro, passos_log

            elif nome in FERRAMENTAS and nome != "buscar_documentos":
                resultado = FERRAMENTAS[nome](parametro)
                passos_log.append({
                    "ferramenta": nome,
                    "parametro": parametro,
                    "resultado": resultado[:200]
                })
                historico.append({"role": "assistant", "content": conteudo})
                historico.append({"role": "user", "content": f"Resultado: {resultado}"})

            else:
                # modelo tentou buscar documentos de novo — já temos o contexto
                historico.append({"role": "assistant", "content": conteudo})
                historico.append({"role": "user", "content": f"Resultado: {contexto_docs}"})

        except json.JSONDecodeError:
            return conteudo, passos_log

    return "Não consegui concluir.", passos_log

# --- Endpoints ---

class Pergunta(BaseModel):
    texto: str

@app.post("/perguntar")
async def perguntar(pergunta: Pergunta):
    def gerar():
        resposta, passos = rodar_agente(pergunta.texto)

        # envia os passos primeiro
        if passos:
            yield f"data: {json.dumps({'tipo': 'passos', 'passos': passos})}\n\n"

        # streaming da resposta palavra por palavra
        palavras = resposta.split(" ")
        for i, palavra in enumerate(palavras):
            chunk = palavra + (" " if i < len(palavras) - 1 else "")
            yield f"data: {json.dumps({'tipo': 'token', 'texto': chunk})}\n\n"

        # salva na memória
        memoria = carregar_memoria()
        adicionar_conversa(memoria, pergunta.texto, resposta)

        yield f"data: {json.dumps({'tipo': 'fim'})}\n\n"

    return StreamingResponse(gerar(), media_type="text/event-stream")

@app.post("/upload")
async def upload_pdf(arquivo: UploadFile = File(...)):
    try:
        caminho = os.path.join(DOCS_DIR, arquivo.filename)
        conteudo = await arquivo.read()
        with open(caminho, "wb") as f:
            f.write(conteudo)

        doc = fitz.open(caminho)
        texto = "".join(p.get_text() for p in doc)

        if not texto.strip():
            return JSONResponse({"erro": "PDF sem texto — pode ser escaneado."}, status_code=400)

        tamanho, overlap, i = 500, 50, 0
        inicio = 0
        while inicio < len(texto):
            chunk = texto[inicio:inicio + tamanho]
            if chunk.strip():
                colecao.add(
                    documents=[chunk],
                    ids=[f"{arquivo.filename}_chunk_{i}"],
                    metadatas=[{"fonte": arquivo.filename}]
                )
            inicio += tamanho - overlap
            i += 1

        return {"mensagem": f"{arquivo.filename} indexado com {i} chunks."}
    except Exception as e:
        return JSONResponse({"erro": str(e)}, status_code=500)

@app.get("/documentos")
def listar_documentos():
    arquivos = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    return {"documentos": arquivos, "total_chunks": colecao.count()}

app.mount("/", StaticFiles(directory=".", html=True), name="static")