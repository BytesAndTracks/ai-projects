import os
import fitz
import ollama
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = "./docs"
MODELO = "llama3.2"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

def carregar_pdfs(pasta):
    documentos = []
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(pasta, arquivo)
            doc = fitz.open(caminho)
            texto = ""
            for pagina in doc:
                texto += pagina.get_text()
            documentos.append({
                "nome": arquivo,
                "texto": texto
            })
            print(f"  Carregado: {arquivo} ({len(texto)} chars)")
    return documentos

def criar_chunks(texto, tamanho=500, overlap=50):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho
        chunks.append(texto[inicio:fim])
        inicio += tamanho - overlap
    return chunks

def indexar_documentos(colecao, documentos):
    print("\nIndexando documentos...")
    for doc in documentos:
        chunks = criar_chunks(doc["texto"])
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                colecao.add(
                    documents=[chunk],
                    ids=[f"{doc['nome']}_chunk_{i}"],
                    metadatas=[{"fonte": doc["nome"]}]
                )
    print("  Indexação concluída.")

def perguntar(colecao, pergunta):
    resultados = colecao.query(
        query_texts=[pergunta],
        n_results=3
    )

    chunks_relevantes = resultados["documents"][0]
    fontes = [m["fonte"] for m in resultados["metadatas"][0]]
    distancias = resultados["distances"][0]

    contexto = "\n\n".join(chunks_relevantes)

    prompt = f"""Use apenas as informações abaixo para responder a pergunta.
Se a resposta não estiver nas informações, diga que não encontrou nos documentos.

INFORMAÇÕES:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA:"""

    resposta = ollama.chat(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"\nResposta: {resposta['message']['content']}")
    print("\n--- Trechos consultados ---")
    for i, (chunk, fonte, dist) in enumerate(zip(chunks_relevantes, fontes, distancias)):
        print(f"\n[{i+1}] {fonte} (similaridade: {1-dist:.0%})")
        print(f"    \"{chunk[:150].strip()}...\"")

def main():
    print("Carregando PDFs...")
    documentos = carregar_pdfs(DOCS_DIR)

    cliente = chromadb.Client()
    colecao = cliente.get_or_create_collection(
        "meus_docs",
        embedding_function=embedding_fn
    )

    indexar_documentos(colecao, documentos)

    print("\nRAG pronto! Digite sua pergunta (ou 'sair' para encerrar)\n")
    while True:
        pergunta = input("Pergunta: ")
        if pergunta.lower() == "sair":
            break
        perguntar(colecao, pergunta)

if __name__ == "__main__":
    main()