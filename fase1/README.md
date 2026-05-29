# AI Projects — Fase 1

Primeiros experimentos com LLMs rodando localmente usando Ollama.

## Projetos

### hello_llm.py
Primeira chamada a um modelo local via Python usando a lib `requests`.

### chat.py
Chat interativo no terminal com histórico de conversa e system prompt customizado.
Demonstra na prática como funciona o context window de LLMs.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install ollama requests
```

## Modelo utilizado
- llama3.2 via Ollama (100% local, sem API key)
