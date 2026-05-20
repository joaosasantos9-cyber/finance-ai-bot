# Finance AI Bot — Projeto Final Ironhack

ChatBot multimodal com RAG que responde a perguntas sobre vídeos de YouTube de finanças pessoais.

## Stack

- **Transcrição:** yt-dlp + OpenAI Whisper
- **RAG:** LangChain + ChromaDB + OpenAI embeddings
- **Agente:** LangChain agent com tools e memória
- **Interface:** Streamlit (texto e voz)
- **Avaliação:** LangSmith
- **Deploy:** Streamlit Community Cloud

## Setup

```bash
# 1. Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate            # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Instalar ffmpeg (necessário para Whisper)
brew install ffmpeg                # macOS
# sudo apt install ffmpeg          # Linux

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Editar .env e meter a tua OPENAI_API_KEY
```

## Estrutura

```
finance-ai-bot/
├── data/
│   ├── audio/             # áudios baixados (gitignored)
│   └── transcripts/       # transcrições (gitignored)
├── src/                   # código fonte
├── notebooks/             # exploração
├── tests/                 # testes
├── app.py                 # Streamlit app
├── requirements.txt
├── .env.example
└── README.md
```

## Disclaimer

Este projeto é uma demonstração técnica. As respostas do bot **não constituem aconselhamento financeiro**. Para decisões financeiras, consulte um profissional certificado.
