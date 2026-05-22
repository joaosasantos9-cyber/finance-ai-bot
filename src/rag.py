"""
RAG — indexação e consulta das transcrições.
 
Fase 1 (indexação): ler JSONs -> dividir em chunks -> embeddings -> ChromaDB
Fase 2 (consulta):   pergunta -> embedding -> procurar chunks parecidos
 
Usamos LangChain (requisito do projeto) para orquestrar tudo.
"""
 
import os
import logging
 
# Desliga a telemetria anónima do ChromaDB (evita mensagens de erro no output).
# Usamos dois métodos para garantir: variável de ambiente + silenciar o logger.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
 
import json
from pathlib import Path
 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
 
# Carrega as variáveis do .env (OPENAI_API_KEY)
load_dotenv()
 
# --- Configurações ---
TRANSCRIPT_DIR = Path("data/transcripts")
CHROMA_DIR = "chroma_db"                     # onde a base vetorial é guardada
COLLECTION_NAME = "finance_videos"
EMBEDDING_MODEL = "text-embedding-3-small"   # barato e muito bom
 
# Tamanho dos chunks (em caracteres) e sobreposição entre eles
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
 
 
def load_documents() -> list[Document]:
    """
    Lê cada transcrição (.json) e cria um Document do LangChain.
    Cada Document tem o texto + metadados (para sabermos a origem depois).
    """
    docs = []
    for path in sorted(TRANSCRIPT_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        doc = Document(
            page_content=data["text"],
            metadata={
                "video_id": data["video_id"],
                "title": data["title"],
                "channel": data["channel"],
                "url": data["url"],
                "source": data["source"],
            },
        )
        docs.append(doc)
    return docs
 
 
def split_documents(docs: list[Document]) -> list[Document]:
    """
    Divide cada Document em chunks mais pequenos.
    RecursiveCharacterTextSplitter tenta cortar em fronteiras naturais
    (parágrafos, frases) antes de cortar a meio de uma palavra.
    Os metadados (título, url...) são copiados para cada chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
 
 
def get_embeddings() -> OpenAIEmbeddings:
    """Modelo de embeddings da OpenAI (transforma texto em vetores)."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
 
 
def build_vectorstore():
    """
    FASE 1 — constrói a base vetorial a partir das transcrições.
    Corre isto sempre que adicionares ou mudares vídeos.
    """
    docs = load_documents()
    if not docs:
        raise RuntimeError("Sem transcrições em data/transcripts/. Corre o ingest.py primeiro.")
 
    chunks = split_documents(docs)
 
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore, len(docs), len(chunks)
 
 
def get_vectorstore() -> Chroma:
    """
    FASE 2 — abre a base vetorial já construída (sem reindexar).
    Usado pela app e pelo agente para fazer perguntas.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
 
 
# Teste rápido: python src/rag.py "a tua pergunta aqui"
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 2:
        print('Uso: python src/rag.py "a tua pergunta"')
        sys.exit(1)
 
    question = sys.argv[1]
    vs = get_vectorstore()
 
    # Procura os 4 chunks mais relevantes (com a "distância" de similaridade)
    results = vs.similarity_search_with_score(question, k=4)
 
    print(f"\nPergunta: {question}\n")
    print("=" * 70)
    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n[{i}] {doc.metadata['title']}  (score: {score:.3f})")
        print(f"    canal: {doc.metadata['channel']}")
        print(f"    {doc.page_content[:250].strip()}...")