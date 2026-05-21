"""
Constrói (ou reconstrói) a base vetorial ChromaDB a partir das transcrições.
 
Uso:
    python build_index.py
"""
 
from src.rag import build_vectorstore, CHROMA_DIR
 
 
def main():
    print("A construir a base vetorial...")
    vectorstore, n_docs, n_chunks = build_vectorstore()
    print(f"\nIndexação concluída:")
    print(f"  - {n_docs} vídeos")
    print(f"  - {n_chunks} chunks")
    print(f"  - guardado em: {CHROMA_DIR}/")
 
 
if __name__ == "__main__":
    main()
 