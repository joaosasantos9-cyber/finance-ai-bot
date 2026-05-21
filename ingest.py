
"""
Script de ingestão em lote.
 
Lê uma lista de URLs do ficheiro 'videos.txt' (um URL por linha)
e transcreve cada vídeo, guardando o resultado em data/transcripts/.
 
Uso:
    python ingest.py
"""
 
import time
from pathlib import Path
 
from src.transcription import transcribe_video, extract_video_id, TRANSCRIPT_DIR
 
 
VIDEOS_FILE = Path("videos.txt")
 
 
def load_urls() -> list[str]:
    """Lê os URLs do ficheiro videos.txt (ignora linhas vazias e comentários)."""
    if not VIDEOS_FILE.exists():
        raise FileNotFoundError(
            "Cria um ficheiro 'videos.txt' com um URL do YouTube por linha."
        )
    urls = []
    for line in VIDEOS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls
 
 
def already_done(video_id: str) -> bool:
    """Verifica se já transcrevemos este vídeo (para não repetir)."""
    return (TRANSCRIPT_DIR / f"{video_id}.json").exists()
 
 
def main():
    urls = load_urls()
    print(f"Encontrados {len(urls)} vídeos para processar.\n")
 
    for i, url in enumerate(urls, start=1):
        print(f"=== Vídeo {i}/{len(urls)} ===")
        try:
            # Salta vídeos já transcritos (evita repetir trabalho)
            video_id = extract_video_id(url)
            if already_done(video_id):
                print(f"   já transcrito ({video_id}), a saltar.")
                print()
                continue
            transcribe_video(url)
        except Exception as e:
            print(f"   ERRO ao processar {url}: {e}")
        # Pausa entre vídeos para não "martelar" o YouTube (conselho do professor!)
        if i < len(urls):
            time.sleep(3)
        print()
 
    print("Ingestão concluída.")
    print(f"Transcrições em: {TRANSCRIPT_DIR}/")
 
 
if __name__ == "__main__":
    main()