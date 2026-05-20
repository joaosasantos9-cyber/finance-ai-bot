"""
Teste inicial do pipeline de transcrição.
- Baixa o áudio de um vídeo do YouTube com yt-dlp
- Transcreve o áudio com faster-whisper (modelo local)
- Imprime e guarda a transcrição
 
Uso:
    python test_transcription.py "URL_DO_VIDEO"
"""
 
import sys
from pathlib import Path
import yt_dlp
from faster_whisper import WhisperModel
 
 
# Pastas de output
AUDIO_DIR = Path("data/audio")
TRANSCRIPT_DIR = Path("data/transcripts")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
 
 
def download_audio(url: str):
    """Baixa o áudio do vídeo do YouTube em formato mp3."""
    print(f"[1/3] A baixar áudio de: {url}")
 
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": False,
        "no_warnings": True,
    }
 
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]
        title = info.get("title", video_id)
 
    audio_path = AUDIO_DIR / f"{video_id}.mp3"
    print(f"      Áudio guardado em: {audio_path}")
    print(f"      Título: {title}")
    return audio_path, video_id, title
 
 
def transcribe(audio_path: Path, model_size: str = "base") -> str:
    """Transcreve o áudio com faster-whisper (modelo local)."""
    print(f"[2/3] A carregar modelo faster-whisper '{model_size}' (primeira vez demora)...")
    # device="cpu" e compute_type="int8" funcionam em qualquer máquina (Mac incluído)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
 
    print(f"[3/3] A transcrever (pode demorar alguns minutos)...")
    segments, info = model.transcribe(str(audio_path), beam_size=5)
 
    print(f"      Idioma detetado: {info.language} (probabilidade: {info.language_probability:.2f})")
 
    # Juntar todos os segmentos num único texto
    full_text = ""
    for segment in segments:
        full_text += segment.text + " "
 
    return full_text.strip()
 
 
def main():
    if len(sys.argv) < 2:
        print("Uso: python test_transcription.py URL_DO_VIDEO")
        sys.exit(1)
 
    url = sys.argv[1]
 
    audio_path, video_id, title = download_audio(url)
    transcript = transcribe(audio_path, model_size="base")
 
    # Guardar transcrição
    transcript_path = TRANSCRIPT_DIR / f"{video_id}.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n# URL: {url}\n\n{transcript}")
 
    print("\n" + "=" * 60)
    print("TRANSCRIÇÃO (primeiros 500 caracteres):")
    print("=" * 60)
    print(transcript[:500] + "...")
    print(f"\nTranscrição completa em: {transcript_path}")
 
 
if __name__ == "__main__":
    main()
 