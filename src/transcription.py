"""
Módulo de transcrição de vídeos do YouTube.
 
Estratégia de DUAS CAMADAS (fallback):
  1. Tenta obter as legendas já existentes no YouTube (youtube-transcript-api).
     -> Rápido e leve. Funciona quando o vídeo tem legendas.
  2. Se não houver legendas, baixa o áudio (yt-dlp) e transcreve com Whisper.
     -> Lento mas funciona sempre.
 
Cada transcrição é guardada em JSON (texto + metadados) para usarmos no RAG.
"""
 
import json
import os
import re
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
 
import yt_dlp
from faster_whisper import WhisperModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
 
 
# --- Pastas de output ---
AUDIO_DIR = Path("data/audio")
TRANSCRIPT_DIR = Path("data/transcripts")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
 
# Idiomas que aceitamos, por ordem de preferência
PREFERRED_LANGUAGES = ["en", "en-US", "en-GB"]
 
# Modelo Whisper por defeito.
# Usamos a versão "english-only" (.en) porque é mais precisa em inglês
# do que a versão multilingue do mesmo tamanho.
DEFAULT_MODEL = "small.en"
 
# Cache do modelo Whisper (carregar uma só vez, não a cada vídeo)
_whisper_model = None
 
 
@dataclass
class VideoTranscript:
    """Estrutura que guarda uma transcrição e os seus metadados."""
    video_id: str
    title: str
    channel: str
    url: str
    source: str       # "captions" (legendas) ou "whisper"
    language: str
    text: str
    # Segmentos com timestamps: [{"start": float (segundos), "text": str}, ...]
    # Permite ligar cada pedaço da transcrição ao momento exato do vídeo.
    segments: list = field(default_factory=list)
 
    def save(self) -> Path:
        """Guarda a transcrição em JSON na pasta data/transcripts/."""
        path = TRANSCRIPT_DIR / f"{self.video_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        return path
 
 
def extract_video_id(url: str) -> str:
    """Extrai o ID do vídeo a partir de várias formas de URL do YouTube."""
    patterns = [
        r"(?:v=|/videos/|embed/|youtu\.be/|/v/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Se já for só o ID
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url):
        return url
    raise ValueError(f"Não consegui extrair o video_id de: {url}")
 
 
def get_metadata(url: str) -> dict:
    """Obtém metadados (título, canal) sem baixar o vídeo."""
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "video_id": info["id"],
        "title": info.get("title", ""),
        "channel": info.get("uploader", ""),
    }
 
 
def try_get_captions(video_id: str):
    """
    CAMADA 1: tenta obter as legendas existentes do YouTube.
    Retorna (texto, idioma) se existirem, ou None se não.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Procura uma legenda numa das línguas que preferimos
        transcript = transcript_list.find_transcript(PREFERRED_LANGUAGES)
        entries = transcript.fetch()
        text = " ".join(entry["text"] for entry in entries)
        # Segmentos com timestamps (as legendas já trazem o 'start')
        segments = [{"start": float(e["start"]), "text": e["text"]} for e in entries]
        return text, transcript.language_code, segments
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception as e:
        # Qualquer outro erro (ex: 403) -> tratamos como "sem legendas"
        print(f"      (legendas indisponíveis: {type(e).__name__})")
        return None
 
 
def get_whisper_model(model_size: str = DEFAULT_MODEL) -> WhisperModel:
    """Carrega (e faz cache) do modelo Whisper."""
    global _whisper_model
    if _whisper_model is None:
        print(f"      A carregar modelo Whisper '{model_size}'...")
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper_model
 
 
def transcribe_audio_bytes(audio_bytes: bytes, model_size: str = DEFAULT_MODEL) -> str:
    """
    Transcreve áudio gravado pelo utilizador (ex: do microfone na app).
    Recebe os bytes do áudio (WAV), guarda num ficheiro temporário e usa o Whisper.
    Reutiliza o mesmo modelo já em cache (eficiente).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
 
    try:
        model = get_whisper_model(model_size)
        segments, _ = model.transcribe(tmp_path, beam_size=5)
        text = " ".join(segment.text for segment in segments).strip()
    finally:
        os.remove(tmp_path)  # limpar o ficheiro temporário
 
    return text
 
 
def transcribe_with_whisper(url: str, video_id: str, model_size: str = DEFAULT_MODEL):
    """
    CAMADA 2: baixa o áudio e transcreve com Whisper.
    Retorna (texto, idioma, segmentos).
    """
    audio_path = AUDIO_DIR / f"{video_id}.mp3"
 
    # 1. Baixar o áudio — só se ainda não estiver em cache local
    if audio_path.exists():
        print(f"      (áudio já em cache, a reutilizar)")
    else:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
 
    # 2. Transcrever — guardamos cada segmento com o seu timestamp de início
    model = get_whisper_model(model_size)
    raw_segments, info = model.transcribe(str(audio_path), beam_size=5)
 
    seg_list = []
    parts = []
    for seg in raw_segments:
        seg_list.append({"start": float(seg.start), "text": seg.text.strip()})
        parts.append(seg.text)
    text = " ".join(parts).strip()
    return text, info.language, seg_list
 
 
def transcribe_video(
    url: str,
    prefer_captions: bool = True,
    model_size: str = DEFAULT_MODEL,
) -> VideoTranscript:
    """
    Função PRINCIPAL.
    Tenta legendas primeiro (rápido); se não houver, usa Whisper (lento).
    """
    meta = get_metadata(url)
    video_id = meta["video_id"]
    print(f"-> {meta['title']} ({video_id})")
 
    text, language, source, segments = None, None, None, []
 
    # CAMADA 1: legendas existentes
    if prefer_captions:
        result = try_get_captions(video_id)
        if result:
            text, language, segments = result
            source = "captions"
            print(f"   [legendas] obtidas em '{language}' ({len(text)} chars)")
 
    # CAMADA 2: Whisper (fallback)
    if text is None:
        print(f"   [whisper] sem legendas, a transcrever com Whisper...")
        text, language, segments = transcribe_with_whisper(url, video_id, model_size)
        source = "whisper"
        print(f"   [whisper] transcrito em '{language}' ({len(text)} chars)")
 
    transcript = VideoTranscript(
        video_id=video_id,
        title=meta["title"],
        channel=meta["channel"],
        url=url,
        source=source,
        language=language,
        text=text,
        segments=segments,
    )
    path = transcript.save()
    print(f"   guardado em {path}")
    return transcript
 
 
# Permite testar o módulo diretamente: python src/transcription.py "URL"
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python src/transcription.py URL_DO_VIDEO")
        sys.exit(1)
    transcribe_video(sys.argv[1])
