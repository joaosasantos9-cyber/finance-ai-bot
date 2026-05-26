
"""
Interface web (Streamlit) do Finance AI Bot.
 
Funcionalidades:
- Chat por texto com o agente (com memória)
- Input por voz (microfone -> Whisper)
- Output por voz opcional (OpenAI TTS)
- Painel lateral com os vídeos disponíveis
 
Correr com:  streamlit run app.py
"""
 
import io
import uuid
 
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
 
from src.agent import build_agent_with_memory
from src.rag import get_vectorstore, load_documents
 
load_dotenv()
 
st.set_page_config(page_title="Finance Video Q&A Bot", page_icon="💰")
 
 
# --- Recursos pesados: construir uma só vez (cache) ---
@st.cache_resource
def get_agent():
    return build_agent_with_memory()
 
 
@st.cache_data
def get_video_list():
    docs = load_documents()
    return [(d.metadata["title"], d.metadata["channel"], d.metadata["url"]) for d in docs]
 
 
def transcribe_audio_openai(audio_bytes: bytes) -> str:
    """
    Transcreve o áudio gravado pelo utilizador usando a API Whisper da OpenAI.
    (Leve para a cloud — não precisa de modelos locais.)
    """
    client = OpenAI()
    buffer = io.BytesIO(audio_bytes)
    buffer.name = "audio.wav"  # a API precisa de um nome para inferir o formato
    transcript = client.audio.transcriptions.create(model="whisper-1", file=buffer)
    return transcript.text
 
 
def text_to_speech(text: str) -> bytes:
    """Converte texto em áudio (mp3) usando a OpenAI TTS."""
    client = OpenAI()
    # with_streaming_response + read() devolve sempre os bytes mp3 corretos
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="mp3",
    ) as response:
        return response.read()
 
 
def format_timestamp(seconds: float) -> str:
    """Converte segundos em mm:ss (ex: 225 -> '3:45')."""
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"
 
 
def timestamped_link(url: str, start: float) -> str:
    """Cria um link do YouTube que abre no momento certo (ex: ...&t=225s)."""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={int(start)}s"
 
 
def get_related_sources(question: str, max_videos: int = 4):
    """
    Vídeos mais relevantes para a pergunta, com link para o momento certo.
    Mostra cada vídeo SÓ UMA VEZ (no seu chunk mais relevante), para dar
    variedade em vez de repetir o mesmo vídeo várias vezes.
    """
    # Buscamos mais candidatos (k alto) para conseguir vários vídeos distintos
    docs = get_vectorstore().similarity_search(question, k=12)
    sources, seen = [], set()
    for doc in docs:
        vid = doc.metadata.get("video_id")
        start = doc.metadata.get("start_time", 0.0)
        # Cada vídeo aparece só uma vez (a 1ª ocorrência é a mais relevante)
        if vid not in seen:
            seen.add(vid)
            sources.append({
                "title": doc.metadata.get("title"),
                "channel": doc.metadata.get("channel", ""),
                "link": timestamped_link(doc.metadata.get("url", ""), start),
                "time_label": format_timestamp(start),
                # Miniatura do YouTube a partir do ID do vídeo
                "thumbnail": f"https://img.youtube.com/vi/{vid}/mqdefault.jpg",
            })
        if len(sources) >= max_videos:
            break
    return sources
 
 
# --- Estado da sessão ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
 
agent = get_agent()
 
# Avatares para as mensagens
USER_AVATAR = "🧑"
BOT_AVATAR = "💰"
 
 
# --- Barra lateral ---
with st.sidebar:
    st.header("⚙️ Settings")
    tts_enabled = st.toggle("🔊 Spoken answer", value=False,
                            help="Hear the bot's answer as audio (uses OpenAI TTS).")
    st.divider()
    st.subheader("📺 Videos in the knowledge base")
    for title, channel, url in get_video_list():
        st.markdown(f"- [{title}]({url})  \n  _{channel}_")
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())  # nova sessão = nova memória
        st.rerun()
 
 
# --- Cabeçalho ---
st.title("💰 Finance Video Q&A Bot")
st.markdown(
    "#### Your personal finance assistant, powered by YouTube videos 🎥"
)
st.caption("Ask by text or voice — answers come from real videos, "
           "with a link to the exact moment.")
st.divider()
 
 
# --- Ecrã de boas-vindas com perguntas de exemplo (só quando o chat está vazio) ---
EXAMPLE_QUESTIONS = [
    "How do I start investing as a beginner?",
    "What's the best way to build an emergency fund?",
    "Should I rent or buy a home?",
    "How do dividend stocks work?",
]
 
if not st.session_state.messages and not st.session_state.pending_question:
    st.markdown("##### 👋 Not sure where to start? Try one of these:")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()
 
 
# --- Histórico de mensagens ---
for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
 
 
# --- Input por voz ---
st.write("🎤 Or ask by voice:")
# just_once=False: o gravador mantém-se ativo para várias gravações seguidas.
# Controlamos manualmente o que já foi processado através do 'id' (contador).
audio = mic_recorder(start_prompt="Record", stop_prompt="Stop",
                     just_once=False, use_container_width=True, key="recorder")
 
user_input = None
 
# Só transcreve se for uma gravação NOVA (id diferente do último processado)
if audio and audio.get("id") != st.session_state.last_audio_id:
    st.session_state.last_audio_id = audio["id"]
    with st.spinner("Transcribing your question..."):
        user_input = transcribe_audio_openai(audio["bytes"])
 
# Input por texto
typed = st.chat_input("Type your personal finance question...")
if typed:
    user_input = typed
 
# Pergunta vinda de um botão de exemplo
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None
 
 
def search_query_used(result) -> str | None:
    """
    Verifica se o agente usou a tool de pesquisa e devolve a query que usou.
    Devolve None se a pesquisa não foi usada (ex: saudação ou 'não sei').
    """
    for step in result.get("intermediate_steps", []):
        action = step[0]
        if getattr(action, "tool", None) == "search_finance_videos":
            return action.tool_input.get("query", "")
    return None
 
 
# --- Processar a pergunta ---
if user_input:
    # Mostrar mensagem do utilizador
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)
 
    # Resposta do agente
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            with st.spinner("Thinking..."):
                result = agent.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )
                answer = result["output"]
            st.markdown(answer)
 
            # Fontes — só se o agente realmente pesquisou nos vídeos.
            # Usamos a query reformulada pelo agente (mais precisa).
            query = search_query_used(result)
            if query:
                sources = get_related_sources(query)
                if sources:
                    st.markdown("**📺 Related videos (jump to the exact moment)**")
                    for s in sources:
                        col_img, col_txt = st.columns([1, 3])
                        with col_img:
                            st.image(s["thumbnail"], width=140)
                        with col_txt:
                            st.markdown(f"**{s['title']}**")
                            if s["channel"]:
                                st.caption(s["channel"])
                            st.markdown(f"[▶️ Watch at {s['time_label']}]({s['link']})")
                        st.divider()
 
            # Resposta por voz (opcional) — falha em silêncio se a TTS der erro
            if tts_enabled:
                try:
                    with st.spinner("Generating audio..."):
                        audio_bytes = text_to_speech(answer)
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                except Exception:
                    st.warning("Couldn't generate audio this time.")
 
            st.session_state.messages.append({"role": "assistant", "content": answer})
 
        except Exception:
            st.error("⚠️ Something went wrong while processing your question. "
                     "Please check your connection and OpenAI key, then try again.")