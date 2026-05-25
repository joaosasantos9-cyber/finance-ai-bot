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
 
 
def get_related_sources(question: str, k: int = 3):
    """Vídeos mais relevantes para a pergunta (para mostrar como fontes)."""
    docs = get_vectorstore().similarity_search(question, k=k)
    sources, seen = [], set()
    for doc in docs:
        vid = doc.metadata.get("video_id")
        if vid not in seen:
            seen.add(vid)
            sources.append((doc.metadata.get("title"), doc.metadata.get("url")))
    return sources
 
 
# --- Estado da sessão ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
 
agent = get_agent()
 
 
# --- Barra lateral ---
with st.sidebar:
    st.header("⚙️ Definições")
    tts_enabled = st.toggle("🔊 Resposta por voz", value=False,
                            help="Ouvir a resposta do bot em áudio (usa OpenAI TTS).")
    st.divider()
    st.subheader("📺 Vídeos na base de conhecimento")
    for title, channel, url in get_video_list():
        st.markdown(f"- [{title}]({url})  \n  _{channel}_")
    st.divider()
    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())  # nova sessão = nova memória
        st.rerun()
 
 
# --- Cabeçalho ---
st.title("💰 Finance Video Q&A Bot")
st.caption("Faz perguntas sobre finanças pessoais — por texto ou voz. "
           "As respostas baseiam-se em vídeos do YouTube.")
 
 
# --- Histórico de mensagens ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
 
# --- Input por voz ---
st.write("🎤 Ou faz a pergunta por voz:")
# just_once=False: o gravador mantém-se ativo para várias gravações seguidas.
# Controlamos manualmente o que já foi processado através do 'id' (contador).
audio = mic_recorder(start_prompt="Gravar", stop_prompt="Parar",
                     just_once=False, use_container_width=True, key="recorder")
 
user_input = None
 
# Só transcreve se for uma gravação NOVA (id diferente do último processado)
if audio and audio.get("id") != st.session_state.last_audio_id:
    st.session_state.last_audio_id = audio["id"]
    with st.spinner("A transcrever a tua pergunta..."):
        user_input = transcribe_audio_openai(audio["bytes"])
 
# Input por texto
typed = st.chat_input("Escreve a tua pergunta sobre finanças...")
if typed:
    user_input = typed
 
 
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
    with st.chat_message("user"):
        st.markdown(user_input)
 
    # Resposta do agente
    with st.chat_message("assistant"):
        try:
            with st.spinner("A pensar..."):
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
                    with st.expander("📺 Vídeos relacionados"):
                        for title, url in sources:
                            st.markdown(f"- [{title}]({url})")
 
            # Resposta por voz (opcional) — falha em silêncio se a TTS der erro
            if tts_enabled:
                try:
                    with st.spinner("A gerar áudio..."):
                        audio_bytes = text_to_speech(answer)
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                except Exception:
                    st.warning("Não foi possível gerar o áudio desta vez.")
 
            st.session_state.messages.append({"role": "assistant", "content": answer})
 
        except Exception:
            st.error("⚠️ Ocorreu um erro ao processar a tua pergunta. "
                     "Verifica a ligação e a chave da OpenAI, e tenta novamente.")