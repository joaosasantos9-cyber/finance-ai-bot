💰 Finance Video Q&A Bot
A multimodal AI chatbot that answers questions about personal-finance YouTube videos.
Ask by text or voice, and get an answer grounded in real videos — with a link to
the exact moment the topic is discussed.
🔗 Live app: https://finance-ai-bot-finalproject.streamlit.app/

Final project — Ironhack AI Engineering Bootcamp.


What it does

Transcribes a collection of personal-finance YouTube videos.
Indexes the transcripts in a vector database for semantic search.
A LangChain agent answers user questions using RAG (Retrieval-Augmented Generation),
citing the source video and timestamp.
Supports voice input (speech-to-text) and voice output (text-to-speech).
Is evaluated with LangSmith and deployed as a public web app.


Setup
bash
# 1. Virtual environment (Python 3.11 recommended)
python3 -m venv venv
source venv/bin/activate            # macOS/Linux

# 2. Dependencies
pip install -r requirements.txt -r requirements-ingest.txt

# 3. ffmpeg (required by Whisper)
brew install ffmpeg                 # macOS

# 4. Environment variables
cp .env.example .env                # then add your keys
.env keys:
OPENAI_API_KEY=sk-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2-...
LANGCHAIN_PROJECT=finance-ai-bot
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

Usage
bash
# 1. Transcribe the videos listed in videos.txt
python3 ingest.py

# 2. Build the vector database
python3 build_index.py

# 3. Run the web app
streamlit run app.py

# 4. (Optional) Run the LangSmith evaluation
python3 evaluate_bot.py

How it works
Retrieval-Augmented Generation (RAG)
Transcripts are split into ~1000-character chunks (150-char overlap) built from the
Whisper segments, so each chunk keeps its start timestamp. Chunks are embedded with
OpenAI text-embedding-3-small and stored in ChromaDB with metadata (video id, title,
channel, url, start_time). At query time, the most semantically similar chunks are
retrieved and passed to the LLM as context.
The agent (LangChain)
A tool-calling agent (create_tool_calling_agent + AgentExecutor) decides which tool
to use for each question:

search_finance_videos — semantic search over ChromaDB (RAG)
list_available_videos — lists the videos in the knowledge base

Memory: RunnableWithMessageHistory keeps per-session conversation history, so
follow-up questions retain context.
Prompt / safety: the system prompt forces the agent to answer only from retrieved
context (anti-hallucination), keep answers concise, and always add the disclaimer
"This is educational content, not financial advice."
Voice

Input: the user records a question; OpenAI Whisper API transcribes it to text.
Output: OpenAI TTS (tts-1) reads the answer aloud (optional).


Evaluation
Evaluated with LangSmith using an LLM-as-a-judge over a 14-question test set, with two
metrics:
MetricScoreMeaningGroundedness1.00The answer is supported by the retrieved context (no hallucinations)Correctness0.79The answer matches a reference answer
The correctness gap is mostly coverage (the bot correctly declines on topics not in the
corpus), not wrong facts — confirmed by the perfect groundedness score. LangSmith also
tracks latency and token cost per run.

Limitations & future work

Corpus limited to 13 videos — a larger corpus would broaden coverage.
English only — multi-language support is a natural extension.
Inline citations within the answer text (not only in a sources panel).
Richer evaluation set with more edge cases.


Disclaimer
This project is an educational/technical demonstration. The bot's answers do not
constitute financial advice. For financial decisions, consult a certified professional.