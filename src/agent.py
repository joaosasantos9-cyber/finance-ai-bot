"""
Agente LangChain com TOOLS e MEMÓRIA.
 
Diferença para a QA chain (Dia 3):
- A chain faz sempre os mesmos passos (fixa).
- O AGENTE decide sozinho que ferramenta usar e quando (raciocina).
 
Tools disponíveis:
  1. search_finance_videos  -> procura informação nas transcrições (RAG)
  2. list_available_videos  -> lista os vídeos/temas disponíveis
 
Memória: o agente lembra-se da conversa (perguntas de seguimento funcionam).
"""
 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
 
from src.rag import get_vectorstore, load_documents
 
load_dotenv()
 
LLM_MODEL = "gpt-4o-mini"
TOP_K = 4
 
 
# =========================================================================
# TOOLS — cada função decorada com @tool vira uma ferramenta do agente.
# A docstring é CRUCIAL: é por ela que o agente percebe quando usar a tool.
# =========================================================================
 
@tool
def search_finance_videos(query: str) -> str:
    """Search the personal finance video transcripts for information relevant
    to the query. Use this for ANY question about personal finance topics such as
    saving, investing, debt, budgeting, emergency funds, stocks, or real estate."""
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the videos."
    blocks = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        blocks.append(f"[From video: {title}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)
 
 
@tool
def list_available_videos() -> str:
    """List all the video titles and channels available in the knowledge base.
    Use this when the user asks what videos, topics, or content are available."""
    docs = load_documents()
    lines = [f"- {d.metadata['title']} ({d.metadata['channel']})" for d in docs]
    return "Available videos in the knowledge base:\n" + "\n".join(lines)
 
 
TOOLS = [search_finance_videos, list_available_videos]
 
 
# =========================================================================
# AGENTE
# =========================================================================
 
SYSTEM_PROMPT = """You are a helpful personal finance assistant. You answer questions \
based on a collection of YouTube videos about personal finance.
 
Guidelines:
- For finance questions, ALWAYS use the `search_finance_videos` tool first to find \
relevant information before answering.
- Use `list_available_videos` when the user asks what topics or videos are available.
- Answer based ONLY on what the tools return. If the information is not there, say you \
don't have it in the videos. Never invent information.
- Use the conversation history to understand follow-up questions.
- Be clear and concise.
- End any finance answer with: "This is educational content, not financial advice."
"""
 
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
 
 
def build_agent() -> AgentExecutor:
    """Constrói o agente com as tools e o prompt."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True)
 
 
# --- Memória (uma história de conversa por sessão) ---
_session_store: dict[str, ChatMessageHistory] = {}
 
 
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]
 
 
def build_agent_with_memory() -> RunnableWithMessageHistory:
    """Envolve o agente com memória conversacional."""
    return RunnableWithMessageHistory(
        build_agent(),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
 
 
# =========================================================================
# Teste interativo: python -m src.agent
# =========================================================================
if __name__ == "__main__":
    print("Agente de finanças pessoais (escreve 'sair' para terminar)\n")
    agent = build_agent_with_memory()
    config = {"configurable": {"session_id": "test-session"}}
 
    while True:
        question = input("\nTu: ").strip()
        if question.lower() in {"sair", "exit", "quit"}:
            break
        if not question:
            continue
        result = agent.invoke({"input": question}, config=config)
        print(f"\nBot: {result['output']}")