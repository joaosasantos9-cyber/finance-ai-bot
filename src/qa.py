"""
QA chain — junta Retrieval + Generation (o RAG completo).
 
Fluxo:
  pergunta -> procurar chunks relevantes (retrieval)
           -> dar chunks + pergunta ao GPT (generation)
           -> resposta fundamentada, com fontes
"""
 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
 
from src.rag import get_vectorstore
 
load_dotenv()
 
# Modelo de chat: barato, rápido e capaz. Bom para poupar créditos OpenAI.
LLM_MODEL = "gpt-4o-mini"
 
# Quantos chunks vamos buscar para dar de contexto ao modelo
TOP_K = 4
 
 
# O "system prompt" define o comportamento do assistente.
# Damos-lhe regras claras: usar SÓ o contexto, citar fontes, e o disclaimer.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about \
personal finance based ONLY on the provided video transcripts.
 
Rules:
- Answer using ONLY the information in the CONTEXT below. Do not use outside knowledge.
- If the context does not contain the answer, say you don't have that information \
in the videos. Do not make things up.
- Be clear and concise.
- At the end, add a short note: "This is educational content, not financial advice."
 
CONTEXT:
{context}
"""
 
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])
 
 
def format_docs(docs) -> str:
    """Junta os chunks num único bloco de texto, identificando a fonte de cada um."""
    blocks = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        blocks.append(f"[From video: {title}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)
 
 
def build_qa_chain():
    """Constrói a chain de RAG usando LangChain (LCEL)."""
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
 
    # LCEL: a pergunta passa por dois caminhos em paralelo —
    #  - "context": vai ao retriever buscar chunks e formata-os
    #  - "question": passa a pergunta tal como está
    # Depois entra no prompt, no LLM, e o output é convertido para texto.
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
 
 
def answer_question(question: str):
    """Responde a uma pergunta e devolve (resposta, lista de fontes)."""
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)
 
    chain = build_qa_chain()
    answer = chain.invoke(question)
 
    # Fontes únicas (título + url) para mostrar ao utilizador
    sources = []
    seen = set()
    for doc in docs:
        vid = doc.metadata.get("video_id")
        if vid not in seen:
            seen.add(vid)
            sources.append({
                "title": doc.metadata.get("title"),
                "channel": doc.metadata.get("channel"),
                "url": doc.metadata.get("url"),
            })
    return answer, sources
 
 
# Teste: python src/qa.py "a tua pergunta"
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Uso: python src/qa.py "a tua pergunta"')
        sys.exit(1)
 
    question = sys.argv[1]
    answer, sources = answer_question(question)
 
    print(f"\nPergunta: {question}\n")
    print("=" * 70)
    print("\nRESPOSTA:\n")
    print(answer)
    print("\n" + "=" * 70)
    print("\nFONTES:")
    for s in sources:
        print(f"  - {s['title']} ({s['channel']})")
        print(f"    {s['url']}")