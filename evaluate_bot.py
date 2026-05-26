"""
Avaliação do bot com LangSmith (versão expandida).
 
Dois avaliadores (LLM-as-a-judge):
  - correctness:  a resposta é consistente com a resposta de referência?
  - groundedness: a resposta está fundamentada no contexto recuperado
                  (ou admite que não sabe)? -> mede alucinação.
 
Uso:
    python evaluate_bot.py
"""
 
from dotenv import load_dotenv
load_dotenv()  # carregar chaves ANTES de importar o resto
 
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
 
from src.qa import answer_question
from src.rag import get_vectorstore
 
client = Client()
DATASET_NAME = "finance-bot-eval-v2"
TOP_K = 4
 
 
# --- 1. Conjunto de teste (perguntas + respostas de referência) ---
# Nota: podes ajustar/acrescentar perguntas para refletir os teus vídeos específicos.
EXAMPLES = [
    {"question": "What is a good starting goal for an emergency fund?",
     "reference": "A common starting goal is around $1,000."},
    {"question": "Name a few ways to build an emergency fund.",
     "reference": "Set a specific goal, cut unnecessary expenses, automate savings, "
                  "pick up a side hustle, and use windfalls like tax refunds."},
    {"question": "Why might someone keep an emergency fund in a high-liquidity account?",
     "reference": "So the money is safe and can be accessed quickly when an emergency happens."},
    {"question": "What is an advantage of index funds or ETFs for beginners?",
     "reference": "Low fees and broad diversification; most active managers fail to beat "
                  "the market over time."},
    {"question": "Why is diversification important when investing?",
     "reference": "It spreads risk across many assets so a single loss has less impact."},
    {"question": "What does it mean to invest consistently over the long term?",
     "reference": "Investing regularly and holding for the long run rather than trying to time the market."},
    {"question": "What is one consideration when deciding to rent versus buy a home?",
     "reference": "It depends on how long you'll stay, the costs of ownership, and what you "
                  "could earn investing the difference."},
    {"question": "What is a downside of personal loans?",
     "reference": "They can carry high interest rates and fees and add to your debt burden."},
    {"question": "What can dividend stocks provide to an investor?",
     "reference": "A stream of passive income through regular dividend payments."},
    {"question": "Why do some investors buy gold?",
     "reference": "As a hedge against inflation or currency debasement and economic uncertainty."},
    {"question": "What is a simple first step to start budgeting?",
     "reference": "Track your income and expenses and set a plan for where your money goes."},
    {"question": "Should beginners try to pick individual winning stocks?",
     "reference": "Generally no; broad, diversified, low-cost funds tend to work better for beginners."},
    {"question": "Is it a good idea to put all your savings into one cryptocurrency?",
     "reference": "No; that is very risky and not a substitute for a diversified plan or emergency fund."},
    # Pergunta fora do âmbito -> testa anti-alucinação
    {"question": "What is the capital of France?",
     "reference": "The bot should say it does not have this information in the videos."},
]
 
 
def ensure_dataset():
    """Cria o dataset no LangSmith (ou reutiliza se já existir)."""
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' já existe — a reutilizar.")
        return dataset
    except Exception:
        print(f"A criar dataset '{DATASET_NAME}' com {len(EXAMPLES)} exemplos...")
        dataset = client.create_dataset(dataset_name=DATASET_NAME)
        client.create_examples(
            inputs=[{"question": e["question"]} for e in EXAMPLES],
            outputs=[{"reference": e["reference"]} for e in EXAMPLES],
            dataset_id=dataset.id,
        )
        return dataset
 
 
# --- 2. Função-alvo: corre o bot e devolve resposta + contexto usado ---
def run_bot(inputs: dict) -> dict:
    question = inputs["question"]
    answer, _ = answer_question(question)
    # Recuperamos também o contexto, para o avaliador de groundedness
    docs = get_vectorstore().similarity_search(question, k=TOP_K)
    context = "\n\n".join(d.page_content for d in docs)
    return {"answer": answer, "context": context}
 
 
# --- 3. Avaliadores (LLM-as-a-judge) ---
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
 
correctness_prompt = ChatPromptTemplate.from_template(
    """You are grading an answer from a finance Q&A bot.
 
QUESTION: {question}
REFERENCE ANSWER: {reference}
BOT ANSWER: {answer}
 
Is the bot's answer consistent with the reference answer?
Reply with exactly one word: CORRECT or INCORRECT."""
)
 
groundedness_prompt = ChatPromptTemplate.from_template(
    """You are checking whether an answer is grounded in the provided context.
 
CONTEXT:
{context}
 
ANSWER:
{answer}
 
Is every factual claim in the answer supported by the context (no invented facts)?
If the answer states it does not have the information, that counts as grounded.
Reply with exactly one word: GROUNDED or NOT_GROUNDED."""
)
 
 
def correctness(run, example) -> dict:
    msg = correctness_prompt.format_messages(
        question=example.inputs["question"],
        reference=example.outputs["reference"],
        answer=run.outputs.get("answer", ""),
    )
    verdict = judge_llm.invoke(msg).content.strip().upper()
    score = 1 if "CORRECT" in verdict and "INCORRECT" not in verdict else 0
    return {"key": "correctness", "score": score}
 
 
def groundedness(run, example) -> dict:
    msg = groundedness_prompt.format_messages(
        context=run.outputs.get("context", ""),
        answer=run.outputs.get("answer", ""),
    )
    verdict = judge_llm.invoke(msg).content.strip().upper()
    score = 1 if "GROUNDED" in verdict and "NOT_GROUNDED" not in verdict else 0
    return {"key": "groundedness", "score": score}
 
 
def main():
    ensure_dataset()
    print("A correr a avaliação (pode demorar alguns minutos)...\n")
    evaluate(
        run_bot,
        data=DATASET_NAME,
        evaluators=[correctness, groundedness],
        client=client,
        experiment_prefix="finance-bot",
    )
    print("\nAvaliação concluída! Vê os resultados no dashboard do LangSmith.")
 
 
if __name__ == "__main__":
    main()