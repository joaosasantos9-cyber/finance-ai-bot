"""
Avaliação do bot com LangSmith.
 
Como funciona:
  1. Definimos um conjunto de perguntas + respostas de referência (a "verdade").
  2. Criamos um dataset no LangSmith com esses exemplos.
  3. Corremos o bot sobre cada pergunta.
  4. Um LLM "juiz" (LLM-as-a-judge) compara a resposta do bot com a referência
     e dá uma nota (correto / incorreto).
  5. Os resultados aparecem no dashboard do LangSmith.
 
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
 
client = Client()
DATASET_NAME = "finance-bot-eval"
 
 
# --- 1. Conjunto de teste (perguntas + respostas de referência) ---
EXAMPLES = [
    {
        "question": "What is a good starting goal for an emergency fund?",
        "reference": "A common starting goal is around $1,000.",
    },
    {
        "question": "Name a few ways to build an emergency fund.",
        "reference": "Set a specific goal, cut unnecessary expenses, automate savings, "
                     "pick up a side hustle, and use windfalls like tax refunds.",
    },
    {
        "question": "What is an advantage of index funds or ETFs for beginners?",
        "reference": "They offer low fees and broad diversification, and most active "
                     "managers fail to beat the market over time.",
    },
    {
        "question": "What is one consideration when deciding to rent versus buy a home?",
        "reference": "It depends on factors like how long you'll stay, the costs of "
                     "ownership, and what you could earn investing the difference.",
    },
    {
        "question": "What is a downside of personal loans?",
        "reference": "They can carry high interest rates and fees, and add to your debt burden.",
    },
    {
        "question": "What is the capital of France?",
        "reference": "The bot should say it does not have this information in the videos.",
    },
]
 
 
def ensure_dataset():
    """Cria o dataset no LangSmith (ou reutiliza se já existir)."""
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' já existe — a reutilizar.")
        return dataset
    except Exception:
        print(f"A criar dataset '{DATASET_NAME}'...")
        dataset = client.create_dataset(dataset_name=DATASET_NAME)
        client.create_examples(
            inputs=[{"question": e["question"]} for e in EXAMPLES],
            outputs=[{"reference": e["reference"]} for e in EXAMPLES],
            dataset_id=dataset.id,
        )
        return dataset
 
 
# --- 3. Função-alvo: corre o nosso bot numa pergunta ---
def run_bot(inputs: dict) -> dict:
    answer, _ = answer_question(inputs["question"])
    return {"answer": answer}
 
 
# --- 4. Avaliador: LLM-as-a-judge ---
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
judge_prompt = ChatPromptTemplate.from_template(
    """You are grading an answer from a finance Q&A bot.
 
QUESTION: {question}
REFERENCE ANSWER: {reference}
BOT ANSWER: {answer}
 
Is the bot's answer consistent with the reference answer?
Reply with exactly one word: CORRECT or INCORRECT."""
)
 
 
def correctness(run, example) -> dict:
    """Compara a resposta do bot com a referência usando um LLM juiz."""
    answer = run.outputs.get("answer", "")
    question = example.inputs["question"]
    reference = example.outputs["reference"]
 
    msg = judge_prompt.format_messages(
        question=question, reference=reference, answer=answer
    )
    verdict = judge_llm.invoke(msg).content.strip().upper()
    score = 1 if "CORRECT" in verdict and "INCORRECT" not in verdict else 0
    return {"key": "correctness", "score": score}
 
 
def main():
    ensure_dataset()
    print("A correr a avaliação (pode demorar 1-2 min)...\n")
    results = evaluate(
        run_bot,
        data=DATASET_NAME,
        evaluators=[correctness],
        client=client,
        experiment_prefix="finance-bot",
    )
    print("\nAvaliação concluída! Vê os resultados no dashboard do LangSmith.")
    return results
 
 
if __name__ == "__main__":
    main()