from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 15})

llm = Ollama(model="llama3.1:8b")


def rag_answer(question: str) -> str:
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are a Financial Report Analysis AI Agent.

You must use ONLY the provided context from the uploaded OEM financial reports.
Do not use outside knowledge, internet data, assumptions, or memory.

Your task is to extract financial KPIs from the provided reports and create an executive summary.

Rules:
- If a value is not available in a quarterly report, write "Not Reported".
- If a value is not disclosed at all, write "N/A".
- If the exact requested metric is not found, use the closest equivalent company-specific KPI.
- If you use an equivalent KPI, clearly mention it in a short note.
- Do not invent numbers.
- Keep units exactly as shown in the reports, for example EUR million, EUR billion, %, or per share.
- If possible, mention the source report or page context for important values.

Required output:
Create an executive summary table with these rows:
1. Company name
2. Revenue
3. EBIT / Operating Result
4. Operating margin = EBIT divided by Revenue
5. Cash Metric, meaning the company-preferred core cash KPI
6. Net liquidity / liquidity indicator
7. Return on capital metric, meaning the company’s value-based KPI
8. Cost of Capital / hurdle concept, only where disclosed
9. EPS & dividend per share
10. Market cap if share value would be 100 EUR/share

The table should compare:
- Full-year 2025 data
- Latest quarterly data

for all 3 companies.

After the table, add a short note explaining:
- which KPIs were replaced by equivalent company-specific metrics
- which values were Not Reported
- which values were N/A

Context:
{context}

Question:
{question}

Answer:
"""

    return llm.invoke(prompt)