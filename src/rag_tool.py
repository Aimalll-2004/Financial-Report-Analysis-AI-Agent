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

def list_indexed_reports() -> str:
    docs = list(db.docstore._dict.values())

    reports = {}

    for doc in docs:
        source_file = doc.metadata.get("source_file", doc.metadata.get("source", "Unknown"))
        report_type_hint = doc.metadata.get("report_type_hint", "Unknown")

        key = (source_file, report_type_hint)
        reports[key] = reports.get(key, 0) + 1

    output = "| Source file | Report type hint | Indexed chunks/pages |\n"
    output += "|---|---|---|\n"

    for (source_file, report_type_hint), count in sorted(reports.items()):
        output += f"| {source_file} | {report_type_hint} | {count} |\n"

    return output

def rag_answer(question: str) -> str:
    # Direct metadata listing without asking the LLM
    if "list indexed reports" in question.lower() or "list all source filenames" in question.lower():
        return list_indexed_reports()

    retrieved_docs = retriever.invoke(question)

    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source_file = doc.metadata.get("source_file", doc.metadata.get("source", "Unknown source"))
        page = doc.metadata.get("page", "Unknown page")
        report_type = doc.metadata.get("report_type_hint", "Unknown report type")

        context_parts.append(
            f"""
[Source {i}]
File: {source_file}
Page: {page}
Report type hint: {report_type}

Content:
{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a Financial Report Analysis AI Agent.

You analyze uploaded company financial reports.

You must use ONLY the provided context from the uploaded reports.
Do not use internet data, outside knowledge, assumptions, or memory.

Your job:
- identify financial KPIs from annual, full-year, quarterly, and interim reports
- handle different company terminology
- handle different languages where possible
- extract numbers exactly as reported
- keep units exactly as shown
- clearly mark missing data

Important rules:
- If a value is not available in a quarterly/interim report, write "Not Reported".
- If a value is not disclosed anywhere in the provided context, write "N/A".
- If the exact requested metric is not found, use the closest equivalent company-specific KPI.
- If you use an equivalent KPI, explain this in a short note.
- Do not invent numbers.
- For important values, mention the source file and page number when available.

Financial terminology examples:

Revenue may also appear as:
- sales
- group revenues
- Umsatzerlöse

EBIT / Operating Result may also appear as:
- operating result
- profit before financial result
- operatives Ergebnis

Operating margin may also appear as:
- EBIT margin
- return on sales
- operating return on sales
- operative Umsatzrendite

Cash metric may also appear as:
- free cash flow
- industrial free cash flow
- net cash flow
- Netto-Cashflow
- cash flow before interest and taxes
- CFBIT

Net liquidity may also appear as:
- net liquidity
- industrial net liquidity
- net cash position
- liquidity indicator
- Nettoliquidität

Return on capital may also appear as:
- ROCE
- ROIC
- RONA
- return on capital employed
- return on invested capital
- value contribution
- return on sales / return on equity where used as company-specific KPI

EPS and dividend may also appear as:
- earnings per share
- Ergebnis je Aktie
- dividend per share
- Dividende je Aktie

Context:
{context}

Question:
{question}

Answer:
"""

    return llm.invoke(prompt)