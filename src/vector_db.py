from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

all_docs = []

#PDF loader
pdf_loader = DirectoryLoader(
    "../documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
all_docs.extend(pdf_loader.load())

def add_basic_metadata(doc):
    source = doc.metadata.get("source", "")
    filename = Path(source).name

    doc.metadata["source_file"] = filename

    lower_name = filename.lower()

    # Generic report type detection from filename
    if any(word in lower_name for word in ["annual", "report-2025", "group-report", "geschaeftsbericht", "geschäftsbericht"]):
        doc.metadata["report_type_hint"] = "Annual / Full-Year Report"
    elif any(word in lower_name for word in ["q1", "quarter", "interim", "zwischenbericht"]):
        doc.metadata["report_type_hint"] = "Quarterly / Interim Report"
    else:
        doc.metadata["report_type_hint"] = "Unknown"

    return doc

all_docs = [add_basic_metadata(doc) for doc in all_docs]

print(f"Total loaded pages/documents: {len(all_docs)}")

sources = sorted(set(
    (
        doc.metadata.get("source_file", "Unknown"),
        doc.metadata.get("report_type_hint", "Unknown")
    )
    for doc in all_docs
))

print("\nLoaded source files:")
for source_file, report_type_hint in sources:
    print(f"- {source_file} | {report_type_hint}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=250
)

docs = splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embeddings)

db.save_local("vector_db")

print("Vector database saved successfully.")