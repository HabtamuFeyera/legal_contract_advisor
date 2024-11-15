from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from .pdf_loader import extract_text_from_pdf
from .vector_store import store_embeddings, search_embeddings

def process_contract(file_content, question):
    # Step 1: Extract text from PDF
    document_text = extract_text_from_pdf(file_content)

    # Step 2: Store text in Pinecone
    store_embeddings(document_text, {"id": "contract_1"})

    # Step 3: Search for relevant chunks
    relevant_chunks = search_embeddings(question)
    context = " ".join([chunk['metadata']['text'] for chunk in relevant_chunks])

    # Step 4: Use GPT-4 for final answer
    llm = OpenAI(model="gpt-4")
    answer = llm(f"{context}\n\n{question}")
    return answer
