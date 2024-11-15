from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from src.models.rag_pipeline import process_contract
from typing import Dict

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_contract(
    file: UploadFile = File(...),
    query: str = Form(...)
) -> Dict[str, str]:
    """
    Endpoint to analyze a legal contract and answer a user's question.

    Args:
    - file (UploadFile): The uploaded PDF file.
    - query (str): The user's legal question.

    Returns:
    - dict: A dictionary containing the answer.
    """
    try:
        # Read the uploaded file
        content = await file.read()

        # Process the contract and generate an answer
        answer = process_contract(content, query)

        return {"answer": answer}
    except Exception as e:
        print(f"Error processing contract: {e}")
        return {"error": "Unable to process the request. Please try again."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
