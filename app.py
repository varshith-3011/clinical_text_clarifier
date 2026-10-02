from fastapi import FastAPI

from agent import ClinicalTextProcessingAgent
from config import CLINICAL_TEXT_MODEL
from schemas import ClinicalInput, FinalStructuredResponse

app = FastAPI(title="Clinical Text Clarifier")


@app.get("/")
def read_root():
    return {
        "name": "Clinical Text Clarifier",
        "description": "Zero-shot clinical text extraction using a single Groq LLM call and deterministic grounding.",
        "model": CLINICAL_TEXT_MODEL,
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/process-text", response_model=FinalStructuredResponse)
def process_text(payload: ClinicalInput):
    agent = ClinicalTextProcessingAgent()
    return agent.process(payload.text)
