# Clinical Text Clarifier

## Overview

The Clinical Text Clarifier converts free-form English clinical text into structured clinical entities using a single zero-shot LLM call followed by deterministic validation and grounding. The pipeline is intentionally simple: incoming text is lightly cleaned, passed to a Groq model once, checked for schema validity, and then filtered with a deterministic evidence check against the original patient text.

## Architecture

Clinical Text
↓
Light Preprocessing
↓
One Zero-Shot LLM Call
↓
JSON
↓
Validation
↓
Grounding Check
↓
Final Structured Output

## Features

- one LLM call
- zero-shot extraction
- configurable model
- disease extraction
- symptom extraction
- medication extraction
- test extraction
- present/absent/possible status
- evidence grounding
- demographics capture
- no local Hugging Face models

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment

Copy the sample environment file:

```bash
copy .env.example .env
```

Then configure the required values:

```env
GROQ_API_KEY=...
CLINICAL_TEXT_MODEL=openai/gpt-oss-120b
```

Never commit the `.env` file.

## Run

```bash
uvicorn app:app --reload
```

## API

### GET /

Returns basic project information.

### GET /health

Returns:

```json
{
  "status": "healthy"
}
```

### POST /process-text

Request:

```json
{
  "text": "A 56-year-old male presents with severe chest pain."
}
```

Response:

```json
{
  "clinical_text": "A 56-year-old male presents with severe chest pain.",
  "diseases": [],
  "symptoms": ["chest pain"],
  "medications": [],
  "tests": [],
  "entities": [
    {
      "text": "chest pain",
      "type": "symptom",
      "status": "present",
      "evidence_text": "severe chest pain",
      "body_site": null,
      "duration": null,
      "severity": "severe",
      "value": null
    }
  ],
  "demographics": {
    "age": 56,
    "sex": "male"
  },
  "removed_by_grounding_check": 0,
  "model": "openai/gpt-oss-120b",
  "status": "success"
}
```

## Grounding

Evidence is verified against the original input using case-insensitive, whitespace-normalized whole-word/phrase matching. If the model returns non-empty evidence that does not appear in the original patient text, that entity is removed and counted in `removed_by_grounding_check`. Entities with missing or whitespace-only evidence are discarded during validation and are not counted as grounding removals.

## Status

Entity status values are:

- `present`
- `absent`
- `possible`

## Main lists

The primary lists (`diseases`, `symptoms`, `medications`, and `tests`) contain only entities whose status is `present` and whose evidence is grounded in the original text.

## Important limitation

This system extracts and structures information explicitly stated in the text. It does not independently diagnose the patient or infer unsupported clinical facts.

## Knowledge Graph integration

This module does not generate ICD-10 or SNOMED codes. Standard vocabulary mapping is handled by the downstream Knowledge Graph component.

## Model

The model is configurable through `CLINICAL_TEXT_MODEL`, which defaults to `openai/gpt-oss-120b`. This project does not claim to reproduce a specific published implementation exactly and does not claim any specific performance metrics.
