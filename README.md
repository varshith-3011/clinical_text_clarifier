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
- explicitly stated clinical events
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
  "clinical_events": [],
  "demographics": {
    "age": 56,
    "age_unit": null,
    "sex": "male",
    "pregnancy_status": null,
    "gestational_age": null
  },
  "removed_by_grounding_check": 0,
  "model": "openai/gpt-oss-120b",
  "status": "success",
  "error": null
}
```

## Grounding

Evidence is verified against the original input using case-insensitive, whitespace-normalized whole-word/phrase matching. If the model returns non-empty evidence that does not appear in the original patient text, that entity or clinical event is removed and counted in `removed_by_grounding_check`. Items with missing or whitespace-only evidence are discarded during validation and are not counted as grounding removals.

## Status

Entity status values are:

- `present`
- `absent`
- `possible`

## Main lists

The primary lists contain only grounded entities with `present` status. `symptoms` is shaped for downstream phenotype matching: it includes present symptom entities plus test entities explicitly classified during extraction as clinical findings and grounded in their evidence. Tests without an explicit finding classification, including procedures, are not projected into `symptoms`. Projected test findings retain `type: "test"` in `entities` and remain in `tests`. Absent and possible entities are never included in these present-finding lists. The test classification is internal extraction metadata and is not part of the serialized response schema. The downstream Knowledge Graph consumer is not included here, so this repository does not verify the integration.

The existing entity types and statuses are unchanged. Explicit events that do not fit those entity types (for example, a death event) are returned separately in `clinical_events`, with their status and grounded evidence. Demographics retains `age` and `sex` and also reports an explicitly stated `age_unit`, `pregnancy_status`, or `gestational_age` when available. These are additive response fields.

## Important limitation

This system extracts and structures information explicitly stated in the text. It does not independently diagnose the patient or infer unsupported clinical facts.

## Knowledge Graph integration

This module does not generate ICD-10 or SNOMED codes. Vocabulary mapping and downstream consumer behavior are outside this repository and have not been verified here.

## Model

The model is configurable through `CLINICAL_TEXT_MODEL`, which defaults to `openai/gpt-oss-120b`. This project does not claim to reproduce a specific published implementation exactly and does not claim any specific performance metrics.

## Repeatability check

With `GROQ_API_KEY` configured, compare three consecutive production-path extractions of identical test text:

```powershell
python testing/run_determinism.py --text "The patient has a cough and an elevated ESR."
```

The request uses temperature 0 and seed 42. The installed Groq SDK and the configured model accepted the seed parameter during validation, but a seed and temperature 0 do not guarantee identical output across provider or model changes. The report records request attempts, retries, complete public responses, and output differences; it is written to a uniquely named JSON file. Use synthetic or otherwise approved text because the report contains the input and extracted output.
