import json
import re
from typing import Any, Iterable

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from config import CLINICAL_TEXT_MODEL, require_groq_api_key
from grounding import is_grounded
from preprocess import clean_text
from schemas import (
    Demographics,
    Entity,
    FinalStructuredResponse,
    VALID_ENTITY_STATUSES,
    VALID_ENTITY_TYPES,
)


def normalize_entity_name(value: Any) -> str:
    if value is None:
        return ""

    normalized = " ".join(str(value).strip().lower().split())
    return normalized


def deduplicate_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized = normalize_entity_name(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)

    return result


def validate_entity(raw_entity: Any) -> Entity | None:
    if not isinstance(raw_entity, dict):
        return None

    text = raw_entity.get("text")
    entity_type = raw_entity.get("type")
    status = raw_entity.get("status")
    evidence_text = raw_entity.get("evidence_text")

    if not isinstance(text, str) or not text.strip():
        return None
    if entity_type not in VALID_ENTITY_TYPES:
        return None
    if status not in VALID_ENTITY_STATUSES:
        return None
    if not isinstance(evidence_text, str):
        return None

    try:
        return Entity(
            text=text.strip(),
            type=entity_type,
            status=status,
            evidence_text=evidence_text.strip(),
            body_site=raw_entity.get("body_site"),
            duration=raw_entity.get("duration"),
            severity=raw_entity.get("severity"),
            value=raw_entity.get("value"),
        )
    except Exception:
        return None


def filter_present_entities(entities: Iterable[Entity]) -> list[str]:
    present_entities = [entity for entity in entities if getattr(entity, "status", None) == "present"]
    return deduplicate_strings(entity.text for entity in present_entities)


def _build_main_lists(entities: list[Entity]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "diseases": [],
        "symptoms": [],
        "medications": [],
        "tests": [],
    }
    seen_by_type: dict[str, set[str]] = {key: set() for key in grouped}

    for entity in entities:
        if entity.status != "present":
            continue

        normalized_name = normalize_entity_name(entity.text)
        if not normalized_name:
            continue

        key_name = entity.type + "s"
        if entity.type == "disease":
            key_name = "diseases"
        elif entity.type == "symptom":
            key_name = "symptoms"
        elif entity.type == "medication":
            key_name = "medications"
        elif entity.type == "test":
            key_name = "tests"

        if normalized_name in seen_by_type[key_name]:
            continue

        seen_by_type[key_name].add(normalized_name)
        grouped[key_name].append(normalized_name)

    return grouped


def _parse_demographics(payload: Any) -> Demographics:
    if not isinstance(payload, dict):
        return Demographics(age=None, sex=None)

    age_value = payload.get("age")
    sex_value = payload.get("sex")

    age: int | None = None
    if isinstance(age_value, bool):
        age = None
    elif isinstance(age_value, (int, float)):
        age = int(age_value)
    elif isinstance(age_value, str):
        stripped = age_value.strip()
        if stripped:
            try:
                age = int(stripped)
            except ValueError:
                age = None

    sex: str | None = None
    if isinstance(sex_value, str):
        normalized_sex = sex_value.strip().lower()
        sex = normalized_sex if normalized_sex in {"male", "female"} else None

    return Demographics(age=age, sex=sex)


def _error_response(original_text: str, message: str, model_name: str) -> FinalStructuredResponse:
    return FinalStructuredResponse(
        clinical_text=original_text,
        diseases=[],
        symptoms=[],
        medications=[],
        tests=[],
        entities=[],
        demographics=Demographics(age=None, sex=None),
        removed_by_grounding_check=0,
        model=model_name,
        status="error",
        error=message,
    )


def _safe_groq_error_message(exc: Exception) -> str:
    status_code = None
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "status_code", None)

    if isinstance(exc, AuthenticationError):
        return f"Groq authentication failed (HTTP {status_code or 401})"
    if isinstance(exc, PermissionDeniedError):
        return f"Groq permission denied (HTTP {status_code or 403})"
    if isinstance(exc, NotFoundError):
        return f"Groq model unavailable (HTTP {status_code or 404})"
    if isinstance(exc, RateLimitError):
        return f"Groq rate limit exceeded (HTTP {status_code or 429})"
    if isinstance(exc, BadRequestError):
        return f"Groq bad request (HTTP {status_code or 400})"
    if isinstance(exc, InternalServerError):
        return f"Groq server error (HTTP {status_code or 500})"
    if isinstance(exc, APIConnectionError):
        return "Groq connection error"
    if isinstance(exc, APITimeoutError):
        return "Groq request timed out"
    if isinstance(exc, APIStatusError):
        if status_code and 400 <= status_code < 500:
            return f"Groq API status error (HTTP {status_code})"
        if status_code and 500 <= status_code < 600:
            return f"Groq server error (HTTP {status_code})"
        return f"Groq API status error (HTTP {status_code or 'unknown'})"

    message = str(exc).strip()
    if not message:
        return "Groq API request failed"

    lowered = message.lower()
    if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return f"Groq authentication failed (HTTP {status_code or 401})"
    if "model" in lowered and ("not found" in lowered or "unavailable" in lowered):
        return f"Groq model unavailable (HTTP {status_code or 404})"
    if "rate limit" in lowered or "too many requests" in lowered:
        return f"Groq rate limit exceeded (HTTP {status_code or 429})"
    if "access" in lowered or "permission" in lowered or "forbidden" in lowered:
        return f"Groq permission denied (HTTP {status_code or 403})"
    if "bad request" in lowered or "validation" in lowered:
        return f"Groq bad request (HTTP {status_code or 400})"

    redacted = re.sub(r"(Authorization|authorization)\s*:\s*Bearer\s+[A-Za-z0-9._-]+", "Authorization: [REDACTED]", message)
    redacted = re.sub(r"(sk|gsk|xg)[A-Za-z0-9_-]{8,}", "[REDACTED]", redacted)
    redacted = re.sub(r"(?i)(api[_-]?key)\s*[:=]\s*[A-Za-z0-9._-]+", "API key=[REDACTED]", redacted)
    redacted = re.sub(r"(?i)(groq[_-]?api[_-]?key)\s*[:=]\s*[A-Za-z0-9._-]+", "GROQ_API_KEY=[REDACTED]", redacted)
    redacted = re.sub(r"\s+", " ", redacted).strip()

    if status_code:
        return f"Groq API request failed (HTTP {status_code}): {redacted}"
    return f"Groq API request failed: {redacted}"


class ClinicalTextProcessingAgent:
    def __init__(self, api_key: str | None = None, model: str | None = None, client: Groq | None = None):
        self.api_key = api_key
        self.model = model or CLINICAL_TEXT_MODEL
        self.client = client

    def process(self, text: str):
        original_text = text if isinstance(text, str) else ""
        cleaned_text = clean_text(original_text)
        model_name = self.model or CLINICAL_TEXT_MODEL

        try:
            api_key = self.api_key or require_groq_api_key()
            client = self.client or Groq(api_key=api_key)

            prompt_system = (
                "ROLE: You are a Clinical Text Clarifier. Convert unstructured English clinical text into structured clinical entities.\n"
                "\n"
                "EXTRACTION RULES:\n"
                "Extract every clinically relevant item explicitly supported by the input. Allowed entity types are only disease, symptom, medication, and test. "
                "Give every entity exactly one status: present, absent, or possible. Never diagnose or infer unsupported information, and do not add a disease merely because symptoms suggest it. "
                "If the text says a condition is suspected, possible, considered, or being evaluated, use possible. A condition explicitly described as patient history is present unless the text says it is absent or resolved.\n"
                "\n"
                "ABBREVIATIONS AND NORMALIZATION:\n"
                "Expand standard clinical abbreviations only when their meaning is clear, such as HTN to hypertension, DM to diabetes mellitus, SOB to shortness of breath, and ECG to electrocardiogram. "
                "Ignore obvious spelling mistakes when normalizing entity text. Never expand, normalize, or rewrite evidence_text.\n"
                "\n"
                "ENTITY BOUNDARIES AND DETAILS:\n"
                "Do not create separate symptom entities for attributes, modifiers, or manifestations that describe one primary symptom. For example, in severe chest pain radiating to the left arm, extract chest pain as the symptom and do not create a separate radiation symptom. "
                "Preserve explicitly stated body_site, duration, severity, and value in their corresponding fields whenever appropriate. Do not invent details; use null when not explicitly supported.\n"
                "\n"
                "MEDICATIONS AND TESTS:\n"
                "Extract every explicitly mentioned medication as its own entity. For example, if the patient takes aspirin and metformin, return both aspirin and metformin. Do not omit later items in a list. "
                "Extract each explicitly mentioned test individually, including ECG, chest X-ray, MRI, CT, and blood tests. When a test or clinical measurement has an explicitly stated result or measurement, preserve it in value when appropriate; for example, represent an explicitly stated blood pressure as a test/measurement with its literal value.\n"
                "\n"
                "HISTORY AND NEGATION:\n"
                "Do not turn lifestyle or history statements into diseases. A statement such as no history of smoking must not produce a tobacco-use disease entity. "
                "Represent explicitly negated clinical findings as absent, not present; for example, denies fever means fever / symptom / absent, and no cough means cough / symptom / absent.\n"
                "\n"
                "DEMOGRAPHICS:\n"
                "Extract age and sex only when explicitly stated. Return age as an integer and sex as male or female when stated; otherwise use null for each missing value.\n"
                "\n"
                "EVIDENCE_TEXT (REQUIRED):\n"
                "Every entity must have evidence_text containing an exact, contiguous substring copied from the ORIGINAL INPUT TEXT. Do not paraphrase, summarize, rewrite, normalize, expand abbreviations, invent wording, or join non-contiguous text. "
                "The normalized entity text may differ from the source wording, but evidence_text must occur literally in the original input. For example, if the input says associated with SOB, use that exact wording as evidence, not associated with shortness of breath. "
                "Multiple entities may share overlapping evidence when the same exact phrase supports them.\n"
                "\n"
                "OUTPUT FORMAT:\n"
                "Return only valid JSON, with no markdown, explanation, or commentary. The top-level object must contain exactly entities and demographics. "
                "demographics must contain exactly age and sex. Each entity must contain exactly text, type, status, evidence_text, body_site, duration, severity, and value. "
                "Use null for any unsupported detail. Do not add other keys."
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": cleaned_text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            payload = json.loads(content)
        except ValueError as exc:
            return _error_response(original_text, str(exc), model_name)
        except Exception as exc:
            safe_message = _safe_groq_error_message(exc)
            return _error_response(original_text, safe_message, model_name)

        if not isinstance(payload, dict):
            return _error_response(original_text, "The model returned an invalid JSON payload.", model_name)

        entities_payload = payload.get("entities", [])
        demographics_payload = payload.get("demographics", {})

        validated_entities: list[Entity] = []
        if isinstance(entities_payload, list):
            for raw_entity in entities_payload:
                entity = validate_entity(raw_entity)
                if entity is not None:
                    validated_entities.append(entity)

        final_entities: list[Entity] = []
        removed_by_grounding_check = 0

        for entity in validated_entities:
            if not is_grounded(entity.evidence_text, original_text):
                removed_by_grounding_check += 1
                continue
            final_entities.append(entity)

        main_lists = _build_main_lists(final_entities)
        demographics = _parse_demographics(demographics_payload)

        return FinalStructuredResponse(
            clinical_text=original_text,
            diseases=main_lists["diseases"],
            symptoms=main_lists["symptoms"],
            medications=main_lists["medications"],
            tests=main_lists["tests"],
            entities=final_entities,
            demographics=demographics,
            removed_by_grounding_check=removed_by_grounding_check,
            model=model_name,
            status="success",
        )
