import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

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
    ClinicalEvent,
    Demographics,
    Entity,
    FinalStructuredResponse,
    VALID_ENTITY_STATUSES,
    VALID_ENTITY_TYPES,
)

TestClassification = Literal["finding", "procedure"]
VALID_TEST_CLASSIFICATIONS = {"finding", "procedure"}
NUMERIC_UNIT_AFTER_VALUE = re.compile(
    r"(?P<separator>\s*)(?P<unit>"
    r"°\s*[CF]|%|"
    r"/(?:min(?:ute)?|hr|hour|sec(?:ond)?|s|day|mm(?:3|³)|uL|µL|μL|mL|L|kg|m2)|"
    r"mm\s*Hg|cm\s*H2O|bpm|beats?\s*/\s*min|"
    r"seconds?|secs?|minutes?|mins?|hours?|days?|weeks?|months?|years?|"
    r"(?:mg|mcg|µg|μg|g|mmol|mEq|IU|U|mL|dL|kg|cm|mm)"
    r"(?:\s*/\s*(?:dL|mL|L|kg|m2))?"
    r")(?=$|[\s,;.)])",
    re.IGNORECASE,
)
PAIRED_TEMPERATURE_AFTER_UNIT = re.compile(
    r"\s*\(\s*\d+(?:\.\d+)?\s*°\s*[CF]\s*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ExtractedEntity:
    entity: Entity
    test_classification: TestClassification | None = None
    test_classification_missing: bool = True
    kg_finding_text: str | None = None


def normalize_entity_name(value: Any) -> str:
    if value is None:
        return ""

    normalized = " ".join(str(value).strip().lower().split())
    return normalized


def _stringify_numeric_value(value: int | float, evidence_text: str) -> str:
    value_text = str(value)
    number_pattern = re.compile(
        rf"(?<![\w.]){re.escape(value_text)}(?![\w.])",
        re.IGNORECASE,
    )
    number_match = number_pattern.search(evidence_text)
    if number_match is None:
        return value_text

    unit_match = NUMERIC_UNIT_AFTER_VALUE.match(evidence_text[number_match.end():])
    if unit_match is None:
        return value_text
    unit_text = unit_match.group("unit")
    unit_end = number_match.end() + unit_match.end()
    if unit_text.lstrip().startswith("°"):
        paired_temperature = PAIRED_TEMPERATURE_AFTER_UNIT.match(
            evidence_text[unit_end:]
        )
        if paired_temperature is not None:
            unit_text += paired_temperature.group()
    return f"{value_text}{unit_match.group('separator')}{unit_text}"


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
    extracted_entity = _validate_extracted_entity(raw_entity)
    return extracted_entity.entity if extracted_entity is not None else None


def _validate_extracted_entity(raw_entity: Any) -> _ExtractedEntity | None:
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
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        return None

    try:
        value = raw_entity.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = _stringify_numeric_value(value, evidence_text)
        entity = Entity(
            text=text.strip(),
            type=entity_type,
            status=status,
            evidence_text=evidence_text.strip(),
            body_site=raw_entity.get("body_site"),
            duration=raw_entity.get("duration"),
            severity=raw_entity.get("severity"),
            value=value,
        )
    except Exception:
        return None

    classification_missing = "test_classification" not in raw_entity
    raw_classification = raw_entity.get("test_classification")
    test_classification: TestClassification | None = None
    if (
        entity.type == "test"
        and isinstance(raw_classification, str)
        and raw_classification in VALID_TEST_CLASSIFICATIONS
    ):
        test_classification = raw_classification

    raw_kg_finding_text = raw_entity.get("kg_finding_text")
    kg_finding_text = (
        raw_kg_finding_text.strip()
        if entity.type == "test"
        and isinstance(raw_kg_finding_text, str)
        and raw_kg_finding_text.strip()
        else None
    )

    return _ExtractedEntity(
        entity=entity,
        test_classification=test_classification,
        test_classification_missing=classification_missing,
        kg_finding_text=kg_finding_text,
    )


def validate_clinical_event(raw_event: Any) -> ClinicalEvent | None:
    if not isinstance(raw_event, dict):
        return None

    text = raw_event.get("text")
    status = raw_event.get("status")
    evidence_text = raw_event.get("evidence_text")

    if not isinstance(text, str) or not text.strip():
        return None
    if status not in VALID_ENTITY_STATUSES:
        return None
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        return None

    return ClinicalEvent(
        text=text.strip(),
        status=status,
        evidence_text=evidence_text.strip(),
    )


def filter_present_entities(entities: Iterable[Entity]) -> list[str]:
    present_entities = [entity for entity in entities if getattr(entity, "status", None) == "present"]
    return deduplicate_strings(entity.text for entity in present_entities)


def _build_main_lists(entities: list[_ExtractedEntity]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "diseases": [],
        "symptoms": [],
        "medications": [],
        "tests": [],
    }
    seen_by_type: dict[str, set[str]] = {key: set() for key in grouped}
    statuses_by_name: dict[str, set[str]] = {}
    for extracted_entity in entities:
        entity = extracted_entity.entity
        normalized_name = normalize_entity_name(entity.text)
        if normalized_name:
            statuses_by_name.setdefault(normalized_name, set()).add(entity.status)
    conflicting_symptom_names = {
        name
        for name, statuses in statuses_by_name.items()
        if "present" in statuses and "absent" in statuses
    }

    def append_unique(
        list_name: str, text: str, conflict_name: str | None = None
    ) -> None:
        normalized_name = normalize_entity_name(text)
        if not normalized_name or normalized_name in seen_by_type[list_name]:
            return
        if list_name == "symptoms" and (
            normalized_name in conflicting_symptom_names
            or normalize_entity_name(conflict_name) in conflicting_symptom_names
        ):
            return
        seen_by_type[list_name].add(normalized_name)
        grouped[list_name].append(normalized_name)

    for extracted_entity in entities:
        entity = extracted_entity.entity
        if entity.status != "present":
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

        append_unique(key_name, entity.text)
        if entity.type == "test":
            if extracted_entity.test_classification == "finding":
                symptom_text = entity.text
                if (
                    extracted_entity.kg_finding_text
                    and is_grounded(
                        extracted_entity.kg_finding_text, entity.evidence_text
                    )
                ):
                    symptom_text = extracted_entity.kg_finding_text
                elif is_grounded(entity.text, entity.evidence_text):
                    symptom_text = entity.text
                append_unique("symptoms", symptom_text, conflict_name=entity.text)

    return grouped


def _parse_demographics(payload: Any) -> Demographics:
    if not isinstance(payload, dict):
        return Demographics()

    age_value = payload.get("age")
    age_unit_value = payload.get("age_unit")
    sex_value = payload.get("sex")

    age: int | None = None
    age_unit: str | None = None
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
                age_match = re.fullmatch(
                    r"(\d+)\s*[- ]?(year|month|week|day)s?(?:[- ]?old)?",
                    stripped,
                    flags=re.IGNORECASE,
                )
                if age_match:
                    age = int(age_match.group(1))
                    age_unit = f"{age_match.group(2).lower()}s"
    if isinstance(age_unit_value, str) and age_unit_value.strip():
        normalized_age_unit = age_unit_value.strip().lower().rstrip("s")
        age_unit = {
            "year": "years",
            "month": "months",
            "week": "weeks",
            "day": "days",
        }.get(normalized_age_unit)

    sex: str | None = None
    if isinstance(sex_value, str):
        normalized_sex = sex_value.strip().lower()
        sex = normalized_sex if normalized_sex in {"male", "female"} else None

    pregnancy_status = payload.get("pregnancy_status")
    if isinstance(pregnancy_status, str):
        pregnancy_status = pregnancy_status.strip().lower()
        if pregnancy_status not in {"pregnant", "not pregnant"}:
            pregnancy_status = None
    else:
        pregnancy_status = None

    gestational_age = payload.get("gestational_age")
    if isinstance(gestational_age, str):
        gestational_age = gestational_age.strip() or None
    else:
        gestational_age = None

    return Demographics(
        age=age,
        age_unit=age_unit,
        sex=sex,
        pregnancy_status=pregnancy_status,
        gestational_age=gestational_age,
    )


def _error_response(original_text: str, message: str, model_name: str) -> FinalStructuredResponse:
    return FinalStructuredResponse(
        clinical_text=original_text,
        diseases=[],
        symptoms=[],
        medications=[],
        tests=[],
        entities=[],
        clinical_events=[],
        demographics=Demographics(),
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
                "Distinguish a named disease diagnosis from a phenotype finding: do not use disease as a catchall for an unfamiliar or disorder-like term. If the source presents an item as an observed clinical finding or manifestation rather than a disease diagnosis, classify it as symptom; classify laboratory and diagnostic test findings as test with test_classification finding. Do not turn a stated diagnosis into a symptom solely to increase coverage. "
                "If the text says a condition is suspected, possible, considered, or being evaluated, use possible. A condition explicitly described as patient history is present unless the text says it is absent or resolved.\n"
                "\n"
                "Extract other explicitly stated clinical events, such as death, in clinical_events rather than misclassifying them as a disease or symptom. "
                "Only include events that the input states actually occurred; provide text, status, and exact evidence_text for each.\n"
                "In MedQA-style input, extract facts from the clinical vignette only. Do not answer the question, choose an option, or treat a hypothetical question or answer choices as clinical findings.\n"
                "\n"
                "ABBREVIATIONS AND NORMALIZATION:\n"
                "Expand standard clinical abbreviations only when their meaning is clear, such as HTN to hypertension, DM to diabetes mellitus, SOB to shortness of breath, and ECG to electrocardiogram. "
                "Ignore obvious spelling mistakes when normalizing entity text. Never expand, normalize, or rewrite evidence_text.\n"
                "\n"
                "ENTITY BOUNDARIES AND DETAILS:\n"
                "Do not create separate symptom entities for attributes, modifiers, or manifestations that describe one primary symptom. For example, in severe chest pain radiating to the left arm, extract chest pain as the symptom and do not create a separate radiation symptom. "
                "Do not shorten, paraphrase, or drop clinically meaningful parts of a finding. Preserve the complete source finding phrase, including anatomical location and meaningful qualifiers such as duration, measurement, abnormality direction, and positive-test wording. "
                "Use the complete source finding phrase as the entity text, not a generic base name when the source includes a meaningful qualifier. Keep evidence_text to the shortest exact source span that contains the complete finding phrase, without surrounding narrative or neighboring findings. "
                "Preserve explicitly stated body_site, duration, severity, and value in their corresponding fields whenever appropriate. For any explicitly stated vital sign or numeric laboratory measurement, put the measurement value and its unit as a string in value, preserving the complete reported measurement, including paired-unit forms when stated (for example, 38.0°C (100.4°F), 112/min, 150/90 mm Hg, or 43 seconds). Do not put a measurement in value unless the source states it. "
                "Assign value only when the source states a result or measurement that genuinely belongs to that specific test or measurement; do not attach a nearby sentence or an unrelated event as its value. "
                "For a procedure with no stated result, use null. Do not invent details; use null when not explicitly supported.\n"
                "\n"
                "MEDICATIONS AND TESTS:\n"
                "Extract every explicitly mentioned medication as its own entity. For example, if the patient takes aspirin and metformin, return both aspirin and metformin. Do not omit later items in a list. "
                "Extract each explicitly mentioned test individually, including ECG, chest X-ray, MRI, CT, and blood tests. Keep laboratory measurements and diagnostic test findings typed as test; do not relabel them as symptoms. "
                "For each test entity, set the internal test_classification to finding only when the source explicitly reports a clinical, laboratory, or diagnostic finding; this includes named abnormalities such as anaemia even when no numeric value is stated. "
                "For a test classified as finding, set the internal kg_finding_text to the complete source phrase for that one finding only, copied exactly from its evidence_text; exclude surrounding narrative and neighboring findings. Use null when the source does not support one exact finding phrase. "
                "Set it to procedure when an investigation or procedure is mentioned without a reported finding. Do not infer a finding from a procedure name, and do not invent a result, measurement, or value. "
                "Keep status separate from classification: explicitly abnormal or positive findings are present, explicitly normal or negative findings are absent, and uncertain findings are possible. "
                "A test procedure without a stated finding may be present as a test but must have value null. For older or uncertain cases, leave test_classification null rather than guessing.\n"
                "\n"
                "HISTORY AND NEGATION:\n"
                "Do not turn lifestyle or history statements into diseases. A statement such as no history of smoking must not produce a tobacco-use disease entity. "
                "Represent explicitly negated clinical findings as absent, not present; for example, denies fever means fever / symptom / absent, and no cough means cough / symptom / absent.\n"
                "\n"
                "DEMOGRAPHICS:\n"
                "Extract age and sex only when explicitly stated. Return age as an integer and preserve its unit in age_unit (years, months, weeks, or days) when stated; otherwise use null for each missing value. "
                "Do not interpret gestational age as the patient's age. Record explicitly stated pregnancy as pregnancy_status ('pregnant' or 'not pregnant') and preserve gestational_age with its unit, such as '22 weeks'. Use null when not stated.\n"
                "\n"
                "EVIDENCE_TEXT (REQUIRED):\n"
                "Every entity and clinical event must have evidence_text containing an exact, contiguous substring copied from the ORIGINAL INPUT TEXT. Do not paraphrase, summarize, rewrite, normalize, expand abbreviations, invent wording, or join non-contiguous text. "
                "The normalized entity text may differ from the source wording, but evidence_text must occur literally in the original input. For example, if the input says associated with SOB, use that exact wording as evidence, not associated with shortness of breath. "
                "Multiple entities may share overlapping evidence when the same exact phrase supports them.\n"
                "\n"
                "OUTPUT FORMAT:\n"
                "Return only valid JSON, with no markdown, explanation, or commentary. The top-level object must contain exactly entities, clinical_events, and demographics. "
                "demographics must contain exactly age, age_unit, sex, pregnancy_status, and gestational_age. Each entity must contain exactly text, type, status, evidence_text, body_site, duration, severity, value, test_classification, and kg_finding_text. "
                "For non-test entities, test_classification must be null. For test entities, it must be finding, procedure, or null when the source does not support a reliable choice. This field is internal extraction metadata and must not be added to public response entities. "
                "kg_finding_text is also internal extraction metadata and must never be added to public response entities. Set it to null for non-test entities and procedures. "
                "Each clinical event must contain exactly text, status, and evidence_text. Use null for any unsupported detail and an empty array when there are no clinical events. Do not add other keys."
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": cleaned_text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                seed=42,
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

        if "entities" not in payload:
            return _error_response(original_text, "The model response is missing the entities field.", model_name)

        entities_payload = payload["entities"]
        demographics_payload = payload.get("demographics", {})
        clinical_events_payload = payload.get("clinical_events", [])

        if not isinstance(entities_payload, list):
            return _error_response(original_text, "The model returned an invalid entities field.", model_name)
        if not isinstance(clinical_events_payload, list):
            return _error_response(original_text, "The model returned an invalid clinical_events field.", model_name)

        validated_entities: list[_ExtractedEntity] = []
        for raw_entity in entities_payload:
            extracted_entity = _validate_extracted_entity(raw_entity)
            if extracted_entity is not None:
                validated_entities.append(extracted_entity)

        validated_events: list[ClinicalEvent] = []
        for raw_event in clinical_events_payload:
            event = validate_clinical_event(raw_event)
            if event is not None:
                validated_events.append(event)

        final_entities: list[_ExtractedEntity] = []
        final_events: list[ClinicalEvent] = []
        removed_by_grounding_check = 0

        for extracted_entity in validated_entities:
            entity = extracted_entity.entity
            if not is_grounded(entity.evidence_text, original_text):
                removed_by_grounding_check += 1
                continue
            final_entities.append(extracted_entity)

        for event in validated_events:
            if not is_grounded(event.evidence_text, original_text):
                removed_by_grounding_check += 1
                continue
            final_events.append(event)

        main_lists = _build_main_lists(final_entities)
        demographics = _parse_demographics(demographics_payload)

        return FinalStructuredResponse(
            clinical_text=original_text,
            diseases=main_lists["diseases"],
            symptoms=main_lists["symptoms"],
            medications=main_lists["medications"],
            tests=main_lists["tests"],
            entities=[extracted_entity.entity for extracted_entity in final_entities],
            clinical_events=final_events,
            demographics=demographics,
            removed_by_grounding_check=removed_by_grounding_check,
            model=model_name,
            status="success",
        )
