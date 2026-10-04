import json
from types import SimpleNamespace
from unittest.mock import Mock

from agent import ClinicalTextProcessingAgent, deduplicate_strings, filter_present_entities, normalize_entity_name, validate_entity
from schemas import Entity


def test_filter_present_entities():
    entities = [
        Entity(text="fever", type="symptom", status="absent", evidence_text="denies fever"),
        Entity(text="Chest Pain", type="symptom", status="present", evidence_text="chest pain"),
        Entity(text="Cough", type="symptom", status="possible", evidence_text="possible cough"),
        Entity(text="fever", type="symptom", status="present", evidence_text="fever"),
    ]
    assert filter_present_entities(entities) == ["chest pain", "fever"]


def test_deduplication_and_normalization():
    values = [" Fever ", "fever", "COUGH", "cough", " chest   pain ", "chest pain"]
    assert deduplicate_strings(values) == ["fever", "cough", "chest pain"]


def test_malformed_entity_handling():
    assert validate_entity({"text": "fever", "type": "symptom", "status": "present", "evidence_text": "fever"}) is not None
    assert validate_entity({"text": "", "type": "symptom", "status": "present", "evidence_text": "fever"}) is None
    assert validate_entity({"text": "fever", "type": "unknown", "status": "present", "evidence_text": "fever"}) is None
    assert validate_entity({"text": "fever", "type": "symptom", "status": "present"}) is None
    assert validate_entity({"text": "fever", "type": "symptom", "status": "present", "evidence_text": "   "}) is None


def test_non_list_entities_returns_error_response():
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps({"entities": {}, "demographics": {}})))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process("Patient has fever.")

    assert result.status == "error"
    assert result.error == "The model returned an invalid entities field."
    assert result.removed_by_grounding_check == 0


def test_empty_evidence_is_not_counted_as_grounding_removal():
    payload = {
        "entities": [
            {"text": "fever", "type": "symptom", "status": "present", "evidence_text": "   "},
            {"text": "pneumonia", "type": "disease", "status": "present", "evidence_text": "pneumonia"},
            {"text": "cough", "type": "symptom", "status": "present", "evidence_text": "cough"},
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process("Patient has cough.")

    assert result.status == "success"
    assert [entity.text for entity in result.entities] == ["cough"]
    assert result.removed_by_grounding_check == 1


def test_normalize_entity_name():
    assert normalize_entity_name("  chest   pain  ") == "chest pain"
    assert normalize_entity_name(None) == ""
