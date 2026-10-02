from agent import deduplicate_strings, filter_present_entities, normalize_entity_name, validate_entity
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


def test_normalize_entity_name():
    assert normalize_entity_name("  chest   pain  ") == "chest pain"
    assert normalize_entity_name(None) == ""
