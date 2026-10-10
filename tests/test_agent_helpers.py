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


def test_missing_entities_returns_error_response():
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps({"demographics": {}})))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process("Patient has fever.")

    assert result.status == "error"
    assert result.error == "The model response is missing the entities field."
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
    assert result.symptoms == ["cough"]
    assert result.removed_by_grounding_check == 1


def test_present_absent_conflicts_are_suppressed_independent_of_entity_order():
    text = "Patient reports fever but denies fever."
    entries = [
        {
            "text": "Fever",
            "type": "symptom",
            "status": "present",
            "evidence_text": "reports fever",
        },
        {
            "text": " fever ",
            "type": "symptom",
            "status": "absent",
            "evidence_text": "denies fever",
        },
    ]

    for entities in (entries, list(reversed(entries))):
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({"entities": entities, "demographics": {}})
                )
            )
        ]

        result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

        assert result.status == "success"
        assert result.symptoms == []
        assert [(entity.text, entity.status) for entity in result.entities] == [
            (entry["text"].strip(), entry["status"]) for entry in entities
        ]


def test_absent_and_possible_entities_are_retained_but_not_projected_as_symptoms():
    payload = {
        "entities": [
            {
                "text": "fever",
                "type": "symptom",
                "status": "absent",
                "evidence_text": "denies fever",
            },
            {
                "text": "cough",
                "type": "symptom",
                "status": "possible",
                "evidence_text": "possible cough",
            },
            {
                "text": "pneumonia",
                "type": "disease",
                "status": "present",
                "evidence_text": "pneumonia",
            },
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(
        "Patient denies fever; possible cough; pneumonia."
    )

    assert result.status == "success"
    assert result.symptoms == []
    assert [(entity.text, entity.status) for entity in result.entities] == [
        ("fever", "absent"),
        ("cough", "possible"),
        ("pneumonia", "present"),
    ]
    assert result.diseases == ["pneumonia"]


def test_symptom_deduplication_preserves_qualified_phrases():
    text = "Patient has cough and persistent cough."
    payload = {
        "entities": [
            {
                "text": "Cough",
                "type": "symptom",
                "status": "present",
                "evidence_text": "cough",
            },
            {
                "text": "  cough  ",
                "type": "symptom",
                "status": "present",
                "evidence_text": "cough",
            },
            {
                "text": "persistent cough",
                "type": "symptom",
                "status": "present",
                "evidence_text": "persistent cough",
            },
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == ["cough", "persistent cough"]
    assert len(result.entities) == 3


def test_ungrounded_entity_is_removed_before_symptom_projection():
    payload = {
        "entities": [
            {
                "text": "cough",
                "type": "symptom",
                "status": "present",
                "evidence_text": "cough",
            },
            {
                "text": "fever",
                "type": "symptom",
                "status": "present",
                "evidence_text": "fever",
            },
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(
        "Patient has cough."
    )

    assert result.status == "success"
    assert result.symptoms == ["cough"]
    assert [entity.text for entity in result.entities] == ["cough"]
    assert result.removed_by_grounding_check == 1


def test_medqa_pregnancy_and_gestational_age_are_preserved():
    text = (
        "A pregnant woman at 22 weeks gestation presents with burning during "
        "urination for one day."
    )
    payload = {
        "entities": [
            {
                "text": "dysuria",
                "type": "symptom",
                "status": "present",
                "evidence_text": "burning during urination",
                "body_site": None,
                "duration": "one day",
                "severity": None,
                "value": None,
            }
        ],
        "clinical_events": [],
        "demographics": {
            "age": None,
            "age_unit": None,
            "sex": None,
            "pregnancy_status": "pregnant",
            "gestational_age": "22 weeks",
        },
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == ["dysuria"]
    assert result.entities[0].duration == "one day"
    assert result.demographics.pregnancy_status == "pregnant"
    assert result.demographics.gestational_age == "22 weeks"
    client.chat.completions.create.assert_called_once()
    request_kwargs = client.chat.completions.create.call_args.kwargs
    prompt = request_kwargs["messages"][0]["content"]
    assert request_kwargs["temperature"] == 0
    assert request_kwargs["seed"] == 42
    assert not {"top_p", "frequency_penalty", "presence_penalty"}.intersection(request_kwargs)
    assert "Do not interpret gestational age as the patient's age" in prompt
    assert "Do not answer the question" in prompt
    assert "test_classification" in prompt
    assert "This field is internal extraction metadata" in prompt
    assert "do not use disease as a catchall" in prompt.casefold()


def test_present_test_findings_are_projected_without_changing_entity_types():
    text = (
        "Exam shows calf muscle pseudohypertrophy. ESR was elevated, mildly elevated on repeat, "
        "and ANCA was positive. "
        "MRI was performed; the patient denies fever and takes metformin."
    )
    payload = {
        "entities": [
            {
                "text": "calf muscle pseudohypertrophy",
                "type": "symptom",
                "status": "present",
                "evidence_text": "calf muscle pseudohypertrophy",
            },
            {
                "text": "elevated ESR",
                "type": "test",
                "status": "present",
                "evidence_text": "ESR was elevated",
                "value": "elevated",
                "test_classification": "finding",
            },
            {
                "text": "ELEVATED ESR",
                "type": "test",
                "status": "present",
                "evidence_text": "ESR was elevated",
                "value": "elevated",
                "test_classification": "finding",
            },
            {
                "text": "mildly elevated ESR",
                "type": "test",
                "status": "present",
                "evidence_text": "mildly elevated on repeat",
                "value": "mildly elevated",
                "test_classification": "finding",
            },
            {
                "text": "ANCA positivity",
                "type": "test",
                "status": "present",
                "evidence_text": "ANCA was positive",
                "value": "positive",
                "test_classification": "finding",
            },
            {
                "text": "MRI",
                "type": "test",
                "status": "present",
                "evidence_text": "MRI was performed",
                "value": None,
                "test_classification": "procedure",
            },
            {
                "text": "fever",
                "type": "symptom",
                "status": "absent",
                "evidence_text": "denies fever",
            },
            {
                "text": "metformin",
                "type": "medication",
                "status": "present",
                "evidence_text": "metformin",
            },
        ],
        "clinical_events": [],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    agent = ClinicalTextProcessingAgent(api_key="test-key", client=client)
    results = [agent.process(text) for _ in range(3)]
    result = results[0]

    assert result.status == "success"
    assert all(candidate.model_dump() == result.model_dump() for candidate in results[1:])
    assert result.symptoms == [
        "calf muscle pseudohypertrophy",
        "elevated esr",
        "mildly elevated esr",
        "anca positivity",
    ]
    assert result.tests == ["elevated esr", "mildly elevated esr", "anca positivity", "mri"]
    assert result.medications == ["metformin"]
    assert [(entity.text, entity.type) for entity in result.entities if entity.type == "test"] == [
        ("elevated ESR", "test"),
        ("ELEVATED ESR", "test"),
        ("mildly elevated ESR", "test"),
        ("ANCA positivity", "test"),
        ("MRI", "test"),
    ]
    assert result.removed_by_grounding_check == 0
    assert "findings" not in result.model_dump()
    assert all("test_classification" not in entity.model_dump() for entity in result.entities)
    assert set(result.entities[1].model_dump()) == {
        "text",
        "type",
        "status",
        "evidence_text",
        "body_site",
        "duration",
        "severity",
        "value",
    }
    assert set(result.model_dump()) == {
        "clinical_text",
        "diseases",
        "symptoms",
        "medications",
        "tests",
        "entities",
        "clinical_events",
        "demographics",
        "removed_by_grounding_check",
        "model",
        "status",
        "error",
    }


def test_test_finding_projection_preserves_qualifiers_from_grounded_evidence():
    findings = [
        ("IgG4 level", "increased circulating IgG4 level", None),
        ("creatinine concentration", "elevated circulating creatinine concentration", None),
        ("proteinuria", "moderate proteinuria", "moderate"),
        ("proteinuria", "heavy proteinuria", "heavy"),
        ("urinary carboxylic acid", "elevated urinary carboxylic acid", "elevated"),
    ]
    text = "Findings: " + "; ".join(evidence for _, evidence, _ in findings) + "."
    payload = {
        "entities": [
            {
                "text": entity_text,
                "type": "test",
                "status": "present",
                "evidence_text": evidence_text,
                "value": value,
                "test_classification": "finding",
                "kg_finding_text": evidence_text,
            }
            for entity_text, evidence_text, value in findings
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == [
        "increased circulating igg4 level",
        "elevated circulating creatinine concentration",
        "moderate proteinuria",
        "heavy proteinuria",
        "elevated urinary carboxylic acid",
    ]
    assert result.tests == [
        "igg4 level",
        "creatinine concentration",
        "proteinuria",
        "urinary carboxylic acid",
    ]
    assert [entity.text for entity in result.entities] == [
        entity_text for entity_text, _, _ in findings
    ]


def test_test_finding_projection_uses_one_grounded_phrase_not_broad_evidence():
    text = "Patient reports persistent cough and fever."
    payload = {
        "entities": [
            {
                "text": "cough",
                "type": "test",
                "status": "present",
                "evidence_text": "Patient reports persistent cough and fever",
                "value": None,
                "test_classification": "finding",
                "kg_finding_text": "persistent cough",
            }
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == ["persistent cough"]
    assert "Patient reports persistent cough and fever" not in result.symptoms
    assert "kg_finding_text" not in result.model_dump()
    assert "kg_finding_text" not in result.entities[0].model_dump()


def test_ungrounded_kg_finding_text_is_not_projected():
    text = "The patient has leukocytosis."
    payload = {
        "entities": [
            {
                "text": "leukocytosis",
                "type": "test",
                "status": "present",
                "evidence_text": "leukocytosis",
                "value": None,
                "test_classification": "finding",
                "kg_finding_text": "thrombocytosis",
            }
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == ["leukocytosis"]
    assert result.removed_by_grounding_check == 0


def test_numeric_measurements_survive_validation_and_response_construction():
    text = (
        "Temperature is 38.0°C (100.4°F), pulse is 112/min, blood pressure is "
        "150/90 mm Hg, and PTT is 43 seconds."
    )
    measurements = [
        ("temperature", "symptom", "temperature is 38.0°C (100.4°F)", "38.0°C (100.4°F)"),
        ("pulse", "symptom", "pulse is 112/min", "112/min"),
        ("blood pressure", "symptom", "blood pressure is 150/90 mm Hg", "150/90 mm Hg"),
        ("PTT", "test", "PTT is 43 seconds", "43 seconds"),
    ]
    payload = {
        "entities": [
            {
                "text": entity_text,
                "type": entity_type,
                "status": "present",
                "evidence_text": evidence_text,
                "value": value,
                **(
                    {
                        "test_classification": "finding",
                        "kg_finding_text": "PTT is 43 seconds",
                    }
                    if entity_type == "test"
                    else {}
                ),
            }
            for entity_text, entity_type, evidence_text, value in measurements
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert [(entity.text, entity.value) for entity in result.entities] == [
        (entity_text, value) for entity_text, _, _, value in measurements
    ]
    assert result.symptoms == [
        "temperature",
        "pulse",
        "blood pressure",
        "ptt is 43 seconds",
    ]
    assert result.tests == ["ptt"]
    assert "kg_finding_text" not in result.model_dump_json()


def test_json_numeric_measurement_values_are_converted_to_public_strings():
    text = (
        "Pulse is 112/min, temperature is 38.0°C (100.4°F), "
        "and PTT is 43 seconds."
    )
    payload = {
        "entities": [
            {
                "text": "pulse",
                "type": "symptom",
                "status": "present",
                "evidence_text": "pulse is 112/min",
                "value": 112,
            },
            {
                "text": "temperature",
                "type": "symptom",
                "status": "present",
                "evidence_text": "temperature is 38.0°C (100.4°F)",
                "value": 38.0,
            },
            {
                "text": "PTT",
                "type": "test",
                "status": "present",
                "evidence_text": "PTT is 43 seconds",
                "value": 43,
            },
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert [(entity.text, entity.value) for entity in result.entities] == [
        ("pulse", "112/min"),
        ("temperature", "38.0°C (100.4°F)"),
        ("PTT", "43 seconds"),
    ]
    assert result.removed_by_grounding_check == 0


def test_qualified_test_projection_keeps_present_absent_conflict_suppressed():
    text = "The patient has moderate proteinuria but no proteinuria."
    payload = {
        "entities": [
            {
                "text": "proteinuria",
                "type": "test",
                "status": "present",
                "evidence_text": "moderate proteinuria",
                "value": "moderate",
                "test_classification": "finding",
            },
            {
                "text": "proteinuria",
                "type": "test",
                "status": "absent",
                "evidence_text": "no proteinuria",
                "value": None,
                "test_classification": "finding",
            },
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == []
    assert [(entity.text, entity.status) for entity in result.entities] == [
        ("proteinuria", "present"),
        ("proteinuria", "absent"),
    ]


def test_medqa_month_age_and_death_event_are_preserved_without_autopsy_value():
    text = (
        "A 3-month-old baby dies suddenly during sleep, and the autopsy does not "
        "determine the cause. Which precaution could have prevented the death?"
    )
    payload = {
        "entities": [
            {
                "text": "autopsy",
                "type": "test",
                "status": "present",
                "evidence_text": "the autopsy",
                "body_site": None,
                "duration": None,
                "severity": None,
                "value": None,
            }
        ],
        "clinical_events": [
            {
                "text": "sudden death during sleep",
                "status": "present",
                "evidence_text": "dies suddenly during sleep",
            }
        ],
        "demographics": {
            "age": "3-month-old",
            "age_unit": None,
            "sex": None,
            "pregnancy_status": None,
            "gestational_age": None,
        },
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.demographics.age == 3
    assert result.demographics.age_unit == "months"
    assert len(result.clinical_events) == 1
    assert result.clinical_events[0].text == "sudden death during sleep"
    assert result.clinical_events[0].evidence_text == "dies suddenly during sleep"
    autopsy = next(entity for entity in result.entities if entity.text == "autopsy")
    assert autopsy.value is None
    assert all("Which precaution" not in event.evidence_text for event in result.clinical_events)
    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "For a procedure with no stated result, use null" in prompt
    assert "Do not shorten, paraphrase, or drop clinically meaningful parts of a finding" in prompt
    assert "Use the complete source finding phrase as the entity text" in prompt
    assert "kg_finding_text" in prompt
    assert "measurement value and its unit as a string" in prompt
    assert "Keep laboratory measurements and diagnostic test findings typed as test" in prompt


def test_absent_and_possible_findings_are_kept_and_ungrounded_event_is_removed():
    payload = {
        "entities": [
            {
                "text": "fever",
                "type": "symptom",
                "status": "absent",
                "evidence_text": "denies fever",
            },
            {
                "text": "pneumonia",
                "type": "disease",
                "status": "possible",
                "evidence_text": "possible pneumonia",
            },
        ],
        "clinical_events": [
            {
                "text": "death",
                "status": "present",
                "evidence_text": "death occurred",
            }
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(
        "Patient denies fever; possible pneumonia."
    )

    assert result.status == "success"
    assert [(entity.text, entity.status) for entity in result.entities] == [
        ("fever", "absent"),
        ("pneumonia", "possible"),
    ]
    assert result.clinical_events == []
    assert result.symptoms == []
    assert result.removed_by_grounding_check == 1


def test_null_valued_finding_is_projected_but_procedure_disease_and_event_are_not():
    text = (
        "The patient has leukocytosis and calf muscle pseudohypertrophy. MRI was performed. "
        "Systemic lupus erythematosus was diagnosed. The patient died after a premature birth."
    )
    payload = {
        "entities": [
            {
                "text": "leukocytosis",
                "type": "test",
                "status": "present",
                "evidence_text": "leukocytosis",
                "value": None,
                "test_classification": "finding",
            },
            {
                "text": "calf muscle pseudohypertrophy",
                "type": "symptom",
                "status": "present",
                "evidence_text": "calf muscle pseudohypertrophy",
            },
            {
                "text": "MRI",
                "type": "test",
                "status": "present",
                "evidence_text": "MRI was performed",
                "value": None,
                "test_classification": "procedure",
            },
            {
                "text": "systemic lupus erythematosus",
                "type": "disease",
                "status": "present",
                "evidence_text": "Systemic lupus erythematosus was diagnosed",
            },
        ],
        "clinical_events": [
            {
                "text": "death",
                "status": "present",
                "evidence_text": "The patient died",
            },
            {
                "text": "premature birth",
                "status": "present",
                "evidence_text": "premature birth",
            }
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == ["leukocytosis", "calf muscle pseudohypertrophy"]
    assert result.tests == ["leukocytosis", "mri"]
    assert result.diseases == ["systemic lupus erythematosus"]
    assert [event.text for event in result.clinical_events] == ["death", "premature birth"]
    assert all("test_classification" not in entity.model_dump() for entity in result.entities)


def test_absent_and_possible_classified_test_findings_are_not_projected():
    text = "No leukocytosis was present, and elevated ESR is possible."
    payload = {
        "entities": [
            {
                "text": "leukocytosis",
                "type": "test",
                "status": "absent",
                "evidence_text": "No leukocytosis was present",
                "value": None,
                "test_classification": "finding",
            },
            {
                "text": "elevated ESR",
                "type": "test",
                "status": "possible",
                "evidence_text": "elevated ESR is possible",
                "value": None,
                "test_classification": "finding",
            },
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == []
    assert result.tests == []
    assert [(entity.text, entity.status) for entity in result.entities] == [
        ("leukocytosis", "absent"),
        ("elevated ESR", "possible"),
    ]


def test_missing_or_invalid_test_classification_does_not_project_to_symptoms():
    text = (
        "ESR was elevated, ANCA was positive, MRI was performed, WBC count was high, "
        "and ECG was performed."
    )
    payload = {
        "entities": [
            {
                "text": "elevated ESR",
                "type": "test",
                "status": "present",
                "evidence_text": "ESR was elevated",
                "value": "elevated",
            },
            {
                "text": "ANCA positivity",
                "type": "test",
                "status": "present",
                "evidence_text": "ANCA was positive",
                "value": "positive",
                "test_classification": "test_result",
            },
            {
                "text": "MRI",
                "type": "test",
                "status": "present",
                "evidence_text": "MRI was performed",
                "value": "performed",
                "test_classification": "procedure",
            },
            {
                "text": "leukocytosis",
                "type": "test",
                "status": "present",
                "evidence_text": "WBC count was high",
                "value": None,
                "test_classification": "unknown",
            },
            {
                "text": "ECG",
                "type": "test",
                "status": "present",
                "evidence_text": "ECG was performed",
                "value": None,
            },
            {
                "text": "raised CRP",
                "type": "test",
                "status": "present",
                "evidence_text": "ESR was elevated",
                "value": "markedly raised",
            },
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == []
    assert result.tests == [
        "elevated esr",
        "anca positivity",
        "mri",
        "leukocytosis",
        "ecg",
        "raised crp",
    ]
    assert result.removed_by_grounding_check == 0


def test_missing_classification_remains_distinct_from_invalid_classification():
    from agent import _validate_extracted_entity

    missing = _validate_extracted_entity(
        {
            "text": "ECG",
            "type": "test",
            "status": "present",
            "evidence_text": "ECG performed",
        }
    )
    invalid = _validate_extracted_entity(
        {
            "text": "ECG",
            "type": "test",
            "status": "present",
            "evidence_text": "ECG performed",
            "test_classification": "unknown",
        }
    )

    assert missing is not None and missing.test_classification_missing
    assert invalid is not None and not invalid.test_classification_missing
    assert missing.test_classification is None
    assert invalid.test_classification is None


def test_explicit_null_classification_does_not_project_to_symptoms():
    text = "CRP was elevated."
    payload = {
        "entities": [
            {
                "text": "elevated CRP",
                "type": "test",
                "status": "present",
                "evidence_text": "CRP was elevated",
                "value": "elevated",
                "test_classification": None,
            }
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == []
    assert result.tests == ["elevated crp"]
    assert len(result.entities) == 1
    assert result.entities[0].value == "elevated"
    assert "test_classification" not in result.entities[0].model_dump()


def test_plain_ecg_procedure_is_not_projected_to_symptoms():
    text = "ECG performed."
    payload = {
        "entities": [
            {
                "text": "ECG",
                "type": "test",
                "status": "present",
                "evidence_text": "ECG performed",
                "value": None,
                "test_classification": "procedure",
            }
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(text)

    assert result.status == "success"
    assert result.symptoms == []
    assert result.tests == ["ecg"]
    assert [(entity.text, entity.status) for entity in result.entities] == [
        ("ECG", "present")
    ]


def test_grounded_finding_classification_does_not_bypass_entity_grounding():
    payload = {
        "entities": [
            {
                "text": "leukocytosis",
                "type": "test",
                "status": "present",
                "evidence_text": "the patient has leukocytosis",
                "value": None,
                "test_classification": "finding",
            }
        ],
        "demographics": {},
    }
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
    ]

    result = ClinicalTextProcessingAgent(api_key="test-key", client=client).process(
        "The patient has an elevated white blood cell count."
    )

    assert result.status == "success"
    assert result.symptoms == []
    assert result.entities == []
    assert result.removed_by_grounding_check == 1


def test_normalize_entity_name():
    assert normalize_entity_name("  chest   pain  ") == "chest pain"
    assert normalize_entity_name(None) == ""
