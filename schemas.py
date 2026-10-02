from typing import Optional

from pydantic import BaseModel, ConfigDict

VALID_ENTITY_TYPES = {"disease", "symptom", "medication", "test"}
VALID_ENTITY_STATUSES = {"present", "absent", "possible"}


class ClinicalInput(BaseModel):
    text: str


class Entity(BaseModel):
    text: str
    type: str
    status: str
    evidence_text: str
    body_site: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    value: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class Demographics(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class FinalStructuredResponse(BaseModel):
    clinical_text: str
    diseases: list[str] = []
    symptoms: list[str] = []
    medications: list[str] = []
    tests: list[str] = []
    entities: list[Entity] = []
    demographics: Demographics = Demographics(age=None, sex=None)
    removed_by_grounding_check: int = 0
    model: str = "openai/gpt-oss-120b"
    status: str = "success"
    error: Optional[str] = None

    model_config = ConfigDict(extra="ignore")
