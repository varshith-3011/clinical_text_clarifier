import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_groq_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY")


def get_clinical_text_model() -> str:
    return os.getenv("CLINICAL_TEXT_MODEL", DEFAULT_MODEL)


def require_groq_api_key() -> str:
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your environment or .env file before calling the model."
        )
    return api_key


GROQ_API_KEY = get_groq_api_key()
CLINICAL_TEXT_MODEL = get_clinical_text_model()


def get_runtime_config_diagnostic() -> dict[str, bool | str]:
    return {
        "groq_api_key_present": bool(GROQ_API_KEY),
        "model": CLINICAL_TEXT_MODEL,
    }
