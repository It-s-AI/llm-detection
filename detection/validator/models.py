from pydantic import BaseModel


class ValDataRow(BaseModel):
    text: str
    label: bool
    prompt: str | None = None
    data_source: str | None = None
    model_name: str | None = None
    model_params: dict | None = None

    persona_profile: str | None = None
    persona_mood: str | None = None
    person_ton: str | None = None
    task: str | None = None

    augmentations: list[str] = []

