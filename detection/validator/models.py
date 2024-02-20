from pydantic import BaseModel


class ValDataRow(BaseModel):
    text: str
    label: bool
    prompt: str | None = None
    data_source: str
    model_name: str | None = None