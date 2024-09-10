from pydantic import BaseModel


class ValDataRow(BaseModel):
    text: str
    text_auged: str | None = None
    label: bool
    segmentation_labels: list[bool]
    auged_segmentation_labels: list[bool]
    prompt: str | None = None
    data_source: str | None = None
    model_name: str | None = None
    model_params: dict | None = None
    topic: str | None = None

    augmentations: list[str] = []

