import pydantic
from typing import List, Optional
import bittensor as bt

# from detection import __version__

class TextRequest(pydantic.BaseModel):
    texts: List[str] = pydantic.Field(
        ...,
        title="Texts",
        description="A list of texts to check."
    )
    
    predictions: List[List[float]] = pydantic.Field(
        ...,
        title="Predictions",
        description="List of predicted probabilities."
    )
    
    version: str = pydantic.Field(default="", title="Version", description="Version of the request schema.")
