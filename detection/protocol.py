# The MIT License (MIT)
 # Copyright © 2024 It's AI 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pydantic
from typing import List, Optional
import bittensor as bt


class TextSynapse(bt.Synapse):
    """
    A protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.

    Attributes:
    - texts: List of texts that needs to be evaluated for AI generation
    - predictions: List of probabilities in response to texts

    """

    texts: List[str] = pydantic.Field(
        ...,
        title="Texts",
        description="A list of texts to check. Immuatable.",
        allow_mutation=False,
    )

    predictions: List[float] = pydantic.Field(
        ...,
        title="Predictions",
        description="List of predicted probabilities. This attribute is mutable and can be updated.",
    ) 

    def deserialize(self) -> float:
        """
        Deserialize output. This method retrieves the response from
        the miner in the form of self.text, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - List[float]: The deserialized response, which in this case is the list of preidictions.
        """
        return self
