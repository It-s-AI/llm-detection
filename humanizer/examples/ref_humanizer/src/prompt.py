"""The instruction shared by training and inference: the model sees exactly this at run time."""

PROMPT = (
    "Rewrite the following {genre} so that it reads as if a person wrote it. Keep every fact, name, number "
    "and the order of ideas; add nothing and drop nothing. Output only the rewritten text.\n\n{text}"
)


def build_prompt(text: str, genre: str | None) -> str:
    return PROMPT.format(genre=(genre or "text").replace("_", " "), text=text.strip())
