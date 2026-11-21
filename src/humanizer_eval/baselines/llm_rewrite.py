"""Baseline humanizer: one LLM rewrite pass with a 'sound human' instruction.

Owner tool for pilots: inputs contain hidden period data -> HIDDEN routing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from humanizer_eval.llm.openrouter import HIDDEN

PROMPT = (
    "Rewrite the following text so that it reads as if a real person wrote it: vary sentence length, use "
    "natural word choices, avoid stock phrases and overly polished transitions. Keep every fact, the "
    "meaning, "
    "the language, the genre, the format and roughly the same length. Output only the rewritten text.\n\n"
    "<text>\n{text}\n</text>"
)


def rewrite_all(
    llm, model: str, texts: list[str], workers: int = 8, reasoning=None
) -> tuple[list[str], float]:
    def one(t: str) -> tuple[str, float]:
        try:
            r = llm.chat(
                model,
                [{"role": "user", "content": PROMPT.format(text=t)}],
                max_tokens=3000,
                temperature=0.9,
                policy=HIDDEN,
                purpose="baseline:rewrite",
                reasoning=reasoning,
            )
            return r.text.strip(), r.cost_usd
        except Exception:
            return "", 0.0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, texts))
    return [r[0] for r in res], sum(r[1] for r in res)
