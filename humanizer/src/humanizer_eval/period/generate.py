"""AI-input generation for a period (D-51..D-53).

Inputs depend on the human pool ONLY via public slot metadata (genre, topic-cluster
label, length bin, register). Concrete topics are invented by a topic model that
never sees pool texts. Prompts imitate how real users ask chat assistants.
All calls touch hidden period data -> HIDDEN routing.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from humanizer_eval.llm.openrouter import HIDDEN, ChatResult

TOPIC_SCHEMA = {
    "title": "topics",
    "type": "object",
    "additionalProperties": False,
    "required": ["topics"],
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["topic", "audience", "stance"],
                "properties": {
                    "topic": {"type": "string"},
                    "audience": {"type": "string"},
                    "stance": {"type": "string"},
                },
            },
        }
    },
}

TOPIC_SYSTEM = (
    "You propose writing topics. Each topic must be concrete and specific, different from the others, "
    "plausible for the given genre and subject area, and must not name real private individuals. "
    "Reply with JSON only."
)

#: User-like requests per genre. Template counts follow the production mix inside a genre (Q-2, owner's
#: measurement 2026-11-09): a template is drawn uniformly, so e.g. 6 of 7 academic templates ask for a
#: section of a paper (prod: 85% sections, 12% abstracts) and essays are ~50% expository, ~20%
#: argumentative, ~20% narrative/personal, ~10% literary analysis.
GENRE_TEMPLATES: dict[str, list[str]] = {
    "essay": [
        # expository
        "Write a {words}-word essay explaining {topic}. Main idea: {stance}.",
        "I need a {words} word essay for school that explains {topic} to {audience}.",
        "write an informative essay (about {words} words) on {topic}, focus on causes and effects",
        "Write a compare-and-contrast essay of about {words} words on {topic}.",
        "Can you write a {words}-word expository essay about {topic}? Key point: {stance}.",
        # argumentative
        "Write a {words}-word essay on this topic: {topic}. My position: {stance}.",
        "write an essay about {topic} for {audience}, around {words} words, arguing that {stance}",
        # narrative / personal
        "Write a personal narrative essay (~{words} words) about my own experience: {topic}.",
        "write a {words} word reflective essay about my experience with {topic}",
        # literary analysis
        "Write a literary analysis essay of about {words} words on {topic}. Thesis: {stance}.",
    ],
    "article_blog": [
        "Write a blog post of about {words} words about {topic} for {audience}. Angle: {stance}.",
        "Can you write a short article (~{words} words) on {topic}? Readers: {audience}. Main point: {stance}.",
    ],
    "academic": [
        "Write the introduction section of my research paper on {topic}, about {words} words. Key claim: {stance}.",
        "write a literature review paragraph (~{words} words) on {topic} for my thesis",
        "Draft the methods section (about {words} words) for a study on {topic}.",
        "Write the discussion section of a paper on {topic}, ~{words} words. Main finding: {stance}.",
        "Write 2-3 paragraphs (about {words} words) for the background section of my dissertation on {topic}.",
        "write the results and discussion part (~{words} words) of a report on {topic} for {audience}",
        "Write an academic abstract of about {words} words for a paper on {topic}. Key claim: {stance}.",
    ],
    "news": [
        "Write a news article of about {words} words reporting on: {topic}. Include an angle: {stance}.",
        "write a short local news story (~{words} words) about {topic}",
    ],
    "email_letter": [
        "Write an email of about {words} words to {audience} about {topic}. Point to make: {stance}.",
        "Help me write a letter (~{words} words) to {audience} regarding {topic}. I want to say {stance}.",
    ],
    "review": [
        "Write a {words}-word customer review about {topic}. Overall opinion: {stance}.",
        "write a review (about {words} words) of {topic} from the point of view of {audience}",
    ],
    "forum_qa": [
        "Write a forum reply of about {words} words answering someone who asked about {topic}. Advice: {stance}.",
        "Answer this question like on a Q&A site, ~{words} words: {topic}",
    ],
    "creative": [
        "Write a short story of about {words} words about {topic}.",
        "write a {words} word flash fiction piece: {topic}",
    ],
    "application": [
        "Write my personal statement (~{words} words) for {audience}. It should be about {topic}. Key message: {stance}.",
        "help me write a {words}-word college application essay about {topic}",
        "Write a cover letter of about {words} words for a position related to {topic}. Emphasize: {stance}.",
        "Write a scholarship essay (~{words} words) on {topic} for {audience}.",
    ],
    "business_marketing": [
        "Write a product description of about {words} words for {topic}, aimed at {audience}.",
        "write website copy (~{words} words) for a business offering {topic}. Selling point: {stance}.",
        "Draft a short internal memo (about {words} words) about {topic}. Key point: {stance}.",
        "Write a {words}-word section of a business report on {topic} for {audience}.",
    ],
    "homework_answer": [
        "Answer this homework question in about {words} words: {topic}",
        "explain {topic} in ~{words} words for my class assignment",
        "Q: {topic}. Write a clear answer of about {words} words for {audience}.",
    ],
}
REGISTER_HINT = {"formal": " Use a formal tone.", "informal": " Keep it casual."}

_PREAMBLE = re.compile(r"^(sure|certainly|of course|here(?:'s| is| are)|absolutely)\b[^\n]*[:!.]\s*\n+", re.I)
_TITLE = re.compile(r"^(#{1,6}\s+[^\n]+|\*\*[^\n]{1,120}\*\*|title:[^\n]+)\s*\n+", re.I)
_OUTRO = re.compile(r"\n+[^\n]*(let me know|hope this helps|feel free to|i hope you)[^\n]*$", re.I)


def clean_generation(text: str) -> str:
    """Remove chat-assistant wrappers that users typically delete before using a text."""
    s = text.strip()
    for _ in range(2):
        s = _PREAMBLE.sub("", s).strip()
        s = _TITLE.sub("", s).strip()
    s = _OUTRO.sub("", s).strip()
    s = s.replace("**", "")
    return s


@dataclass
class GenConfig:
    generators: list[dict[str, Any]]
    genre_proportions: dict[str, float]
    length_words: dict[str, int]
    topic_model: str
    facts_model: str
    max_tokens: int = 1200
    #: two judge families that must both confirm every extracted fact on the input (G1 v2)
    fact_validators: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenConfig:
        return cls(
            **{
                k: d[k]
                for k in (
                    "generators",
                    "genre_proportions",
                    "length_words",
                    "topic_model",
                    "facts_model",
                    "max_tokens",
                    "fact_validators",
                )
                if k in d
            }
        )


@dataclass
class Generated:
    text: str
    model: str
    provider: str | None
    template: str
    target_words: int
    cost_usd: float
    attempts: int
    ok: bool
    notes: list[str] = field(default_factory=list)


def target_length(rng: random.Random, length_bin: str | None, cfg: GenConfig) -> int:
    if length_bin and "-" in length_bin:
        lo, hi = (int(x) for x in length_bin.split("-")[:2])
        return rng.randint(lo, hi)
    lw = cfg.length_words
    return max(lw["min"], min(lw["max"], int(rng.gauss(lw["mean"], 40))))


def pick_generator(rng: random.Random, cfg: GenConfig) -> str:
    models = [g["model"] for g in cfg.generators]
    weights = [g["weight"] for g in cfg.generators]
    return rng.choices(models, weights=weights, k=1)[0]


_DEFAULT_REASONING = {"effort": "low"}


def generator_settings(cfg: GenConfig, model: str) -> tuple[int, dict[str, Any] | None]:
    """Per-model overrides from the config: `max_tokens` and `reasoning` ("none" -> omit the parameter).
    Reasoning tokens count toward max_tokens, and providers differ in which settings they accept."""
    entry = next((g for g in cfg.generators if g["model"] == model), {})
    reasoning = entry.get("reasoning", _DEFAULT_REASONING)
    return int(entry.get("max_tokens", cfg.max_tokens)), (None if reasoning == "none" else reasoning)


def propose_topics(
    llm, model: str, genre: str, cluster_label: str, n: int, seed: int
) -> tuple[list[dict], float]:
    user = (
        f"Genre: {genre}\nSubject area: {cluster_label}\nVariation seed: {seed}\n"
        f"Propose {n} topics. For each give: topic (one line), audience, stance (the author's position/angle)."
    )
    r: ChatResult = llm.chat(
        model,
        [{"role": "system", "content": TOPIC_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=4000,
        temperature=1.0,
        json_schema=TOPIC_SCHEMA,
        policy=HIDDEN,
        purpose="period:topics",
        reasoning={"effort": "low"},
    )
    topics = (r.json or {}).get("topics", []) if isinstance(r.json, dict) else []
    return [t for t in topics if isinstance(t, dict) and t.get("topic")][:n], r.cost_usd


def generate_input(
    llm, rng: random.Random, cfg: GenConfig, slot: dict[str, Any], topic: dict[str, Any]
) -> Generated:
    genre = slot["genre"]
    template = rng.choice(GENRE_TEMPLATES.get(genre, GENRE_TEMPLATES["article_blog"]))
    words = target_length(rng, slot.get("length_bin"), cfg)
    prompt = template.format(
        words=words,
        topic=topic["topic"].rstrip("."),
        audience=topic.get("audience", "a general audience"),
        stance=topic.get("stance", "").rstrip("."),
    )
    prompt += REGISTER_HINT.get(slot.get("register", ""), "")
    model = pick_generator(rng, cfg)
    cost, notes = 0.0, []
    for attempt in range(1, 3):
        max_tokens, reasoning = generator_settings(cfg, model)
        try:
            r = llm.chat(
                model,
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=rng.choice([0.7, 1.0, 1.0]),
                policy=HIDDEN,
                purpose="period:generate",
                reasoning=reasoning,
            )
        except Exception as e:  # provider unavailable for this model -> try another generator once
            notes.append(f"{model}: {str(e)[:120]}")
            model = pick_generator(rng, cfg)
            continue
        cost += r.cost_usd
        text = clean_generation(r.text)
        n = len(text.split())
        if 0.7 * words <= n <= 1.4 * words:
            return Generated(text, r.model, r.provider, template, words, cost, attempt, True, notes)
        notes.append(f"length {n} vs target {words}")
        prompt += f" Keep it to about {words} words."
    return Generated("", model, None, template, words, cost, 2, False, notes)
