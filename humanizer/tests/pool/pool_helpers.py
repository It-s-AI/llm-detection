"""Synthetic English prose for offline pool tests (importable: pytest puts this directory on sys.path)."""

SENTENCES = [
    "The committee met on a rainy afternoon to discuss the new library budget.",
    "Several members argued that the reading room needed better lighting and quieter heating.",
    "Others pointed out that the children's section had not been renovated in many years.",
    "After a long debate, they agreed to ask local families what they wanted most.",
    "The survey was short, but it asked about opening hours, events and the collection.",
    "Most people wrote that they would visit more often if the library stayed open later.",
    "A few asked for a small cafe, while others wanted more space for quiet study.",
    "The librarian promised to publish the results before the next public meeting.",
    "Volunteers offered to help carry boxes when the old shelves are finally replaced.",
    "Everyone left the meeting feeling that the plan was modest but genuinely useful.",
]


def prose(n_words: int, offset: int = 0) -> str:
    """English prose of at least ``n_words`` words built from whole sentences."""
    out: list[str] = []
    i = offset
    while sum(len(s.split()) for s in out) < n_words:
        out.append(SENTENCES[i % len(SENTENCES)])
        i += 1
    return " ".join(out)


def doc_with_paragraphs(sizes: list[int]) -> str:
    return "\n\n".join(prose(n, offset=k) for k, n in enumerate(sizes))


_VOCAB = (
    "river mountain coffee engine garden letter winter market pencil castle doctor ticket silver window "
    "harbor violin forest bridge planet camera island rocket lantern meadow cookie canvas tunnel blanket "
    "compass orchard pebble saddle thunder velvet whistle anchor basket candle desert feather glacier "
    "hammer jacket kettle ladder mirror needle oyster parrot quilt ribbon shovel tablet umbrella wagon"
).split()


def distinct_doc(i: int, n_words: int = 200) -> str:
    """A document whose word sequence is unique to ``i`` (no near-duplicates between different i)."""
    import random

    rng = random.Random(i)
    words = [rng.choice(_VOCAB) for _ in range(n_words)]
    sents = [" ".join(words[k : k + 10]).capitalize() + "." for k in range(0, n_words, 10)]
    return " ".join(sents[: len(sents) // 2]) + "\n\n" + " ".join(sents[len(sents) // 2 :])
