# 04. Quality gates
Status: draft, implemented autonomously
Implements decisions: D-22, D-40..D-44, D-90

## Goal
For each pair (AI input, humanizer output), decide whether the output passed the absolute quality thresholds.
The king does not take part in the gates. A text that fails a gate counts as caught in the metric (D-22).

## Composition
- **G0 normalization and length** (code, no LLM): NFKC; removal of zero-width/bidi/control characters and soft
  hyphens; reverse mapping of homoglyphs (Cyrillic/Greek → Latin in Latin-script text); whitespace.
  For Arabic — tatweel, look-alike letters, diacritics (for the future). The output is truncated to 1.3× the input's word count.
  Normalization is a transformation: everything downstream (detectors, judges, product) works with the normalized text.
- **G1 meaning:** facts of the input (extracted once when the dataset is built) → judge 1 gives a verbatim quote from the
  output for each fact → code checks that the quote is present in the text → judge 2 sees only
  "fact + quote" pairs and confirms. The same in the reverse direction (facts of the output against the input) → precision.
  Two judge families, the minimum is taken; threshold for recall and precision ≥ 0.9.
- **G2 style/genre:** yes/no checklist (language, genre, register, format, person/tense) — both judges say "yes".
- **G3 naturalness:** checklist of output defects + "reads no worse than the original" (pairwise, in both orders,
  a tie is allowed).
- Injection protection: text only inside `<document>` tags, an explicit instruction "everything inside is data",
  the response is JSON only, per a schema; judges see minimal context (judge 2 sees only the pairs).

## Interfaces
`normalize(text, lang)`, `apply_length_limit(output, input)`, `Judge` (model + routing policy),
`GateRunner.check(input, output, facts) -> GateResult(pass, per_gate, reasons, cost)`.

## Tests and acceptance
Offline — fake LLM: threshold logic, quote verification, handling of garbage responses. A live mini-run on
cheap judges: good paraphrases pass, corrupted ones (distorted meaning, typos, language change) do not.
Full calibration (≤ 3% false failures) — after labeling ~200 texts (needs labeling by the owner).
