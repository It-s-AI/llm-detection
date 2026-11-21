"""Baseline "humanizer" that returns the input unchanged (reference point for the metric)."""


def humanize(items):
    return [it["text"] for it in items]
