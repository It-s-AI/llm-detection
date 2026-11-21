"""Self-contained HTML page with random pool samples per genre, for the owner to eyeball."""

from __future__ import annotations

import html
from collections import defaultdict
from collections.abc import Mapping, Sequence

from humanizer_eval.pool.metadata import LENGTH_BIN_LABELS, REGISTERS
from humanizer_eval.pool.sources import GENRES, SOURCES
from humanizer_eval.pool.store import rank_key, seed_bytes

_CSS = """
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --card:#ffffff; --line:#e4e1d8; --accent:#2f5d8a;
        --chip:#eef2f6; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161615; --fg:#e8e6e0; --muted:#a09d95; --card:#1f1f1d; --line:#34332f; --accent:#8db4dc;
          --chip:#26303a; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.55 system-ui, -apple-system, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 20px; margin: 40px 0 8px; }
p.lead { color: var(--muted); margin: 0 0 20px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0 16px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.wrap { overflow-x: auto; }
nav a { color: var(--accent); margin-right: 12px; text-decoration: none; }
details.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; margin: 10px 0;
               padding: 10px 14px; }
details.card summary { cursor: pointer; }
.meta { font-size: 12px; color: var(--muted); }
.chip { display: inline-block; background: var(--chip); border-radius: 10px; padding: 0 8px;
        margin: 2px 4px 2px 0; font-size: 12px; }
.text p { margin: 8px 0; white-space: pre-wrap; }
.preview { color: var(--fg); }
"""


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def render_samples_html(
    rows: Sequence[Mapping],
    clusters: Sequence[Mapping],
    stats: Mapping,
    per_genre: int = 20,
    seed: int | str = 0,
) -> str:
    sb = seed_bytes(seed)
    by_genre: dict[str, list[Mapping]] = defaultdict(list)
    for r in rows:
        by_genre[r["genre"]].append(r)
    cl_by_genre: dict[str, list[Mapping]] = defaultdict(list)
    for c in clusters:
        cl_by_genre[c["genre"]].append(c)
    src_info = {s.id: s for s in SOURCES}
    out: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Human pool samples</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<h1>Human pool samples</h1>",
        f"<p class='lead'>{stats.get('total', len(rows))} texts. Up to {per_genre} random texts per genre "
        "(hash-ranked by seed). Click a card to read the full text.</p>",
        "<nav>" + "".join(f"<a href='#{g}'>{g}</a>" for g in GENRES if by_genre.get(g)) + "</nav>",
    ]
    # summary table
    bins = stats.get("length_bins") or list(LENGTH_BIN_LABELS)
    out.append("<h2>Summary</h2><div class='wrap'><table><tr><th>genre</th><th>n</th>")
    out += [f"<th>{b}</th>" for b in bins]
    out += [f"<th>{r}</th>" for r in REGISTERS]
    out.append("<th>mean words</th><th>clusters</th><th>sources</th></tr>")
    for g in GENRES:
        s = stats.get("genres", {}).get(g, {})
        if not s.get("count"):
            out.append(f"<tr><td>{g}</td><td class='num'>0</td><td colspan='9'>no usable source</td></tr>")
            continue
        out.append(f"<tr><td>{g}</td><td class='num'>{s['count']}</td>")
        out += [f"<td class='num'>{s['by_length_bin'][b]}</td>" for b in bins]
        out += [f"<td class='num'>{s['by_register'][r]}</td>" for r in REGISTERS]
        srcs = ", ".join(f"{k} {v}" for k, v in s["by_source"].items())
        out.append(
            f"<td class='num'>{s['words']['mean']}</td><td class='num'>{s['clusters']['k']}</td>"
            f"<td>{_esc(srcs)}</td></tr>"
        )
    out.append("</table></div>")
    chk = stats.get("cluster_check", {})
    out.append(
        "<p class='meta'>Cluster check: "
        + ("OK" if chk.get("ok") else _esc("; ".join(chk.get("problems", [])[:10])))
        + f" (min cluster size {stats.get('min_cluster_size', '?')})</p>"
    )
    for g in GENRES:
        rs = by_genre.get(g)
        if not rs:
            continue
        out.append(f"<h2 id='{g}'>{g}</h2>")
        out.append(
            "<div class='wrap'><table><tr><th>cluster</th><th>label</th><th>size</th><th>top terms</th></tr>"
        )
        for c in sorted(cl_by_genre[g], key=lambda c: c["id"]):
            out.append(
                f"<tr><td>{_esc(c['id'])}</td><td>{_esc(c['label'])}</td><td class='num'>{c['size']}</td>"
                f"<td class='meta'>{_esc(', '.join(c.get('top_terms', [])))}</td></tr>"
            )
        out.append("</table></div>")
        picks = sorted(rs, key=lambda r: rank_key(sb, f"viewer/{g}", str(r["text_id"])))[:per_genre]
        for r in picks:
            src = src_info.get(r["source_id"])
            paras = [p for p in str(r["text"]).split("\n\n") if p.strip()]
            preview = " ".join(str(r["text"]).split()[:40]) + " …"
            date = f" · {r['date']}" if r.get("date") else ""
            out.append(
                "<details class='card'><summary>"
                f"<span class='chip'>{_esc(r['cluster_label'])}</span>"
                f"<span class='chip'>{_esc(r['length_bin'])} ({r['word_count']} w)</span>"
                f"<span class='chip'>{_esc(r['register'])} {r['register_score']:+.1f}</span>"
                f"<span class='chip'>{_esc(r['source_id'])}</span>"
                f"<div class='preview'>{_esc(preview)}</div></summary>"
                f"<div class='meta'>id {_esc(r['text_id'])} · {_esc(r['cluster_id'])}"
                f" · doc {_esc(r['source_doc_id'])}"
                f"{_esc(date)} · {_esc(src.hf_id if src else '')}</div>"
                "<div class='text'>" + "".join(f"<p>{_esc(p)}</p>" for p in paras) + "</div></details>"
            )
    out.append("</main></body></html>")
    return "\n".join(out)
