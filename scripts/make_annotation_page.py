"""Build a self-contained HTML page for the owner to label (source, rewrite) pairs.

Labels are used ONCE to choose and calibrate the judges (D-03, D-44); they never score submissions.

Input JSONL rows: {"id": ..., "source": ..., "rewrite": ...}
Usage: .venv/bin/python scripts/make_annotation_page.py pairs.jsonl var/annotate.html
The page stores progress in the browser and exports labels as JSON.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

QUESTIONS = [
    ("meaning", "Is the meaning fully kept (nothing lost, nothing added)?"),
    ("style", "Same language, genre, register and format?"),
    ("natural", "Is the text natural and correct (no inserted errors, odd words, cut-off sentences, junk)?"),
    ("not_worse", "Does it read at least as well as the source?"),
]

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Gate annotation</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6b7280;--card:#f6f7f9;--accent:#2563eb;--border:#e5e7eb}
@media (prefers-color-scheme: dark){:root{--bg:#111318;--fg:#e8e8ea;--muted:#9aa0aa;--card:#1a1d24;--accent:#60a5fa;--border:#2a2f3a}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:800px){.pair{grid-template-columns:1fr}}
.box{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px;white-space:pre-wrap}
h2{font-size:13px;color:var(--muted);margin:0 0 6px}.q{margin:10px 0}.q label{margin-right:14px}
nav{display:flex;gap:8px;align-items:center;margin:12px 0}button{padding:8px 14px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);color:#fff;border:none}#prog{color:var(--muted)}
</style></head><body><main>
<h1 style="font-size:20px">Annotation: source → rewritten text</h1>
<nav><button onclick="go(-1)">← Back</button><button class="primary" onclick="go(1)">Next →</button>
<span id="prog"></span><span style="flex:1"></span><button onclick="exportJson()">Download annotations (JSON)</button></nav>
<div class="pair"><div><h2>SOURCE</h2><div class="box" id="src"></div></div>
<div><h2>REWRITTEN TEXT</h2><div class="box" id="rw"></div></div></div>
<div id="qs"></div><div class="q"><label>Comment: <input id="note" style="width:70%"></label></div>
</main><script>
const ITEMS = __ITEMS__; const QS = __QS__; const KEY = "humeval-annot-v1";
let labels = {}; try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let i = 0;
function save(){ try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function render(){
  const it = ITEMS[i]; document.getElementById("src").textContent = it.source;
  document.getElementById("rw").textContent = it.rewrite;
  const cur = labels[it.id] || {};
  document.getElementById("qs").innerHTML = QS.map(([k, t]) => `<div class="q"><b>${t}</b><br>` +
    ["yes","no","unsure"].map(v => `<label><input type="radio" name="${k}" value="${v}" ${cur[k]===v?"checked":""}
     onchange="setL('${k}', this.value)"> ${v}</label>`).join("") + `</div>`).join("");
  document.getElementById("note").value = cur.note || "";
  document.getElementById("note").oninput = e => setL("note", e.target.value);
  const done = Object.keys(labels).filter(id => QS.every(([k]) => labels[id][k])).length;
  document.getElementById("prog").textContent = `${i + 1} / ${ITEMS.length} · fully annotated: ${done}`;
}
function setL(k, v){ const id = ITEMS[i].id; labels[id] = {...(labels[id] || {}), [k]: v}; save(); render(); }
function go(d){ i = Math.max(0, Math.min(ITEMS.length - 1, i + d)); render(); }
function exportJson(){ const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(labels, null, 1)], {type: "application/json"}));
  a.download = "annotations.json"; a.click(); }
render();
</script></body></html>"""


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    items = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    items = [{"id": str(x["id"]), "source": x["source"], "rewrite": x["rewrite"]} for x in items]
    page = PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__QS__", json.dumps(QUESTIONS, ensure_ascii=False))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(page)
    print(f"wrote {dst} with {len(items)} items; {html.escape('open it in a browser')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
