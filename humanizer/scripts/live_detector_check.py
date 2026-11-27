"""Live contract + determinism check for paid detectors (phase 1). Costs a few dollars.

Usage:  .venv/bin/python scripts/live_detector_check.py [--detectors gptzero,zerogpt,pangram,its_ai]

* 10 human texts (Pile paragraphs, written before 2021) + 10 AI texts (written by an LLM).
* Every text is scored twice WITHOUT cache -> determinism.
* First pass of the AI texts is recorded (secrets stripped) to tests/fixtures/recordings/.
* All paid calls go through the spend ledger (hard caps from .env).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.detectors.fake import RecordingTransport
from humanizer_eval.detectors.registry import build_detector
from humanizer_eval.paths import ledger_path

AI_TEXTS = [
    "Remote work has fundamentally transformed the modern workplace, offering employees unprecedented "
    "flexibility while presenting organizations with new challenges. On one hand, workers save time on "
    "commuting and can structure their days around peak productivity hours. On the other hand, managers "
    "must find innovative ways to maintain team cohesion and company culture without the benefit of shared "
    "physical spaces. Research suggests that hybrid models, which combine in-office collaboration with "
    "remote focus time, may offer the best of both worlds. Ultimately, the success of any arrangement "
    "depends on clear communication, well-defined expectations, and a commitment to trust between "
    "employers and their teams. As technology continues to evolve, the future of work will likely become "
    "even more dynamic and personalized.",
    "Climate change represents one of the most pressing challenges facing humanity in the twenty-first "
    "century. Rising global temperatures are driving more frequent extreme weather events, from devastating "
    "wildfires to unprecedented flooding. Addressing this crisis requires a multifaceted approach that "
    "includes transitioning to renewable energy sources, improving energy efficiency, and protecting "
    "natural ecosystems that absorb carbon dioxide. Individuals can contribute by reducing their carbon "
    "footprint through sustainable transportation choices and mindful consumption. However, meaningful "
    "progress ultimately depends on coordinated action by governments and businesses worldwide. By working "
    "together and embracing innovation, we can build a more resilient and sustainable future for "
    "generations to come.",
    "The Industrial Revolution marked a pivotal turning point in human history. Beginning in Britain in the "
    "late eighteenth century, it introduced mechanized production, steam power, and the factory system, "
    "which together reshaped economies and societies. Cities grew rapidly as people migrated from rural "
    "areas in search of work, leading to both new opportunities and significant social problems, including "
    "overcrowding and poor working conditions. Over time, the innovations of this era laid the foundation "
    "for modern manufacturing, transportation, and communication. Understanding the Industrial Revolution "
    "helps us appreciate how technological change can transform everyday life, and it offers valuable "
    "lessons as we navigate today's rapid digital transformation.",
    "Regular physical activity offers a wide range of benefits for both physical and mental health. "
    "Engaging in exercise strengthens the heart, improves circulation, and helps maintain a healthy body "
    "weight. Beyond these physical advantages, exercise also releases endorphins, which can reduce stress "
    "and improve mood. Experts recommend at least 150 minutes of moderate aerobic activity each week, "
    "combined with strength training exercises twice a week. Importantly, finding an activity you genuinely "
    "enjoy, whether it is dancing, swimming, or hiking, makes it far easier to stay consistent. By making "
    "movement a regular part of daily life, individuals can significantly enhance their overall well-being "
    "and quality of life.",
    "Social media has become an integral part of daily life for billions of people around the world. These "
    "platforms allow users to stay connected with friends and family, share experiences, and access news "
    "and information instantly. However, the rise of social media has also raised important concerns about "
    "privacy, misinformation, and its impact on mental health, particularly among young people. Studies have "
    "linked excessive use to feelings of anxiety and low self-esteem. To navigate these challenges, it is "
    "essential for users to develop digital literacy skills and practice mindful usage. Striking a healthy "
    "balance between online engagement and real-world interactions is key to harnessing the benefits of "
    "social media while minimizing its drawbacks.",
    "Artificial intelligence is rapidly reshaping industries across the globe, from healthcare and finance "
    "to education and entertainment. Machine learning algorithms can now analyze vast amounts of data, "
    "identify patterns, and make predictions with remarkable accuracy. In healthcare, for example, AI-powered "
    "tools assist doctors in diagnosing diseases earlier and more precisely. Nevertheless, the widespread "
    "adoption of AI also raises important ethical questions regarding bias, accountability, and the future "
    "of employment. As society continues to integrate these technologies, it will be crucial to establish "
    "thoughtful regulations and ensure that AI systems are developed responsibly, transparently, and in ways "
    "that benefit everyone.",
    "Reading books offers countless benefits that extend far beyond entertainment. When we immerse "
    "ourselves in a good book, we expand our vocabulary, improve our concentration, and develop greater "
    "empathy by experiencing the world through different perspectives. Fiction allows us to explore "
    "imaginative worlds, while nonfiction deepens our understanding of history, science, and human nature. "
    "Moreover, reading before bed can be a relaxing alternative to screen time, promoting better sleep. "
    "In an age of constant digital distractions, cultivating a regular reading habit can be a powerful way "
    "to nurture the mind, reduce stress, and foster lifelong learning and personal growth.",
    "Effective time management is a crucial skill for success in both academic and professional settings. "
    "By prioritizing tasks, setting clear goals, and breaking large projects into manageable steps, "
    "individuals can significantly boost their productivity. Tools such as calendars, to-do lists, and "
    "productivity apps can help keep track of deadlines and responsibilities. Additionally, techniques like "
    "the Pomodoro method encourage focused work sessions followed by short breaks, which can prevent burnout. "
    "It is also important to learn to say no to unnecessary commitments. Ultimately, mastering time "
    "management not only reduces stress but also creates more space for the activities and relationships "
    "that truly matter.",
    "Traveling to new places broadens our horizons and enriches our understanding of the world. Experiencing "
    "different cultures, cuisines, and traditions firsthand fosters a deeper appreciation for diversity and "
    "challenges our assumptions. Travel also encourages personal growth by pushing us out of our comfort "
    "zones and requiring us to adapt to unfamiliar situations. Whether exploring a bustling city or a quiet "
    "countryside village, each journey offers unique lessons and lasting memories. While travel can "
    "sometimes be expensive, budget-friendly options such as off-season trips and local exploration make it "
    "accessible to more people. In the end, the experiences we gather on the road often shape who we become.",
    "Healthy eating plays a vital role in maintaining overall well-being and preventing chronic diseases. "
    "A balanced diet rich in fruits, vegetables, whole grains, and lean proteins provides the essential "
    "nutrients the body needs to function optimally. Reducing the intake of processed foods, added sugars, "
    "and excessive salt can lower the risk of conditions such as heart disease and diabetes. Staying "
    "hydrated by drinking enough water throughout the day is equally important. Rather than following "
    "restrictive diets, experts recommend building sustainable habits, such as planning meals ahead and "
    "practicing mindful eating. Small, consistent changes can lead to significant long-term improvements in "
    "health.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default="gptzero,zerogpt,pangram,its_ai")
    ap.add_argument("--out", default="docs/reports/data/01-live-check.json")
    args = ap.parse_args()
    human = [r["text"] for r in json.loads(Path("var/live_human_texts.json").read_text())]
    texts = human + AI_TEXTS
    labels = ["human"] * len(human) + ["ai"] * len(AI_TEXTS)
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.detectors.split(","):
        rec_path = Path(f"tests/fixtures/recordings/{name}.jsonl")
        rec_path.unlink(missing_ok=True)  # never append duplicate recordings
        rec = RecordingTransport(rec_path)
        # pass 1: AI texts recorded (fixtures), human texts not recorded (not ours to republish)
        det_rec = build_detector(name, ledger=led, use_cache=False, transport=rec)
        det = build_detector(name, ledger=led, use_cache=False)
        t0 = time.time()
        run1 = det.score(human) + det_rec.score(AI_TEXTS)
        t1 = time.time()
        run2 = det.score(texts)
        t2 = time.time()
        errs = [x.error for x in run1 + run2 if not x.ok]
        diffs = [abs(a.score - b.score) for a, b in zip(run1, run2, strict=True) if a.ok and b.ok]
        h = [x.score for x, lab in zip(run1, labels, strict=True) if x.ok and lab == "human"]
        a = [x.score for x, lab in zip(run1, labels, strict=True) if x.ok and lab == "ai"]
        results[name] = {
            "ok": len(run1 + run2) - len(errs),
            "errors": errs[:5],
            "n_errors": len(errs),
            "version": next((x.version for x in run1 if x.version), None),
            "sec_per_text_pass1": round((t1 - t0) / len(texts), 2),
            "sec_per_text_pass2": round((t2 - t1) / len(texts), 2),
            "determinism_max_abs_diff": max(diffs) if diffs else None,
            "determinism_identical_share": (sum(d == 0 for d in diffs) / len(diffs)) if diffs else None,
            "mean_score_human": statistics.mean(h) if h else None,
            "mean_score_ai": statistics.mean(a) if a else None,
            "per_text": [
                {"label": lab, "run1": x.score, "run2": y.score, "candidates": x.candidates}
                for x, y, lab in zip(run1, run2, labels, strict=True)
            ],
        }
        print(name, {k: v for k, v in results[name].items() if k != "per_text"})
    results["_ledger"] = led.summary()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    print("ledger:", json.dumps(led.summary(), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
