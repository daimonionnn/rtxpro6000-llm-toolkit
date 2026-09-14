#!/usr/bin/env python3
"""Collect answers to a fixed set of non-English prompts, for comparing quantizations.

    python3 bench/language_samples.py collect LABEL [--lang sk] [--base URL]
    python3 bench/language_samples.py blind [LABEL ...] [--lang sk]

collect  sends every prompt of the language set to the running server (thinking
         off, temperature 0, one request at a time) and writes
         logs/language-samples/<lang>/<LABEL>.json. LABEL is a name for the
         profile under test, e.g. vllm-awq-w4a16.
blind    merges the collected files into one Markdown sheet with the answers to
         each prompt shuffled and labelled A, B, C..., plus a separate key file,
         so the answers can be judged without knowing which profile wrote them.

Benchmarks in English miss most of the damage quantization does to other
languages; reading the same prompts side by side does not. Greedy decoding keeps
a profile's answers repeatable. Standard library only.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs", "language-samples")

# Each prompt exercises something quantization tends to break first in a
# lower-resource language: inflection and agreement, register, idiom, spelling
# with diacritics, and keeping the language while handling English input.
PROMPTS = {
    "sk": [
        ("explain", "Vysvetli jednoduchými slovami, prečo je obloha modrá. Odpovedz v jednom odseku."),
        ("formal-email", "Napíš krátky formálny e-mail vedúcej oddelenia, v ktorom sa ospravedlníš za meškanie "
                         "projektu o dva týždne, uvedieš dôvod a navrhneš nový termín."),
        ("inflection", "Vytvor päť viet, v ktorých použiješ slovo „mesto“ postupne v inom páde "
                       "(genitív, datív, akuzatív, lokál, inštrumentál). Pri každej vete uveď pád."),
        ("numbers-agreement", "Napíš vetami, nie číslicami: 1 žena, 2 ženy, 5 žien, 21 mužov, 3 deti, 100 kníh "
                              "— vždy ako celú vetu s „V knižnici je/sú …“."),
        ("idioms", "Vysvetli význam týchto slovenských frazeologizmov a ku každému daj príklad použitia: "
                   "„mať maslo na hlave“, „hádzať hrach na stenu“, „robiť z komára somára“."),
        ("translate", "Prelož do prirodzenej slovenčiny, nie doslovne: \"The deadline has been pushed back, "
                      "so we finally have some breathing room to polish the release.\""),
        ("summary", "Zhrň v troch vetách po slovensky: Fotosyntéza je proces, pri ktorom rastliny, riasy a niektoré "
                    "baktérie premieňajú svetelnú energiu na chemickú. Z oxidu uhličitého a vody vzniká glukóza a "
                    "ako vedľajší produkt kyslík. Prebieha v chloroplastoch a je základom potravových reťazcov na Zemi."),
        ("code-explain", "Vysvetli po slovensky, čo robí tento kód a aký má problém:\n\n"
                         "def average(xs):\n    return sum(xs) / len(xs)"),
        ("story", "Napíš krátky príbeh (asi 120 slov) o starom rybárovi na Orave, s priamou rečou."),
        ("grammar-fix", "Oprav gramatické a pravopisné chyby a vysvetli každú opravu: "
                        "„Včera sme boly v obchode a kúpili sme si dva rožky a tri jablka, lebo sme mali "
                        "hlad a nevedeli sme čo budeme večerať.“"),
    ],
}


def collect(a):
    prompts = PROMPTS[a.lang]
    url = a.base.rstrip("/") + "/v1/chat/completions"
    results = []
    for pid, text in prompts:
        body = {"model": a.model, "messages": [{"role": "user", "content": text}],
                "max_tokens": a.max_tokens, "temperature": 0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        dt = time.time() - t0
        msg = r["choices"][0]["message"]
        answer = msg.get("content") or ""
        usage = r.get("usage") or {}
        results.append({"id": pid, "prompt": text, "answer": answer,
                        "finish_reason": r["choices"][0].get("finish_reason"),
                        "completion_tokens": usage.get("completion_tokens"), "seconds": round(dt, 2)})
        print(f"  {pid:<18} {usage.get('completion_tokens', '?'):>5} tok  {dt:6.1f} s", flush=True)
    os.makedirs(os.path.join(OUT, a.lang), exist_ok=True)
    path = os.path.join(OUT, a.lang, f"{a.label}.json")
    json.dump({"label": a.label, "lang": a.lang, "base": a.base,
               "collected": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results},
              open(path, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {path}")


def blind(a):
    d = os.path.join(OUT, a.lang)
    labels = a.labels or sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))
    runs = {lab: {r["id"]: r for r in json.load(open(os.path.join(d, lab + ".json")))["results"]}
            for lab in labels}
    rng = random.Random(a.seed)
    sheet = [f"# Blind comparison ({a.lang}, {len(labels)} answers per prompt)\n",
             "Letters are shuffled separately for every prompt; the key is in the matching `-key.md` file.\n"]
    key = ["# Key\n"]
    for pid, text in PROMPTS[a.lang]:
        order = [lab for lab in labels if pid in runs[lab]]
        rng.shuffle(order)
        sheet.append(f"\n## {pid}\n\n> {text}\n")
        key.append(f"\n## {pid}\n")
        for i, lab in enumerate(order):
            letter = chr(ord("A") + i)
            sheet.append(f"\n### {letter}\n\n{runs[lab][pid]['answer'].strip()}\n")
            key.append(f"- {letter}: {lab}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    sp, kp = os.path.join(d, f"blind-{stamp}.md"), os.path.join(d, f"blind-{stamp}-key.md")
    open(sp, "w").write("\n".join(sheet) + "\n")
    open(kp, "w").write("\n".join(key) + "\n")
    print(f"wrote {sp}\n      {kp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("label")
    c.add_argument("--lang", default="sk", choices=sorted(PROMPTS))
    c.add_argument("--base", default=os.environ.get("BASE", "http://127.0.0.1:8090"))
    c.add_argument("--model", default="Qwen3.8-Flash-Next")
    c.add_argument("--max-tokens", type=int, default=800)
    b = sub.add_parser("blind")
    b.add_argument("labels", nargs="*")
    b.add_argument("--lang", default="sk", choices=sorted(PROMPTS))
    b.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    return collect(a) if a.cmd == "collect" else blind(a)


if __name__ == "__main__":
    sys.exit(main())
