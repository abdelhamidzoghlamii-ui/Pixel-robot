#!/usr/bin/env python3
"""Regrade copied raw turns and write derived tables inside this archive only."""
import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOMS = {"kitchen", "living_room", "schlafzimmer", "bureau", "bathroom", "balkon"}
EXPECTED = {
    "C1": ("navigate_to", "schlafzimmer"),
    "C2": ("navigate_to", "bureau"),
    "C3": ("navigate_to", "bureau"),
    "C4": ("navigate_to", "living_room"),
    "C5": ("navigate_to", "balkon"),
    "C6": ("REJECT",),
    "C7": ("REJECT",),
    "C8": ("REJECT",),
    "C9": ("find_object", "phone", "kitchen"),
    "C10": ("patrol",),
    "C11": ("find_person", "Abdel"),
    "C12": ("come_back",),
    "C13": ("find_person", "Chiara", "message"),
}
CROSS = {"C1", "C2", "C3", "C4", "C5", "C13"}
PROMPTS_AB = {
    "A1": "Hey, you awake?",
    "A2": "Wie geht's dir heute?",
    "A3": "Tu peux m'expliquer ce que tu fais quand tu te perds ?",
    "A4": "من أنت وماذا تستطيع أن تفعل؟",
    "A5": "I'm looking for something. / It's small and black. / So where should I start?",
    "A6": "I've had a rough day.",
    "A7": "Fahr sofort los und such Chiara.",
    "B1": "How far can you see?",
    "B2": "Where are my keys?",
    "B3": "Warum bleibst du stehen, wenn etwas vor dir ist?",
    "B4": "Combien de roues as-tu et comment tournes-tu sur place ?",
    "B5": "هل تستطيع أن ترى في الظلام؟",
    "B6": "What's the battery at?",
    "B7": "Explain in one sentence why you stopped.",
}


def parsed_reply(raw, lenient=False):
    """Mirror the harness's last parseable array extraction; allow a whole object optionally."""
    clean = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
    clean = re.sub(r"\x60\x60\x60(?:json)?", "", clean)
    decoder = json.JSONDecoder()
    if lenient:
        try:
            value = json.loads(clean)
            if isinstance(value, dict):
                return [value], True
        except (ValueError, TypeError):
            pass
    last = None
    braces = brackets = 0
    quoted = escaped = False
    for i, char in enumerate(clean):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            braces += 1
        elif char == "}":
            braces = max(0, braces - 1)
        elif char == "[":
            if braces == brackets == 0:
                try:
                    value, _ = decoder.raw_decode(clean[i:])
                    if isinstance(value, list):
                        last = value
                except ValueError:
                    pass
            brackets += 1
        elif char == "]":
            brackets = max(0, brackets - 1)
    if last is not None:
        return last, True
    try:
        value = json.loads(clean)
        if isinstance(value, list):
            return value, True
    except (ValueError, TypeError):
        pass
    return [], False


def grade(raw, prompt_id, lenient=False):
    actions, parsed = parsed_reply(raw, lenient)
    if not parsed:
        return dict(parse_ok=False, exact=False, malformed=False,
                    critical=False, reject_violation=False)
    malformed = any(not isinstance(a, dict) for a in actions)
    acts = [a for a in actions if isinstance(a, dict)]
    critical = any(a.get("type") == "navigate_to" and a.get("room") not in ROOMS
                   for a in acts)
    expected = EXPECTED[prompt_id]
    exact = False
    if expected[0] == "REJECT":
        exact = (not actions or
                 (len(actions) == 1 and not malformed and acts[0].get("type") == "say"))
        reject_violation = parsed and not exact
    elif not malformed:
        reject_violation = False
        for act in acts:
            kind = act.get("type")
            if kind != expected[0]:
                continue
            if kind == "navigate_to":
                exact = act.get("room") == expected[1]
            elif kind == "find_object":
                exact = (isinstance(act.get("object"), str)
                         and act["object"].lower() == expected[1]
                         and act.get("room") == expected[2])
            elif kind == "patrol":
                rooms = act.get("rooms")
                exact = rooms == [] or (isinstance(rooms, list) and set(
                    x for x in rooms if isinstance(x, str)) == ROOMS
                    and len(rooms) == len(ROOMS))
            elif kind == "come_back":
                exact = True
            elif kind == "find_person":
                exact = (isinstance(act.get("name"), str)
                         and act["name"].lower() == expected[1].lower()
                         and (len(expected) == 2 or bool(act.get("message"))))
            if exact:
                break
    else:
        reject_violation = False
    return dict(parse_ok=True, exact=exact, malformed=malformed,
                critical=critical, reject_violation=reject_violation)


def pct(n, d):
    return f"{100 * n / d:.1f}" if d else ""


def write_tsv(name, columns, records):
    with (HERE / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-invalid", action="store_true",
                        help="include configs with any stop_type=limit in quality and blind outputs")
    args = parser.parse_args()
    sources = sorted((HERE / "runs").glob("bench_results_*.json"))
    if not sources:
        parser.error("no runs/bench_results_*.json in this directory")
    rows = []
    seen = set()
    for source in sources:
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            parser.error(f"{source.name}: expected a JSON list")
        for row in data:
            if not isinstance(row, dict):
                parser.error(f"{source.name}: non-object row")
            key = tuple(row.get(k) for k in ("config", "bucket", "prompt_id", "run_index"))
            if not key[0] or key[1] not in {"A", "B", "C"} or key[3] is None:
                parser.error(f"{source.name}: row missing identity fields")
            if key in seen:
                parser.error(f"duplicate turn across result files: {key}")
            seen.add(key)
            rows.append(row)
    by_config = defaultdict(list)
    for row in rows:
        by_config[row["config"]].append(row)
    invalid = {name for name, group in by_config.items()
               if any(r.get("stop_type") == "limit" or r.get("truncated") is True
                      for r in group)}
    included = set(by_config) if args.include_invalid else set(by_config) - invalid

    speed = []
    for name, group in sorted(by_config.items()):
        good = [r for r in group if "error" not in r]
        rate = [r for r in good if (r.get("predicted_n") or 0) >= 10
                and (r.get("timings") or {}).get("predicted_ms", 0) > 0]
        tokens = sum(r["predicted_n"] for r in rate)
        decode_ms = sum(r["timings"]["predicted_ms"] for r in rate)
        ttft = [r["timings"]["prompt_ms"] / 1000 for r in good
                if isinstance(r.get("timings"), dict)
                and isinstance(r["timings"].get("prompt_ms"), (int, float))]
        latency = [r["latency_ms"] / 1000 for r in good
                   if isinstance(r.get("latency_ms"), (int, float))]
        speed.append(dict(config=name, rows=len(group), errors=len(group)-len(good),
                          limits=sum(r.get("stop_type") == "limit" for r in group),
                          decode_tok_s=f"{tokens * 1000 / decode_ms:.2f}" if decode_ms else "",
                          ttft_median_s=f"{statistics.median(ttft):.1f}" if ttft else "",
                          turn_median_s=f"{statistics.median(latency):.1f}" if latency else "",
                          quality_status="INVALID-truncated" if name in invalid else "eligible"))
    write_tsv("combined_speed.tsv",
              ["config", "rows", "errors", "limits", "decode_tok_s",
               "ttft_median_s", "turn_median_s", "quality_status"], speed)

    quality = []
    for name in sorted(included):
        group = [r for r in by_config[name] if r["bucket"] == "C"]
        if not group:
            continue
        grades = []
        for row in group:
            if row.get("prompt_id") not in EXPECTED:
                parser.error(f"unknown C prompt: {row.get('prompt_id')}")
            raw = row.get("reply_raw")
            if "error" in row:
                strict = lenient = dict(parse_ok=False, exact=False, malformed=False,
                                        critical=False, reject_violation=False)
            elif not isinstance(raw, str):
                parser.error(f"missing reply_raw: {name} {row['prompt_id']}")
            else:
                strict = grade(raw, row["prompt_id"])
                lenient = grade(raw, row["prompt_id"], True)
            grades.append((row, strict, lenient))
        n = len(grades)
        count = lambda field, index: sum(bool(g[index][field]) for g in grades)
        run1 = [(r, s) for r, s, _ in grades if r["run_index"] == 1]
        run1_lenient = [(r, l) for r, _, l in grades if r["run_index"] == 1]
        cross = [s for r, s in run1 if r["prompt_id"] in CROSS]
        quality.append(dict(
            config=name, c_rows=n, c_runs=len({r["run_index"] for r in group}),
            strict_parse_ok_pct=pct(count("parse_ok", 1), n),
            lenient_parse_ok_pct=pct(count("parse_ok", 2), n),
            malformed_pct=pct(count("malformed", 1), n),
            strict_exact_pct=pct(count("exact", 1), n),
            lenient_exact_pct=pct(count("exact", 2), n),
            run1_strict_exact_pct=pct(sum(s["exact"] for _, s in run1), len(run1)),
            run1_lenient_exact_pct=pct(sum(l["exact"] for _, l in run1_lenient),
                                       len(run1_lenient)),
            run1_cross_lingual_pct=pct(sum(s["exact"] for s in cross), len(cross)),
            critical_failures=count("critical", 2),
            reject_violations=count("reject_violation", 2),
            run1_reject_violations=sum(l["reject_violation"] for _, l in run1_lenient),
            quality_status="INVALID-truncated INCLUDED" if name in invalid else
                           "SMOKE-ONLY" if len({r["run_index"] for r in group}) == 1 else "VALID"))
    write_tsv("combined_bucket_c.tsv",
              ["config", "c_rows", "c_runs", "strict_parse_ok_pct", "lenient_parse_ok_pct",
               "malformed_pct", "strict_exact_pct", "lenient_exact_pct",
               "run1_strict_exact_pct", "run1_lenient_exact_pct",
               "run1_cross_lingual_pct", "critical_failures", "reject_violations",
               "run1_reject_violations", "quality_status"], quality)

    blind = []
    for name in sorted(included):
        for row in by_config[name]:
            if row["bucket"] in {"A", "B"} and row["run_index"] == 1:
                if row["prompt_id"] not in PROMPTS_AB:
                    parser.error(f"unknown A/B prompt: {row['prompt_id']}")
                if "error" not in row and isinstance(row.get("reply_raw"), str):
                    reply = row.get("reply_clean")
                    if not isinstance(reply, str):
                        reply = re.sub(r"<think>.*?</think>\s*", "", row["reply_raw"],
                                       flags=re.DOTALL).strip()
                    blind.append((name, row["prompt_id"], reply))
    names = sorted({name for name, _, _ in blind})
    rng = random.Random(0)
    rng.shuffle(names)
    ids = {name: f"M{i+1}" for i, name in enumerate(names)}
    write_tsv("blind_key.tsv", ["model_id", "config"],
              [dict(model_id=ids[name], config=name) for name in names])
    prompts = sorted({prompt for _, prompt, _ in blind},
                     key=lambda x: (x[0], int(x[1:])))
    with (HERE / "blind_ab.txt").open("w", encoding="utf-8") as stream:
        for prompt in prompts:
            stream.write(f"=== PROMPT {prompt} ===\n{PROMPTS_AB[prompt]}\n")
            items = [(ids[name], reply) for name, pid, reply in blind if pid == prompt]
            rng.shuffle(items)
            for model_id, reply in items:
                stream.write(f"\n--- {model_id} ---\n{reply}\n")
            stream.write("\n")
    print(f"{len(sources)} files, {len(rows)} turns; quality excluded: {', '.join(sorted(invalid - included)) or 'none'}")
    print("Wrote combined_speed.tsv, combined_bucket_c.tsv, blind_ab.txt, blind_key.tsv")


if __name__ == "__main__":
    main()
