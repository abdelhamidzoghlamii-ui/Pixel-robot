#!/usr/bin/env python3
"""Blind grading sheet for the #110 conversation (A) and Q&A (B) buckets.

Usage: make_blind_sheet.py RESULTS.json SHEET.md KEY.tsv
One section per prompt, run 1 of every model, answers shuffled independently per prompt (OS-seeded, so a letter
never follows one model across prompts) and labelled A, B, C, ... The key (prompt, letter, model) goes to a
separate file; the sheet names no model and carries no model-specific metadata except a truncation mark.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/termux-home/robot/benchmark/llm_objective_setting")
import bench  # noqa: E402  prompt texts

LETTERS = "ABCDEFGHIJ"
NOTES = {"B2": "Honesty trap: nothing in the prompt tells the robot where the keys are.",
         "B6": "Honesty trap: nothing in the prompt gives the robot a battery reading.",
         "A5": "Multi-turn: only the model's reply to the last user turn is stored and shown; the earlier replies were "
               "generated but not saved by bench.py."}


def main(results, sheet, key):
    rows = [r for r in json.loads(Path(results).read_text()) if r.get("bucket") in ("A", "B") and r.get("run_index") == 1]
    models = sorted({r["config"] for r in rows})
    if len(models) > len(LETTERS):
        raise SystemExit("too many models for the letter set")
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], {})[r["config"]] = r
    rng = random.SystemRandom()
    out = ["# Blind grading sheet: robot conversation (A) and Q&A (B)", "",
           f"{len(models)} answers per prompt, labelled {LETTERS[0]}-{LETTERS[len(models) - 1]}; the order is shuffled "
           "separately for every prompt, so a letter does not stand for the same model twice. The system prompt asked "
           "for brief, natural spoken replies (at most two short sentences unless asked for more), in the user's "
           "language, saying plainly when the robot does not know or cannot sense something.",
           "", "Do not open the key file until every prompt is graded.", ""]
    key_lines = ["prompt\tletter\tmodel"]
    for p in bench.PROMPTS_A + bench.PROMPTS_B:
        pid = p["id"]
        text = " / ".join(t["user"] for t in p["text"]) if p.get("multi") else p["text"]
        out += ["---", "", f"## {pid} ({p['lang']})", "", f"> {text}", ""]
        if pid in NOTES:
            out += [f"*{NOTES[pid]}*", ""]
        order = list(models)
        rng.shuffle(order)
        for letter, model in zip(LETTERS, order):
            r = by_prompt.get(pid, {}).get(model)
            if r is None:
                answer = "(no answer: this model has no run-1 row for this prompt)"
            elif "error" in r:
                answer = "(no answer: request failed)"
            else:
                answer = r["reply_clean"].strip() or "(empty reply)"
                if r.get("truncated"):
                    answer += "\n\n*(cut off at the 512-token limit)*"
            out += [f"**{letter}.**", "", *("    " + l for l in answer.splitlines()), "",
                    "Score (1-5): ____   Notes: ______________________________", ""]
            key_lines.append(f"{pid}\t{letter}\t{model}")
    Path(sheet).write_text("\n".join(out) + "\n")
    Path(key).write_text("\n".join(key_lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
