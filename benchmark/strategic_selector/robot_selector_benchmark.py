#!/usr/bin/env python3
"""Offline, simulation-only Pixel Robot mission selector benchmark.

Run from the existing Debian/proot Python venv:
    taskset -c 4-7 python robot_selector_benchmark.py --model laya
    taskset -c 4-7 python robot_selector_benchmark.py --model von

The two models see different cases. This benchmarks framing *within* each
model; cross-model accuracy percentages are not a controlled comparison.
No motor, camera, network, or ESP32 interfaces are used.
"""

import argparse
import json
import math
import os
import random
import resource
import statistics
import sys
import time
from pathlib import Path


ROOMS = ("bedroom", "living_room", "kitchen", "toilet", "hall")
GENERAL = {
    "search_here": "Search the current room for Chiara using the room-coverage script.",
    "call_chiara": "Call out for Chiara and listen for a reply; do not move.",
    "ask_gemma": "Ask the dialogue planner to revise the mission after search attempts fail.",
    "finish": "Stop the search: Chiara's identity and position have been confirmed.",
    "relocalize": "Stop travel and re-estimate the robot pose using a new LiDAR scan.",
}

FAMILIES = (
    "new_room", "room_finished", "hall_hint", "target_confirmed",
    "possible_person", "all_rooms_first", "all_rooms_called",
    "localization_lost", "route_blocked", "repeat_search", "heard_from_room",
)
FRAMES = (
    "wide_text", "filtered_text", "filtered_json", "two_stage_text", "two_stage_json",
)


def quantile(xs, q):
    if not xs:
        return None
    values = sorted(xs)
    p = (len(values) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] * (hi - p) + values[hi] * (p - lo)


def build_case(model, family, variation):
    """Paired changes within a model; disjoint room/text variants between models."""
    offset = 0 if model == "laya" else 7
    ix = variation + offset
    current = ("bedroom", "living_room", "kitchen")[ix % 3]
    candidates = [r for r in ROOMS if r != current and r != "hall"]
    nearest = candidates[(ix + 1) % len(candidates)]
    other = next(r for r in candidates if r != nearest)
    distances = {r: (2.0 if r == nearest else 6.0 + j)
                 for j, r in enumerate(candidates)}
    room_status = {r: "unsearched" for r in ROOMS}
    room_status["hall"] = "transit"
    pose = "confirmed"
    target = "not_seen"
    last_script = "entered_room"
    last_result = "arrived"
    calls = 0
    hint = "none"
    scripted = "search_here"
    acceptable = None
    reachable = list(candidates)
    note = "Search room interiors methodically; check each room once."

    if family in ("room_finished", "repeat_search", "hall_hint", "heard_from_room"):
        room_status[current] = "searched_no_target"
        last_script = "search_here"
        last_result = "room_coverage_complete"
        scripted = "travel:" + nearest
        if family == "repeat_search":
            last_result = "room_coverage_complete_twice"
            note = "The room search already repeated without finding Chiara."
        if family == "hall_hint":
            current = "hall"
            reachable = [r for r in ROOMS if r != "hall"]
            nearest = reachable[(ix + 1) % len(reachable)]
            distances = {r: (2.0 if r == nearest else 7.0 + j)
                         for j, r in enumerate(reachable)}
            scripted = "travel:" + nearest
            last_script = "entered_hall"
            last_result = "ready_to_pick_a_room"
        if family == "heard_from_room":
            hint = nearest
            note = f"A voice answered from {nearest}; its identity is unconfirmed."

    elif family == "target_confirmed":
        target = "identity_confirmed_Chiara_ahead_at_50_cm"
        scripted = "finish"
        note = "Chiara verified by a separate identity confirmation step."
    elif family == "possible_person":
        target = "person_seen_60_cm_identity_unknown"
        scripted = "call_chiara"
        acceptable = ["call_chiara", "search_here"]
        note = "Camera detected a person; it has not identified Chiara."
    elif family in ("all_rooms_first", "all_rooms_called"):
        room_status = {r: "searched_no_target" for r in ROOMS}
        room_status["hall"] = "transit"
        last_script = "travel_and_search"
        last_result = "all_reachable_rooms_searched"
        scripted = "call_chiara" if family == "all_rooms_first" else "ask_gemma"
        calls = 0 if family == "all_rooms_first" else 2
        if calls:
            last_script = "call_chiara"
            last_result = "no_answer_to_two_calls"
        note = "All currently reachable rooms have been searched."
    elif family == "localization_lost":
        pose = "uncertain_after_wheel_slip"
        last_script = "travel"
        last_result = "pose_estimate_lost"
        scripted = "relocalize"
        note = "Room location is unknown until a fresh scan is aligned."
    elif family == "route_blocked":
        pose = "uncertain_after_blocked_doorway"
        last_script = "travel"
        last_result = "route_blocked_and_pose_uncertain"
        scripted = "relocalize"
        note = "Do not commit to another room until pose is recovered."

    if family == "new_room":
        acceptable = ["search_here", "call_chiara"]
    if acceptable is None:
        if scripted.startswith("travel:"):
            acceptable = [f"travel:{r}" for r in reachable
                          if room_status.get(r) == "unsearched"]
        else:
            acceptable = [scripted]

    name = f"{model}_{family}_{variation}"
    case = {
        "id": name, "family": family,
        "mission": "find Chiara", "location": current,
        "pose_quality": pose, "room_status": room_status,
        "reachable_rooms": reachable, "distance_to_room_m": distances,
        "target_evidence": target, "voice_hint_room": hint,
        "calls_without_answer": calls, "last_script": last_script,
        "last_result": last_result, "sensor_age_s": 0.15 + ix * 0.05,
        "front_lidar_cm": 65 + (ix % 3) * 15,
        "front_ultrasonic_cm": 63 + (ix % 3) * 15,
        "note": note, "preferred": scripted, "acceptable": acceptable,
    }
    if family in ("room_finished", "hall_hint", "repeat_search", "heard_from_room"):
        assert scripted.split(":", 1)[1] in reachable
    if family == "all_rooms_called":
        assert calls == 2
    return case


def cases_for(model):
    return [build_case(model, family, variation)
            for family in FAMILIES for variation in range(3)]


def to_state(case, structured):
    keys = (
        "mission", "location", "pose_quality", "room_status", "reachable_rooms",
        "distance_to_room_m", "target_evidence", "voice_hint_room",
        "calls_without_answer", "last_script", "last_result", "sensor_age_s",
        "front_lidar_cm", "front_ultrasonic_cm", "note",
    )
    info = {k: case[k] for k in keys}
    if structured:
        return info
    status = "; ".join(f"{k}: {v}" for k, v in info["room_status"].items())
    routes = ", ".join(f"{r} ({info['distance_to_room_m'].get(r, '?')} m)"
                       for r in info["reachable_rooms"])
    return (f"Mission: find Chiara. Current room: {info['location']}. "
            f"Localization: {info['pose_quality']}. Room history: {status}. "
            f"Reachable rooms: {routes}. Target evidence: {info['target_evidence']}. "
            f"Voice hint: {info['voice_hint_room']}. Calls without answer: {info['calls_without_answer']}. "
            f"Previous script: {info['last_script']}; outcome: {info['last_result']}. "
            f"LiDAR front: {info['front_lidar_cm']} cm; ultrasonic front: "
            f"{info['front_ultrasonic_cm']} cm; sensor age {info['sensor_age_s']:.2f} s. "
            f"{info['note']}")


def options_for(case, frame, stage, rng):
    wide = frame == "wide_text"
    eligible = {}
    confirmed = case["target_evidence"].startswith("identity_confirmed")
    pose_ok = case["pose_quality"] == "confirmed"
    unsearched_here = case["room_status"][case["location"]] == "unsearched"
    can_travel = pose_ok and not confirmed and any(
        case["room_status"].get(r) == "unsearched" for r in case["reachable_rooms"]
    )
    if stage == "destination":
        rooms = [r for r in case["reachable_rooms"]
                 if case["room_status"].get(r) == "unsearched"]
        if wide:
            rooms = [r for r in ROOMS if r != "hall"]
        return {f"travel:{r}": f"Travel to {r}; distance "
                f"{case['distance_to_room_m'].get(r, 'unknown')} m; "
                f"room status {case['room_status'].get(r, 'unknown')}." for r in rooms}

    if wide or (pose_ok and unsearched_here and not confirmed):
        eligible["search_here"] = GENERAL["search_here"]
    if wide or not confirmed:
        eligible["call_chiara"] = GENERAL["call_chiara"]
    eligible["ask_gemma"] = GENERAL["ask_gemma"]
    if wide or confirmed:
        eligible["finish"] = GENERAL["finish"]
    if wide or not pose_ok:
        eligible["relocalize"] = GENERAL["relocalize"]
    if stage == "mode":
        if wide or can_travel:
            eligible["travel_next"] = "Travel to one unsearched reachable room."
    else:
        eligible.update(options_for(case, frame, "destination", rng)) if (wide or can_travel) else None
    # Keep each frame's option order stable during tuning. Order is perturbed only
    # in the separate permutation check on held-out cases.
    return eligible


def render_options(options, mode, rng):
    items = list(options.items())
    if mode == "reverse":
        items.reverse()
    elif mode == "shuffle":
        rng.shuffle(items)
    return dict(items)


class Adapter:
    def __init__(self, model):
        self.name = model
        self.calls = 0
        self.first_call_ms = None
        self.load_ms = 0
        if model == "mock":
            return
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        print(f"TORCH={torch.__version__} THREADS={torch.get_num_threads()} ", flush=True)
        began = time.perf_counter()
        if model == "laya":
            import laya
            self.agent = laya.load("convaiinnovations/laya")
        else:
            import von
            self.agent = von
        self.load_ms = (time.perf_counter() - began) * 1000

    def decide(self, state, options, instruction):
        if len(options) < 2:
            raise ValueError("Decision requires at least two available options")
        start = time.perf_counter()
        if self.name == "laya":
            import torch
            with torch.inference_mode():
                raw = self.agent.predict(state, {"next": {
                    "type": "choice", "instructions": instruction,
                    "criteria": options,
                }})
            answer = raw["answers"]["next"]
            choice = answer["choice"]
            probs = answer.get("probabilities", {})
            confidence = answer.get("confidence")
            tokens = raw.get("usage", {}).get("input_tokens")
        elif self.name == "von":
            import torch
            with torch.inference_mode():
                answer = self.agent.decide(
                    state=state, choices=options, instructions=instruction)
            choice = answer.choice
            probs = answer.probabilities
            confidence = answer.confidence
            tokens = None
        else:
            choice = next(iter(options))
            probs = {key: int(key == choice) for key in options}
            confidence = None
            tokens = None
        elapsed = (time.perf_counter() - start) * 1000
        self.calls += 1
        if self.first_call_ms is None:
            self.first_call_ms = elapsed
        return {"choice": choice, "probabilities": dict(probs),
                "confidence": confidence, "tokens": tokens, "elapsed_ms": elapsed}


def run_case(adapter, case, frame, order="normal"):
    rng = random.Random(case["id"] + frame + order)
    structured = frame.endswith("json")
    state = to_state(case, structured)
    multi = frame.startswith("two_stage")
    stage = "mode" if multi else "flat"
    opts = render_options(options_for(case, frame, stage, rng), order, rng)
    instruction = (
        "Given the mission, evidence, room history and previous script result, "
        "choose the single most useful next high-level mission action. "
        "A travel script will handle movement and obstacles itself."
    )
    first = adapter.decide(state, opts, instruction)
    calls = [first]
    chosen = first["choice"]
    if chosen == "travel_next" and multi:
        dest = render_options(options_for(case, frame, "destination", rng), order, rng)
        if len(dest) >= 2:
            second = adapter.decide(
                state, dest,
                "Which unsearched reachable room should the robot travel to next? "
                "Use the room history, any voice hint, and travel distances.")
            calls.append(second)
            chosen = second["choice"]
        elif len(dest) == 1:
            chosen = next(iter(dest))
        else:
            chosen = "INVALID_NO_DESTINATION"
    elif chosen == "travel_next":
        chosen = "INVALID_NO_DESTINATION"
    elif chosen not in opts:
        chosen = "INVALID_OUTSIDE_OPTIONS"
    score = {
        "model": adapter.name, "frame": frame, "case_id": case["id"],
        "family": case["family"], "order": order,
        "preferred": case["preferred"], "acceptable": case["acceptable"],
        "choice": chosen, "first_choice": first["choice"],
        "preferred_match": chosen == case["preferred"],
        "acceptable_match": chosen in case["acceptable"],
        "invalid_or_ineligible": chosen not in options_for(case, "filtered_text", "flat", rng),
        "total_ms": round(sum(x["elapsed_ms"] for x in calls), 3),
        "model_calls": len(calls), "input_tokens": sum(x["tokens"] or 0 for x in calls),
        "confidence": calls[-1]["confidence"],
        "probabilities": calls[-1]["probabilities"],
        "state": state, "offered_first": list(opts),
        "first_instruction": instruction, "offered_first_descriptions": opts,
        "second_instruction": (
            "Which unsearched reachable room should the robot travel to next? "
            "Use the room history, any voice hint, and travel distances."
        ) if len(calls) > 1 else None,
        "offered_second_descriptions": dest if len(calls) > 1 else None,
    }
    return score


def aggregate(rows):
    if not rows:
        return {}
    ms = [r["total_ms"] for r in rows]
    return {
        "n": len(rows),
        "preferred": sum(r["preferred_match"] for r in rows),
        "acceptable": sum(r["acceptable_match"] for r in rows),
        "invalid_or_ineligible": sum(r["invalid_or_ineligible"] for r in rows),
        "inference_calls": sum(r["model_calls"] for r in rows),
        "p50_ms": round(statistics.median(ms), 1),
        "p95_ms": round(quantile(ms, 0.95), 1),
    }


def audit_fixtures(model):
    cases = cases_for(model)
    assert len(cases) == 33
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["preferred"] in options_for(c, "filtered_text", "flat", random.Random(0)), c["id"]
        for frame in FRAMES:
            opts = options_for(c, frame, "mode" if frame.startswith("two_stage") else "flat", random.Random(0))
            assert len(opts) >= 2, (c["id"], frame)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("laya", "von", "mock"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Four cases per frame for integration smoke")
    args = parser.parse_args()
    model_cases = "laya" if args.model == "mock" else args.model
    cases = audit_fixtures(model_cases)
    dev = [c for c in cases if int(c["id"].rsplit("_", 1)[1]) < 2]
    held = [c for c in cases if int(c["id"].rsplit("_", 1)[1]) == 2]
    if args.quick:
        dev = dev[:4]
        held = held[:2]
    out = args.output or Path(f"robot-selector-{args.model}-results.json")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    print("BENCHMARK_VERSION=2 MODEL=" + args.model, flush=True)
    print("CASE_COUNTS", len(dev), len(held), "DISJOINT_MODEL_FIXTURES=true", flush=True)
    print("NO_MOTOR_OR_NETWORK_IO=true", flush=True)
    print("AFFINITY=" + str(sorted(os.sched_getaffinity(0))), flush=True)
    adapter = Adapter(args.model)
    print(f"IMPORT_AND_LOAD_MS={adapter.load_ms:.1f}", flush=True)
    warm_state = to_state(dev[0], False)
    warm_options = options_for(dev[0], "filtered_text", "flat", random.Random(0))
    for n in range(2):
        warm = adapter.decide(
            warm_state, warm_options,
            "Which high-level script should the robot run next to find Chiara?")
        print(f"WARMUP_{n+1}_MS={warm['elapsed_ms']:.1f}", flush=True)
    rows = []
    summary = {}
    frame_rows = {frame: [] for frame in FRAMES}
    for i, case in enumerate(dev):
        # Rotate frames so heat/drift does not systematically favor one frame.
        for frame in FRAMES[i % len(FRAMES):] + FRAMES[:i % len(FRAMES)]:
            row = run_case(adapter, case, frame)
            row["split"] = "development"
            rows.append(row)
            frame_rows[frame].append(row)
            print(f"DEV {frame} {case['id']} expected={row['preferred']} "
                  f"chosen={row['choice']} acceptable={row['acceptable_match']} "
                  f"ms={row['total_ms']:.0f}", flush=True)
    for frame in FRAMES:
        summary[frame] = aggregate(frame_rows[frame])
        print("DEV_SUMMARY", frame, json.dumps(summary[frame]), flush=True)

    # Select by acceptable count, then exact preferred, then fewer invalid
    # choices, then lower latency. Held-out labels never influence selection.
    best = max(FRAMES, key=lambda f: (
        summary[f]["acceptable"], summary[f]["preferred"],
        -summary[f]["invalid_or_ineligible"], -summary[f]["p50_ms"]))
    print("SELECTED_ON_DEVELOPMENT=" + best, flush=True)
    held_rows = []
    order_rows = []
    for case in held:
        row = run_case(adapter, case, best)
        row["split"] = "heldout"
        rows.append(row)
        held_rows.append(row)
        print(f"HELDOUT {case['id']} expected={row['preferred']} "
              f"chosen={row['choice']} acceptable={row['acceptable_match']} "
              f"ms={row['total_ms']:.0f}", flush=True)
    for case in held:
        row = run_case(adapter, case, best, "reverse")
        row["split"] = "heldout_order_reversed"
        rows.append(row)
        order_rows.append(row)
    flips = sum(a["choice"] != b["choice"] for a, b in zip(held_rows, order_rows))
    report = {
        "model": args.model, "architecture": "simulation_only",
        "case_data_disjoint_by_model": True,
        "development": summary, "selected": best,
        "heldout": aggregate(held_rows),
        "heldout_order_reversal_flips": flips,
        "heldout_order_reversal_count": len(held),
        "first_call_ms": adapter.first_call_ms,
        "import_and_load_ms": adapter.load_ms,
        "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "rows": rows,
        "method": "Synthetic preference labels; not measured robot success or calibrated safety.",
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("FINAL_HELDOUT", json.dumps(report["heldout"]), flush=True)
    print(f"ORDER_REVERSAL_FLIPS={flips}/{len(held)}", flush=True)
    print(f"INFERENCE_CALLS={adapter.calls} MAX_RSS_MIB={report['max_rss_mib']}", flush=True)
    print(f"RESULT_FILE={out.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"BENCHMARK_ERROR={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
