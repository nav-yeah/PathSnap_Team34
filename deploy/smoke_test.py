"""Demo-day rehearsal: call a running API for every trip and print a summary.

    python deploy/smoke_test.py --api http://<laptop1-ip>:8000 [--out results.json]

Exit code 0 only if /health is OK and every trip returns HTTP 200.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ui.api_client import ApiError, PathSnapClient  # noqa: E402


def fmt(value):
    return "  n/a " if value is None else f"{value * 100:6.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    client = PathSnapClient(args.api)
    try:
        health = client.health()
        trips = client.trips()
    except ApiError as error:
        print(f"FAIL: {error}")
        return 1
    print(f"API {args.api}  model={health['model_version']}  decoder={health['decoder']}  "
          f"trips={len(trips)}")
    if not health["model_loaded"]:
        print(f"FAIL: {health['model_error']}")
        return 1
    print(f"{'trip':>10} {'points':>6} {'top1%':>6} {'clf%':>6} {'cand%':>6} {'cont%':>6} "
          f"{'jumps':>5} {'server_ms':>9} {'rtt_ms':>7}")
    failures, rows = 0, []
    for trip in trips:
        start = time.perf_counter()
        try:
            result = client.predict(trip["trip_id"])
        except ApiError as error:
            failures += 1
            print(f"{trip['trip_id']:>10}  FAIL {error}")
            continue
        rtt = (time.perf_counter() - start) * 1000
        m, q = result["metrics"] or {}, result["route_quality"]
        print(f"{result['trip_id']:>10} {result['observations']:>6} {fmt(m.get('top1_accuracy'))} "
              f"{fmt(m.get('classifier_top1_accuracy'))} {fmt(m.get('candidate_recall'))} "
              f"{fmt(q['continuity'])} {q['disconnected_jumps']:>5} "
              f"{result['timings_ms']['total_s']:>9.1f} {rtt:>7.1f}")
        rows.append({k: result[k] for k in ("trip_id", "observations", "model_version", "decoder",
                                           "metrics", "route_quality", "timings_ms")} | {"rtt_ms": rtt})
    if args.out:
        args.out.write_text(json.dumps(dict(api=args.api, health=health, trips=rows), indent=2))
        print(f"saved {args.out}")
    print("OK" if not failures else f"{failures} trip(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
