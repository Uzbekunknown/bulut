"""blind_report.py - the blind test's answers from the cloud simulator layer's log.

In the game: F8 turns blind mode on; each F6 then starts a trial with a mode picked at random and
hidden (the corner mark too); play a little, then answer: F9 "no lag", F10 "lag". The last answer
in a trial counts. The log keeps the truth.

  python -I doom/blind_report.py [doom/layer/build/cloud-sim-vk.log ...]

Prints, per true mode and delay: the trials answered and the share answered "no lag".
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent


def main():
    logs = [Path(p) for p in sys.argv[1:]] or [HERE / "layer" / "build" / "cloud-sim-vk.log"]
    trials = {}
    for log in logs:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"blind trial (\d+): (\w+) \(hidden\), delay (\d+) ms", line)
            if m:
                trials[(log, int(m.group(1)))] = {"mode": m.group(2), "delay": int(m.group(3)), "guess": None}
                continue
            m = re.search(r"blind trial (\d+) guess: ([\w-]+) \(truth (\w+), delay (\d+) ms\)", line)
            if m and (log, int(m.group(1))) in trials:
                t = trials[(log, int(m.group(1)))]
                t["guess"] = m.group(2)
                t["mode"], t["delay"] = m.group(3), int(m.group(4))   # the truth when answered
    table = defaultdict(lambda: [0, 0])
    for t in trials.values():
        if t["guess"] is None:
            continue
        row = table[(t["delay"], t["mode"])]
        row[0] += 1
        row[1] += t["guess"] == "no-lag"
    if not table:
        print("no answered blind trials in the log: F8, then F6 for each trial, F9 (no lag) or F10 (lag) to answer")
        return
    print("delay | true mode | trials | answered 'no lag'")
    for (delay, mode), (n, nolag) in sorted(table.items()):
        print(f"{delay:5d} | {mode:9s} | {n:6d} | {nolag:3d} ({100.0 * nolag / n:.0f} %)")
    print(f"{sum(r[0] for r in table.values())} trials answered of {len(trials)} started")


if __name__ == "__main__":
    main()
