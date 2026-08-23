"""Wait for a training run to finish, then launch the weighted-dice run.

Polls the reference run's checkpoint directory for the final epoch's .pth file,
waits for it to stop growing, copies the dataset into an isolated directory so the
second run never touches the first run's files, and launches the weighted run.

Designed to be started detached and left alone. It writes everything it does to a
log file, so if it fails overnight the reason is recoverable in the morning.
"""

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def log(msg, log_file):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-dir", required=True, help="Reference run's checkpoint directory")
    ap.add_argument("--final-epoch", type=int, required=True,
                    help="Zero-based index of the last epoch, e.g. 19 for a 20-epoch run")
    ap.add_argument("--src-data", required=True, help="Directory holding dataset.hdf5 and the CSVs")
    ap.add_argument("--dest-data", required=True, help="Directory to copy the dataset into")
    ap.add_argument("--out-dir", required=True, help="Output directory for the weighted run")
    ap.add_argument("--weights", default="0.1,0.7,0.2")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--quiet-seconds", type=int, default=180,
                    help="Checkpoint must stop changing for this long before we act")
    ap.add_argument("--timeout-hours", type=float, default=14.0)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    log_file = Path(args.log)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    watch_dir = Path(args.watch_dir)

    log(f"watching {watch_dir} for Loss-{args.final_epoch}_*.pth", log_file)
    log(f"poll {args.poll_seconds}s | quiet {args.quiet_seconds}s | timeout {args.timeout_hours}h", log_file)

    deadline = time.time() + args.timeout_hours * 3600
    stable_since = None
    last_size = -1
    last_report = 0.0

    while True:
        if time.time() > deadline:
            log("TIMED OUT waiting for the reference run to finish. Not launching.", log_file)
            return 1

        matches = sorted(watch_dir.glob(f"Loss-{args.final_epoch}_*.pth"))
        if matches:
            size = matches[0].stat().st_size
            if size == last_size and size > 0:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= args.quiet_seconds:
                    log(f"final checkpoint settled: {matches[0].name} ({size / 1e6:.1f} MB)", log_file)
                    break
            else:
                stable_since = None
                last_size = size
        elif time.time() - last_report > 900:
            done = sorted(watch_dir.glob("Loss-*.pth"))
            log(f"still waiting; {len(done)} of {args.final_epoch + 1} epochs saved", log_file)
            last_report = time.time()

        time.sleep(args.poll_seconds)

    # Copy the dataset so the second run reads its own files. Same CSVs means the
    # same train/val split, which is what makes this a controlled comparison.
    src, dest = Path(args.src_data), Path(args.dest_data)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.hdf5", "patches.csv", "val.csv", "ratios.csv"):
        s, d = src / name, dest / name
        if d.exists() and d.stat().st_size == s.stat().st_size:
            log(f"{name} already copied, skipping", log_file)
            continue
        t = time.time()
        shutil.copy2(s, d)
        log(f"copied {name} ({d.stat().st_size / 1e6:.1f} MB) in {time.time() - t:.0f}s", log_file)

    config = watch_dir / "Config.yaml"
    if not config.exists():
        log(f"ERROR: no Config.yaml at {config}; cannot match settings. Not launching.", log_file)
        return 1

    cmd = [
        "uv", "run", "--no-sync", "python", str(REPO / "Scripts" / "run_weighted_training.py"),
        "--config", str(config),
        "--data-dir", str(dest),
        "--out-dir", str(args.out_dir),
        "--weights", args.weights,
    ]
    log(f"launching: {' '.join(cmd)}", log_file)

    train_log = log_file.with_name("weighted_training.log")
    with open(train_log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT)
    log(f"started pid {proc.pid}; training output -> {train_log}", log_file)

    rc = proc.wait()
    log(f"weighted run exited with code {rc}", log_file)
    return rc


if __name__ == "__main__":
    sys.exit(main())
