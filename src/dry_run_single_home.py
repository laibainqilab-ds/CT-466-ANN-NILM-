"""
Supervised single-home extraction dry run (CT-466 Phase 2, Stage 2).

Streams the archive via tarfile (read-only, never extracts/modifies it),
filters rows for ONE home (dataid=TARGET_HOME), writes the 7 relevant
columns to a small CSV in data/interim/. Standard library only.
"""
import tarfile
import csv
import io
import os
import time

ARCHIVE = r"E:\CT-466\ANN_NILM\1minute_data_austin.tar.gz"
OUTPUT = r"E:\CT-466\ANN_NILM\data\interim\home_7951_dry_run.csv"
TARGET_HOME = "7951"
KEEP_COLS = ["dataid", "localminute", "air1", "dishwasher1", "refrigerator1", "grid", "solar"]
PROGRESS_EVERY = 2_000_000

print(f"PID={os.getpid()}", flush=True)
print(f"START archive={ARCHIVE}", flush=True)

t0 = time.time()
total_rows = 0
matched_rows = 0

with tarfile.open(ARCHIVE, mode="r:gz") as tf, open(OUTPUT, "w", newline="", encoding="utf-8") as out_f:
    member = tf.getmember("1minute_data_austin/1minute_data_austin.csv")
    f = tf.extractfile(member)
    text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
    reader = csv.reader(text)
    header = next(reader)
    idx = {c: i for i, c in enumerate(header)}
    col_i = [idx[c] for c in KEEP_COLS]
    di = idx["dataid"]

    writer = csv.writer(out_f)
    writer.writerow(KEEP_COLS)

    for row in reader:
        total_rows += 1
        if row[di] == TARGET_HOME:
            writer.writerow([row[i] for i in col_i])
            matched_rows += 1

        if total_rows % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            print(f"PROGRESS total_rows={total_rows:,} matched_rows={matched_rows:,} elapsed={elapsed:.1f}s", flush=True)

elapsed = time.time() - t0
out_size = os.path.getsize(OUTPUT)
print(f"DONE total_rows={total_rows:,} matched_rows={matched_rows:,} elapsed={elapsed:.1f}s output_bytes={out_size:,}", flush=True)
