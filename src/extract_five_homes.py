"""
Stage 3: full five-home streaming extraction (CT-466 Phase 2).

Streams the archive via tarfile (read-only, never extracts/modifies it),
filters rows for the five approved homes, writes raw values to a Parquet
file incrementally in bounded batches (pyarrow.parquet.ParquetWriter).
Row order is preserved exactly as encountered -- no sorting is performed.
Timestamp parsing here is used ONLY for integrity verification (row
counts, ranges, duplicate detection); it does not alter output content
or order.
"""
import tarfile
import csv
import io
import os
import time
from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

ARCHIVE = r"E:\CT-466\ANN_NILM\1minute_data_austin.tar.gz"
OUTPUT = r"E:\CT-466\ANN_NILM\data\interim\five_homes.parquet"
APPROVED_HOMES = {"7901", "7951", "8386", "8565", "3538"}
KEEP_COLS = ["dataid", "localminute", "air1", "dishwasher1", "refrigerator1", "grid", "solar"]
NUMERIC_COLS = ["air1", "dishwasher1", "refrigerator1", "grid", "solar"]
BATCH_SIZE = 100_000
PROGRESS_EVERY = 2_000_000

REF_EPOCH = datetime(2015, 1, 1)
ARRAY_MINUTES = 7 * 365 * 24 * 60  # ~7 years, safely covers the 2018 data range

print(f"PID={os.getpid()}", flush=True)
print(f"START archive={ARCHIVE}", flush=True)
t0 = time.time()

schema = pa.schema([
    ("dataid", pa.string()),
    ("localminute", pa.string()),
    ("air1", pa.float64()),
    ("dishwasher1", pa.float64()),
    ("refrigerator1", pa.float64()),
    ("grid", pa.float64()),
    ("solar", pa.float64()),
])

row_count = {h: 0 for h in APPROVED_HOMES}
min_ts = {}
max_ts = {}
minute_seen = {h: bytearray(ARRAY_MINUTES) for h in APPROVED_HOMES}
parse_errors = {h: 0 for h in APPROVED_HOMES}
out_of_range = {h: 0 for h in APPROVED_HOMES}

buffers = {c: [] for c in KEEP_COLS}


def flush_batch(writer):
    if not buffers["dataid"]:
        return
    arrays = [
        pa.array(buffers["dataid"], type=pa.string()),
        pa.array(buffers["localminute"], type=pa.string()),
    ]
    for c in NUMERIC_COLS:
        arrays.append(pa.array(buffers[c], type=pa.float64()))
    table = pa.Table.from_arrays(arrays, schema=schema)
    writer.write_table(table)
    for c in KEEP_COLS:
        buffers[c].clear()


total_rows = 0
matched_rows = 0

with tarfile.open(ARCHIVE, mode="r:gz") as tf, pq.ParquetWriter(OUTPUT, schema) as writer:
    member = tf.getmember("1minute_data_austin/1minute_data_austin.csv")
    f = tf.extractfile(member)
    text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
    reader = csv.reader(text)
    header = next(reader)
    idx = {c: i for i, c in enumerate(header)}
    di, ti = idx["dataid"], idx["localminute"]
    col_i = {c: idx[c] for c in KEEP_COLS}

    for row in reader:
        total_rows += 1
        did = row[di]
        if did in APPROVED_HOMES:
            matched_rows += 1
            row_count[did] += 1

            buffers["dataid"].append(did)
            ts_raw = row[ti]
            buffers["localminute"].append(ts_raw)
            for c in NUMERIC_COLS:
                raw = row[col_i[c]]
                buffers[c].append(float(raw) if raw != "" else None)

            try:
                naive = datetime.strptime(ts_raw[:19], "%Y-%m-%d %H:%M:%S")
                off_sign = ts_raw[19]
                off_val = int(ts_raw[20:])
                offset_hours = off_val if off_sign == "+" else -off_val
                utc = naive - timedelta(hours=offset_hours)
            except Exception:
                parse_errors[did] += 1
                utc = None

            if utc is not None:
                if did not in min_ts or utc < min_ts[did]:
                    min_ts[did] = utc
                if did not in max_ts or utc > max_ts[did]:
                    max_ts[did] = utc
                offset_min = int((utc - REF_EPOCH).total_seconds() // 60)
                if 0 <= offset_min < ARRAY_MINUTES:
                    arr = minute_seen[did]
                    if arr[offset_min] < 255:
                        arr[offset_min] += 1
                else:
                    out_of_range[did] += 1

            if len(buffers["dataid"]) >= BATCH_SIZE:
                flush_batch(writer)

        if total_rows % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            print(f"PROGRESS total_rows={total_rows:,} matched_rows={matched_rows:,} elapsed={elapsed:.1f}s", flush=True)

    flush_batch(writer)

elapsed = time.time() - t0
out_size = os.path.getsize(OUTPUT)
print(f"DONE total_rows={total_rows:,} matched_rows={matched_rows:,} elapsed={elapsed:.1f}s output_bytes={out_size:,}", flush=True)

print("\n--- PER-HOME INTEGRITY ---", flush=True)
for h in sorted(APPROVED_HOMES, key=int):
    arr = minute_seen[h]
    if h in min_ts:
        start = int((min_ts[h] - REF_EPOCH).total_seconds() // 60)
        end = int((max_ts[h] - REF_EPOCH).total_seconds() // 60)
        span = end - start + 1
        present = sum(1 for i in range(start, end + 1) if arr[i] >= 1)
        dup_minutes = sum(1 for i in range(start, end + 1) if arr[i] >= 2)
        dup_extra = sum(arr[i] - 1 for i in range(start, end + 1) if arr[i] >= 2)
        print(f"HOME {h}: rows={row_count[h]:,} ts_min={min_ts[h]} ts_max={max_ts[h]} "
              f"span_minutes={span:,} present={present:,} missing={span-present:,} "
              f"dup_minutes={dup_minutes:,} dup_extra_rows={dup_extra:,} "
              f"parse_errors={parse_errors[h]} out_of_range={out_of_range[h]}", flush=True)
    else:
        print(f"HOME {h}: rows={row_count[h]:,} NO VALID TIMESTAMPS parse_errors={parse_errors[h]}", flush=True)
