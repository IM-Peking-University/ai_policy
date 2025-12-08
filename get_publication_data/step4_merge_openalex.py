import csv
from collections import OrderedDict

# File paths
OLD_FILE = 'Q1_papers_v0.csv'
NEW_FILE = '2025_06.csv'
MERGED_FILE = 'Q1_papers_v1.csv'

print("Starting merge and deduplication (work_id)...")

# 1. Read old file, record existing work_ids
seen_ids = set()
old_rows = []

with open(OLD_FILE, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames  # Preserve original field order
    for row in reader:
        wid = row['work_id'].strip()
        if wid and wid not in seen_ids:
            seen_ids.add(wid)
            old_rows.append(row)

print(f"  Old file: {len(old_rows)} papers")

# 2. Read new file, only add unseen ones
new_rows = []
with open(NEW_FILE, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    if reader.fieldnames != fieldnames:
        print("Warning: New file fields differ from old file, will use old file field order")
    for row in reader:
        wid = row['work_id'].strip()
        if wid and wid not in seen_ids:
            seen_ids.add(wid)
            new_rows.append(row)

print(f"  New: {len(new_rows)} papers")

# 3. Merge and write
all_rows = old_rows + new_rows
print(f"  After merge: {len(all_rows)} papers")

with open(MERGED_FILE, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Merge complete: {len(all_rows)} papers → {MERGED_FILE}")