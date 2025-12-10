import csv

INPUT_FILE = 'Q1_papers_v1.csv'
OUTPUT_FILE = 'missing_abstract.csv'

print("Extracting papers with empty abstracts...")

missing_count = 0
written_count = 0

with open(INPUT_FILE, 'r', encoding='utf-8', newline='') as fin, \
     open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as fout:

    reader = csv.DictReader(fin)
    fieldnames = ['work_id', 'doi', 'title']
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        abstract = row.get('abstract', '').strip()
        title = row.get('title', '').strip()
        doi = row.get('doi', '').strip()
        work_id = row.get('work_id', '').strip()

        # Check if abstract is empty (including null, None, empty string)
        is_empty = (
            not abstract or
            abstract.lower() in {'null', 'none', 'na', ''} or
            abstract == ''
        )

        if is_empty:
            missing_count += 1
            # Write if at least one of work_id, doi, or title exists
            if work_id or doi or title:
                writer.writerow({
                    'work_id': work_id,
                    'doi': doi,
                    'title': title
                })
                written_count += 1

print(f"\nComplete!")
print(f"  Papers with empty abstract: {missing_count:,}")
print(f"  Exported records: {written_count:,}")
print(f"  Output file: {OUTPUT_FILE}")