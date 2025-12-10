import csv, json, re
from collections import defaultdict

# Load Q1 journal mapping: source_id → q1_name
print("Step 1: Loading Q1 journal mapping...")
q1_map = {}
with open('Q1_matched_final.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sid = row['df_id'].split('/')[-1]
        q1_map[sid] = row['q1_name']
print(f"  Loaded {len(q1_map)} Q1 journals")

# Extract work_id + source_id + is_oa from works_locations.tsv
print("Step 2: Filtering works_locations.tsv...")
work_source = {}
work_oa = {}
with open('../openalex_data/works_locations.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    wid_idx = header.index('work_id')
    sid_idx = header.index('source_id')
    oa_idx = header.index('is_oa')
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt % 100_000 == 0:
            print(f"    Processed {cnt:,} rows (works_locations.tsv)")

        sid = row[sid_idx].split('/')[-1]
        if sid not in q1_map: 
            continue
        wid = row[wid_idx].split('/')[-1]
        work_source[wid] = sid
        work_oa[wid] = row[oa_idx] == 'True'

print(f"  Found {len(work_source)} candidate papers")

# Filter works.tsv for 2021-2025.6 + en + hasabstract
print("Step 3: Filtering works.tsv...")
valid_works = {}

def invert_abstract(inverted_index):
    """Convert OpenAlex abstract_inverted_index to normal abstract text"""
    if not inverted_index or str(inverted_index).strip() in {'', 'null', 'NULL'}:
        return ""

    s = str(inverted_index).strip()

    # Step 1: Remove outer quotes
    if len(s) >= 2 and s[0] in {'"', "'"} and s[-1] == s[0]:
        s = s[1:-1]

    # Step 2: Replace single-quote keys 'word': with "word":
    result = []
    i = 0
    while i < len(s):
        if s[i:i+2] == "\\\\'":
            result.append(s[i:i+2])
            i += 2
        elif s[i] == "'":
            # Find matching closing quote
            j = i + 1
            while j < len(s):
                if s[j] == "\\\\":
                    j += 2
                    continue
                if s[j] == "'":
                    break
                j += 1
            # Found 'word' pattern
            if j < len(s) and s[j+1:j+3] in {': ', ':['}:
                word = s[i+1:j]
                result.append(f'"{word}"')
                i = j + 1
            else:
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1

    json_str = ''.join(result)

    # Step 3: Parse as dict
    try:
        index_data = json.loads(json_str)
    except json.JSONDecodeError:
        return ""

    if not isinstance(index_data, dict):
        return ""

    # Step 4: Rebuild abstract efficiently
    max_pos = max((p for ps in index_data.values() for p in ps), default=-1)
    if max_pos < 0:
        return ""

    words = [''] * (max_pos + 1)
    for word, positions in index_data.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                words[pos] = word

    return ' '.join(filter(None, words))

with open('../openalex_data/works.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    wid_idx = header.index('id')
    doi_idx = header.index('doi')
    title_idx = header.index('title')
    date_idx = header.index('publication_date')
    abs_idx = header.index('abstract_inverted_index')
    lang_idx = header.index('language')
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt % 100_000 == 0:
            print(f"    Processed {cnt:,} rows (works.tsv)")

        wid = row[wid_idx].split('/')[-1]
        if wid not in work_source: 
            continue
        date = row[date_idx]
        if not ('2021-01-01' <= date < '2025-07-01'): 
            continue
        if row[lang_idx] != 'en': 
            continue

        valid_works[wid] = {
            'doi': row[doi_idx] or "",
            'title': row[title_idx],
            'date': date,
            'abstract': invert_abstract(row[abs_idx])
        }

print(f"  Valid papers: {len(valid_works)}")

# Collect all institution IDs
inst_ids = set()
with open('../openalex_data/works_authorships.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    wid_idx = header.index('work_id')
    inst_idx = header.index('institution_id')
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt % 100_000 == 0:
            print(f"    Processed {cnt:,} rows (works_authorships.tsv - collecting institutions)")

        wid = row[wid_idx].split('/')[-1]
        if wid not in valid_works: 
            continue
        iid = row[inst_idx]
        if iid and iid != 'null':
            inst_ids.add(iid.split('/')[-1])

# Load institution countries: inst_id → country_code
print("Step 5: Loading institution countries...")
inst_country = {}
with open('../openalex_data/institutions.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    iid_idx = header.index('id')
    cc_idx = header.index('country_code')
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt % 100_000 == 0:
            print(f"    Processed {cnt:,} rows (institutions.tsv)")

        iid = row[iid_idx].split('/')[-1]
        if iid in inst_ids:
            cc = row[cc_idx]
            if cc: 
                inst_country[iid] = cc

# Collect countries for each paper (deduplicated)
print("Step 6: Collecting paper countries...")
paper_countries = defaultdict(set)
with open('../openalex_data/works_authorships.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    wid_idx = header.index('work_id')
    inst_idx = header.index('institution_id')
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt % 100_000 == 0:
            print(f"    Processed {cnt:,} rows (works_authorships.tsv - collecting countries)")

        wid = row[wid_idx].split('/')[-1]
        if wid not in valid_works: 
            continue
        iid = row[inst_idx]
        if iid and iid != 'null':
            iid = iid.split('/')[-1]
            if iid in inst_country:
                paper_countries[wid].add(inst_country[iid])

# Get top-scoring topic
print("Step 7: Getting top-scoring topics...")
work_topic = {}
with open('../openalex_data/works_topics.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    wid_idx = header.index('work_id')
    tid_idx = header.index('topic_id')
    score_idx = header.index('score')
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt % 100_000 == 0:
            print(f"    Processed {cnt:,} rows (works_topics.tsv)")

        wid = row[wid_idx].split('/')[-1]
        if wid not in valid_works: 
            continue
        tid = row[tid_idx].split('/')[-1]
        score = float(row[score_idx])
        if wid not in work_topic or score > work_topic[wid][0]:
            work_topic[wid] = (score, tid)

# Load topic → domain, field
topic_info = {}
with open('../openalex_data/topics.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    tid_idx = header.index('id')
    dom_idx = header.index('domain_display_name')
    fld_idx = header.index('field_display_name')
    cnt = 0
    for row in reader:
        cnt += 1
        if cnt % 100_000 == 0:
            print(f"    Processed {cnt:,} rows (topics.tsv)")

        tid = row[tid_idx].split('/')[-1]
        topic_info[tid] = (row[dom_idx], row[fld_idx])

# Final output CSV
print("Step 8: Writing final results...")
with open('Q1_papers_v0.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'work_id', 'doi', 'title', 'publication_date',
        'abstract', 'country', 'domain', 'field', 'journal', 'is_oa'
    ])

    for wid, info in valid_works.items():
        # Countries
        countries = sorted(paper_countries[wid])
        country_str = ';'.join(countries) if countries else ""

        # Topic
        domain = field = ""
        if wid in work_topic:
            _, tid = work_topic[wid]
            if tid in topic_info:
                domain, field = topic_info[tid]

        # Journal name
        sid = work_source.get(wid, '')
        journal = q1_map.get(sid, "Unknown Journal")

        writer.writerow([
            f"https://openalex.org/{wid}",
            info['doi'],
            info['title'],
            info['date'],
            info['abstract'],
            country_str,
            domain,
            field,
            journal,
            str(work_oa.get(wid, False))
        ])

print(f"Complete! Exported {len(valid_works)} papers → Q1_papers_v0.csv")