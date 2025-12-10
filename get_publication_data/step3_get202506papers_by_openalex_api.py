import requests
import csv
import json
import time
import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
API_KEY = ""
INPUT_Q1 = 'Q1_matched_final.csv'
OUTPUT_FILE = '2025_06.csv'
MAX_PROCESSES = 4
PER_PAGE = 200
SLEEP_PER_PAGE = 0.15

def create_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def load_q1_mapping():
    q1_source_ids = set()
    q1_name_map = {}
    with open(INPUT_Q1, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sid = row['df_id'].split('/')[-1]
            q1_source_ids.add(sid)
            q1_name_map[sid] = row['q1_name']
    return q1_source_ids, q1_name_map

def generate_date_ranges():
    start = datetime(2025, 6, 1)
    end = datetime(2025, 6, 30)
    ranges = []
    current = start
    while current <= end:
        batch_end = current + timedelta(days=5)
        if batch_end > end:
            batch_end = end
        ranges.append((current.strftime('%Y-%m-%d'), batch_end.strftime('%Y-%m-%d')))
        current = batch_end + timedelta(days=1)
    return ranges

def process_date_range(args):
    start_date, end_date, q1_source_ids, q1_name_map = args
    print(f"  [Process {os.getpid()}] Processing {start_date} ~ {end_date}")
    
    session = create_session()
    url = "https://api.openalex.org/works"
    cursor = "*"
    papers = []
    page = 0

    while cursor:
        page += 1
        params = {
            "filter": f"from_publication_date:{start_date},to_publication_date:{end_date},language:en",
            "per_page": PER_PAGE,
            "cursor": cursor,
            "select": "id,doi,title,publication_date,abstract_inverted_index,authorships,primary_location,topics",
            "api_key": API_KEY
        }
        success = False
        for attempt in range(3):
            try:
                resp = session.get(url, params=params, timeout=30).json()
                results = resp.get("results", [])
                print(f"    [{start_date}] Page {page}: {len(results)} items")

                for p in results:
                    primary = p.get("primary_location") or {}
                    source = primary.get("source") or {}
                    sid = source.get("id", "").split("/")[-1]
                    if sid not in q1_source_ids: continue

                    countries = {
                        i.get("country_code")
                        for a in p.get("authorships", [])
                        for i in a.get("institutions", [])
                        if i.get("country_code")
                    }
                    domain = field = ""
                    topics = p.get("topics") or []
                    if topics:
                        best = max(topics, key=lambda t: t.get("score", 0), default={})
                        domain = best.get("domain", {}).get("display_name", "")
                        field = best.get("field", {}).get("display_name", "")
                    is_oa = primary.get("is_oa", False)
                    abstract = ""
                    inv = p.get("abstract_inverted_index")
                    if inv and isinstance(inv, dict):
                        try:
                            max_pos = max(max(ps) for ps in inv.values() if ps)
                            words = [''] * (max_pos + 1)
                            for w, ps in inv.items():
                                for pos in ps:
                                    if pos < len(words): words[pos] = w
                            abstract = ' '.join(filter(None, words))
                        except: pass

                    papers.append({
                        'work_id': p.get('id', ''),
                        'doi': p.get('doi', ''),
                        'title': p.get('title', ''),
                        'publication_date': p.get('publication_date', ''),
                        'abstract': abstract,
                        'country': ';'.join(sorted(countries)) if countries else '',
                        'domain': domain,
                        'field': field,
                        'journal': q1_name_map.get(sid, "Unknown Journal"),
                        'is_oa': str(is_oa)
                    })

                meta = resp.get("meta") or {}
                cursor = meta.get("next_cursor")
                success = True
                break

            except Exception as e:
                print(f"    [{start_date}] Failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        
        if not success:
            print(f"    [{start_date}] Page failed, skipping")
        
        if not cursor:
            print(f"  [{start_date}] Complete")
            break
        
        time.sleep(SLEEP_PER_PAGE)

    return papers

# Main program
if __name__ == '__main__':
    print("Loading Q1 journal mapping...")
    q1_source_ids, q1_name_map = load_q1_mapping()
    print(f"  Total {len(q1_source_ids)} Q1 journals")

    DATE_RANGES = generate_date_ranges()
    print(f"  Divided into {len(DATE_RANGES)} time periods: {DATE_RANGES}")

    all_papers = []
    print(f"\nStarting {MAX_PROCESSES} parallel processes...")

    # Pack arguments
    tasks = [(dr[0], dr[1], q1_source_ids, q1_name_map) for dr in DATE_RANGES]

    with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
        futures = [executor.submit(process_date_range, task) for task in tasks]
        for future in as_completed(futures):
            batch_papers = future.result()
            all_papers.extend(batch_papers)
            print(f"  Total collected: {len(all_papers)}")

    print(f"\nBefore deduplication: {len(all_papers)}")
    unique_papers = {p['work_id']: p for p in all_papers}.values()
    print(f"After deduplication: {len(unique_papers)}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'work_id','doi','title','publication_date','abstract',
            'country','domain','field','journal','is_oa'
        ])
        w.writeheader()
        w.writerows(unique_papers)

    print(f"\nComplete! Results saved to {OUTPUT_FILE}")