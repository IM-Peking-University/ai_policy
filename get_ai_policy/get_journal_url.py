import os
import re
import json
import time
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
INPUT_CSV = './ai_policy_gpt_enhanced_20250924_033445_merged_20250924_235414.csv'
ALL_DATA_JSON = './all_data.json'
OUTPUT_CSV = None

# Serper API Key
SERPER_API_KEY = "YOUR_SERPER_API_KEY_HERE"  # Replaced for safety

# Core Search Logic
def serper_search(api_key: str, query: str, timeout: int = 20) -> dict | None:
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json={"q": query}, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None

def score_candidate(url: str, journal: dict) -> float:
    score = 0.0
    u = (url or "").lower()
    prefer = [
        "springer.com", "springeropen.com", "nature.com", "wiley.com", "onlinelibrary.wiley.com",
        "tandfonline.com", "elsevier.com", "sciencedirect.com", "cell.com",
        "ieee.org", "ieeexplore.ieee.org", "acm.org", "dl.acm.org",
        "oup.com", "academic.oup.com", "cambridge.org", "journals.cambridge.org",
        "sagepub.com", "mdpi.com", "frontiersin.org", "hindawi.com",
        "oxfordjournals.org", "rsc.org", "iop.org", "journals.aps.org", "aip.org",
        "bjournals.org", "amegroups.com", "kluweronline.com"
    ]
    for d in prefer:
        if d in u:
            score += 5
    avoid = [
        "researchgate.net", "semanticscholar.org", "readcube.com", "scimagojr.com",
        "sci-hub", "baike.baidu.com", "wikipedia.org", "scholar.google.",
        "dblp.org", "cnki.net", "wanfangdata", "core.ac.uk",
        "worldcat.org", "crossref.org", "issn.org"
    ]
    for d in avoid:
        if d in u:
            score -= 5
    if "/journal" in u or "/journals" in u:
        score += 3
    if any(k in u for k in ["/home", "/about", "/aims", "/scope"]):
        score += 1
    issn = (journal.get("issn") or "").replace(" ", "").lower()
    if issn and issn in u:
        score += 4
    title = (journal.get("title") or "").lower()
    for token in title.replace("-", " ").replace("/", " ").split():
        if token and token in u:
            score += 0.6
    return score

def pick_homepage_from_serper(resp: dict, journal: dict) -> str | None:
    if not resp:
        return None
    candidates = []
    for block in ("organic", "knowledgeGraph", "answerBox", "topStories", "peopleAlsoSearchFor"):
        v = resp.get(block)
        if isinstance(v, list):
            for item in v:
                url = item.get("link") or item.get("url")
                if isinstance(url, str):
                    candidates.append(url)
        elif isinstance(v, dict):
            url = v.get("link") or v.get("url")
            if isinstance(url, str):
                candidates.append(url)
    
    # Deduplicate
    seen, uniq = set(), []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    if not uniq:
        return None
    ranked = sorted(uniq, key=lambda u: score_candidate(u, journal), reverse=True)
    return ranked[0]

def build_query(title: str | None, issn: str | None, publisher: str | None) -> str:
    parts = []
    if title:
        parts.append(title)
    if issn:
        parts.append(issn)
    if publisher:
        parts.append(publisher)
    parts.append("journal homepage official site")
    return " ".join([p for p in parts if p])

def find_homepage(title: str | None, issn: str | None, publisher: str | None) -> str | None:
    q = build_query(title, issn, publisher)
    resp = serper_search(SERPER_API_KEY, q)
    return pick_homepage_from_serper(resp, {"title": title, "issn": issn, "publisher": publisher})

# Metadata Extraction
def normalize_name(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def choose_first(d: dict, keys: list[str]) -> str | None:
    for k in keys:
        if k in d and d[k]:
            return str(d[k])
    return None

def load_needed_metadata(json_path: str, needed_names: set[str]) -> dict[str, dict]:
    meta = {}
    found = 0
    total_needed = len(needed_names)
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            name = choose_first(obj, ["journal_name", "journalName", "title", "name"])
            nname = normalize_name(name)
            if nname in needed_names and nname not in meta:
                issn = choose_first(obj, ["issn", "ISSN", "print_issn", "printISSN"])
                eissn = choose_first(obj, ["eissn", "EISSN", "online_issn", "onlineISSN"])
                publisher = choose_first(obj, ["publisher", "Publisher", "publisher_name", "publisherName", "出版商"])
                meta[nname] = {
                    "title": name or "",
                    "issn": (issn or eissn or "").strip(),
                    "eissn": (eissn or "").strip(),
                    "publisher": (publisher or "").strip(),
                }
                found += 1
                if found >= total_needed:
                    break
    return meta

# Main Process
def update_failed_urls(input_csv: str, all_json: str, threads: int = 6) -> str:
    df = pd.read_csv(input_csv, encoding="utf-8")
    
    name_col = "journal_name" if "journal_name" in df.columns else ("journalName" if "journalName" in df.columns else None)
    if not name_col:
        raise KeyError(f"Journal name column not found: {list(df.columns)}")

    if "status" not in df.columns:
        raise KeyError("Status column not found.")

    failed_mask = df["status"].fillna("failed").str.lower() != "success"
    df_failed = df[failed_mask].copy()
    if df_failed.empty:
        print("No failed records to update.")
        return input_csv

    needed_names = set(df_failed[name_col].dropna().map(normalize_name))
    needed_names.discard("")

    meta_map = load_needed_metadata(all_json, needed_names)

    tasks = []
    for idx, row in df_failed.iterrows():
        title = str(row.get(name_col, "")).strip()
        nname = normalize_name(title)
        meta = meta_map.get(nname, {})
        issn = meta.get("issn", "")
        publisher = meta.get("publisher", "")
        tasks.append((idx, title, issn, publisher))

    print(f"Failed records to update: {len(tasks)}, Threads: {threads}")

    results = {}
    def worker(arg):
        idx, title, issn, publisher = arg
        try:
            url = find_homepage(title or None, issn or None, publisher or None)
            time.sleep(0.2) # Rate limit
            return idx, url or ""
        except Exception as e:
            return idx, ""

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        done = 0
        for fu in as_completed(futs):
            i, u = fu.result()
            results[i] = u
            done += 1
            if done % 50 == 0:
                print(f"Progress: {done}/{len(tasks)}")

    if "url" not in df.columns:
        df["url"] = ""

    updated = 0
    for i, new_url in results.items():
        if new_url:
            df.at[i, "url"] = new_url
            updated += 1

    print(f"Completed. Updated {updated} URLs.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUTPUT_CSV or os.path.splitext(INPUT_CSV)[0] + f"_url_refreshed_{ts}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Output file:", out_csv)
    return out_csv

if __name__ == "__main__":
    update_failed_urls(INPUT_CSV, ALL_DATA_JSON, threads=6)