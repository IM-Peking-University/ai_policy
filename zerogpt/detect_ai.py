import requests
import json
import pandas as pd
from datetime import datetime
import time
import sys

# Configuration
API_URL = "https://api.zerogpt.com/api/detect/detectText"
API_KEY = ""
INPUT_CSV = r"ai_policy\data\paper_2021-2025.csv"
OUTPUT_CSV = r"ai_policy\results\detection_results.csv"
PARTIAL_CSV = r"ai_policy\results\detection_results_partial.csv"
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2025, 6, 30)
SAMPLES_PER_MONTH = 30
SLEEP_SECONDS = 1
REQUEST_TIMEOUT = 60
MAX_CHARS = 50000  # API limit per detection

headers = {
    'ApiKey': API_KEY,
    'Content-Type': 'application/json'
}


def clean_text(text):
    if text is None:
        return None

    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return None
    text = text.strip()
    if not text:
        return None

    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text


def detect_ai(text):
    if text is None:
        return json.dumps({"error": "Empty text"})
    payload = json.dumps({"input_text": text})
    try:
        response = requests.post(API_URL, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": str(e)})


# Load data
try:
    df = pd.read_csv(
        INPUT_CSV,
        dtype={"paper_id": str, "abstract": str, "date": str},
        low_memory=False
    )
except Exception as e:
    print(f"Failed to read CSV: {e}")
    sys.exit(1)


df['abstract'] = df['abstract'].fillna('').astype(str).str.strip()
df = df[df['abstract'] != '']


df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)]

if df.empty:
    print("No records found in the date range after filtering. Exiting.")
    sys.exit(0)


df['year_month'] = df['date'].dt.to_period('M')
months = pd.period_range(start='2021-01', end='2025-06', freq='M')


samples = []
for ym in months:
    g = df[df['year_month'] == ym]
    if len(g) == 0:
        continue
    n = min(SAMPLES_PER_MONTH, len(g))
    samples.append(g.sample(n, random_state=42))

if not samples:
    print("No samples could be drawn. Exiting.")
    sys.exit(0)

sampled_df = pd.concat(samples, ignore_index=True)

print(f"Starting AI detection for {len(sampled_df)} abstracts...")
results = []

for idx, row in sampled_df.iterrows():
    text = clean_text(row.get('abstract'))
    detection = detect_ai(text)

    date_val = row.get('date')
    if pd.isna(date_val):
        date_str = ''
    else:
        try:
            date_str = date_val.strftime('%Y-%m-%d')
        except Exception:
            date_str = ''

    results.append({
        'paper_id': row.get('paper_id', ''),
        'abstract': row.get('abstract', ''),
        'date': date_str,
        'detection_result': detection
    })

    if (idx + 1) % 25 == 0:
        pd.DataFrame(results).to_csv(PARTIAL_CSV, index=False)
        print(f"Saved progress at {idx + 1} rows -> {PARTIAL_CSV}")

    time.sleep(SLEEP_SECONDS)


pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"Detection complete. Results saved to: {OUTPUT_CSV}")
