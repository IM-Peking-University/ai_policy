import os
import json
import time
import csv
from collections import Counter

import openai


OPENAI_API_KEY = ""

client = openai.OpenAI(api_key=OPENAI_API_KEY)


# Policy classification categories
POLICY_TYPES = [
    "Strict Prohibition",   # Prohibits AI use
    "Open",                  # Allows AI use without disclosure
    "Disclosure Required",   # Requires AI disclosure
    "Not Mentioned",         # No clear AI policy
]


SYSTEM_PROMPT = (
    "You are an expert in academic journal policy analysis. Given a policy text about AUTHORS' use of AI in academic writing, output a structured judgment.\n"
    "Rules:\n"
    "- Category (choose EXACTLY ONE of the following labels):\n"
    "  1) Strict Prohibition: clearly prohibits authors from using AI for academic writing, including editing, translation, or content generation.\n"
    "  2) Open: allows authors to use AI AND explicitly does NOT require disclosure.\n"
    "  3) Disclosure Required: allows authors to use AI BUT requires disclosure in the manuscript (extract disclosure location).\n"
    "  4) Not Mentioned: the text is not about AUTHORS' use of AI (e.g., only about peer review or editors), or insufficient to determine.\n"
    "- Disclosure location: ONLY when category is \"Disclosure Required\", summarize a free-text location from the policy (do NOT choose from fixed options). Examples include but are not limited to: Methods section, Acknowledgements, Cover letter, Title page, End-of-manuscript statement, Submission system, etc. If multiple locations are acceptable, summarize briefly; if not specified, return an empty string.\n"
    "- Reason: one or two sentences summarizing the basis for the classification; paraphrase or quote key phrases succinctly.\n"
    "- Notes: Mentions that AI cannot be listed as an author should be treated as background and MUST NOT change the four-class decision. If the text only addresses peer review/editors' use of AI but not authors, classify as \"Not Mentioned\". When multiple statements exist, prioritize explicit requirements about AUTHORS' use in writing.\n\n"
    "Output format (STRICT): Return ONLY a JSON object with fields:\n"
    "{\"category\": one of [\"Strict Prohibition\", \"Open\", \"Disclosure Required\", \"Not Mentioned\"], \"disclosure_location\": <free text or empty string>, \"reason\": <short justification>}\n"
)


def analyze_policy(text: str, max_retries: int = 3) -> dict:
    """Classify policy text using AI model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    last_err = None
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            # Expect strict JSON
            data = json.loads(content)

            category = str(data.get("category", "")).strip()
            disclosure_location = str(data.get("disclosure_location", "")).strip()
            reason = str(data.get("reason", "")).strip()

            if category not in POLICY_TYPES:
                # Normalize likely variants to the required English labels
                mapping = {
                    "Strict Prohibition": "Strict Prohibition",
                    "Prohibitive": "Strict Prohibition",
                    "Prohibited": "Strict Prohibition",
                    "Prohibit": "Strict Prohibition",
                    "Open Permission": "Open",
                    "Open Use": "Open",
                    "Disclosure-Required": "Disclosure Required",
                    "Require Disclosure": "Disclosure Required",
                    "Not Mentioned": "Not Mentioned",
                    "Undefined": "Not Mentioned",
                }
                category = mapping.get(category, "Not Mentioned")


            if category != "Disclosure Required":
                disclosure_location = ""

            return {
                "category": category,
                "disclosure_location": disclosure_location,
                "reason": reason,
            }
        except Exception as e:
            last_err = e
            time.sleep(1.5)

    raise RuntimeError(f"Model call/parse failed: {last_err}")


def validate_classification(text: str) -> dict:
    """Validate classification with majority voting."""
    results = [analyze_policy(text) for _ in range(3)]

    key_counts = Counter((r["category"], r["disclosure_location"]) for r in results)
    (final_cat, final_loc), _ = key_counts.most_common(1)[0]

    reasons = [r["reason"] for r in results if r["category"] == final_cat and r["disclosure_location"] == final_loc]
    final_reason = Counter(reasons).most_common(1)[0][0] if reasons else results[0]["reason"]

    return {
        "category": final_cat,
        "disclosure_location": final_loc if final_cat == "Disclosure Required" else "",
        "reason": final_reason,
    }


def process_file(input_path: str, output_path: str):
    """Process policy texts and save classification results."""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    results = []
    for i, policy in enumerate(lines, 1):
        print(f"Processing policy {i}/{len(lines)} ...")
        try:
            record = validate_classification(policy)
        except Exception as e:
            record = {"category": "Not Mentioned", "disclosure_location": "", "reason": f"classification failed: {e}"}

        results.append({
            "ID": i,
            "Category": record["category"],
            "Disclosure Location": record["disclosure_location"],
            "Reason": record["reason"],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Category", "Disclosure Location", "Reason"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Done: results saved to {output_path}")


if __name__ == "__main__":
    process_file("data/policy_texts.txt", "results/policy_classification.csv")