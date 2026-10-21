"""Shared utilities for the HaluEval baseline-vs-RAG experiment."""
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATASET_PATH = DATA_DIR / "qa_data.json"
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY", "").strip()
BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "100"))
TOP_K = int(os.getenv("TOP_K", "2"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))
RETRY_BASE_SECONDS = float(os.getenv("RETRY_BASE_SECONDS", "10"))
MAX_RETRY_WAIT_SECONDS = float(os.getenv("MAX_RETRY_WAIT_SECONDS", "120"))
MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("MIN_REQUEST_INTERVAL_SECONDS", "30"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))

_last_request_started = 0.0


def ensure_setup():
    if not API_KEY or API_KEY == "your_actual_groq_api_key":
        raise RuntimeError("GROQ_API_KEY is missing or still a placeholder in .env.")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Put qa_data.json here: {DATASET_PATH}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json_records(path=DATASET_PATH):
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"Dataset is empty: {path}")
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            records = next((obj[key] for key in ("data", "records", "examples", "qa_data")
                            if isinstance(obj.get(key), list)), [obj])
        else:
            raise ValueError("JSON must contain a list of records or an object.")
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("No records found in qa_data.json.")
    missing = {"knowledge", "question"} - set(records[0].keys())
    if missing:
        raise ValueError(f"Missing fields {sorted(missing)}; found {sorted(records[0].keys())}")
    return records


def prepare_records():
    records = load_json_records()
    if SAMPLE_SIZE > 0:
        records = records[:SAMPLE_SIZE]
    output = []
    for index, record in enumerate(records, 1):
        question = str(record.get("question", "") or "").strip()
        if not question:
            continue
        record_id = record.get("question_id", record.get("id"))
        if record_id is None or str(record_id).strip() == "":
            record_id = f"Q{index:04d}"
        output.append({
            "question_id": str(record_id),
            "knowledge": str(record.get("knowledge", "") or "").strip(),
            "question": question,
            "reference_answer": str(record.get("right_answer", "") or "").strip(),
            "dataset_hallucinated_answer": str(record.get("hallucinated_answer", "") or "").strip(),
        })
    if not output:
        raise ValueError("No records with non-empty questions were found.")
    return output


def _parse_seconds(value):
    if not value:
        return None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*", str(value).lower())
    return float(match.group(1)) if match else None


def _retry_wait_seconds(response, attempt, error_message=""):
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
                      error_message, flags=re.IGNORECASE)
    if match:
        return min(max(float(match.group(1)) + 1.0, 1.0), MAX_RETRY_WAIT_SECONDS)
    retry_after = _parse_seconds(response.headers.get("Retry-After"))
    if retry_after is not None:
        return min(max(retry_after, 1.0), MAX_RETRY_WAIT_SECONDS)
    reset_values = []
    for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        value = _parse_seconds(response.headers.get(name, ""))
        if value is not None:
            reset_values.append(value)
    if reset_values:
        return min(max(max(reset_values), 1.0), MAX_RETRY_WAIT_SECONDS)
    return min(RETRY_BASE_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_WAIT_SECONDS)


def _pace_request_start():
    global _last_request_started
    now = time.monotonic()
    if _last_request_started:
        remaining = MIN_REQUEST_INTERVAL_SECONDS - (now - _last_request_started)
        if remaining > 0:
            print(f"Request pacing: waiting {remaining:.1f}s.")
            time.sleep(remaining)
    _last_request_started = time.monotonic()


def call_llm(system_prompt, user_prompt):
    if not API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in .env.")
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": "low",
        "include_reasoning": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    network_failures = 0
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        _pace_request_start()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                details = response.text[:1000]
                try:
                    err = response.json().get("error", {})
                except ValueError:
                    err = {}
                code = str(err.get("code", "")).lower()
                error_type = str(err.get("type", "")).lower()
                message = str(err.get("message", details)).lower()
                if (code == "insufficient_quota" or "insufficient_quota" in message
                        or "credits exhausted" in message or "quota exceeded" in message):
                    raise RuntimeError(
                        "Groq reports an account quota/credit limit. Check Groq Console. "
                        f"Details: {details}"
                    )
                last_error = RuntimeError(f"Groq rate limit ({code or error_type or '429'}): {details}")
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Rate limit persisted after {MAX_RETRIES} attempts. "
                        "Check the model's RPM/TPM limits in Groq Console. "
                        f"Details: {details}"
                    )
                wait = _retry_wait_seconds(response, attempt, message)
                print(f"Groq rate limit. Retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s.")
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                last_error = RuntimeError(
                    f"Groq server error {response.status_code}: {response.text[:500]}"
                )
                if attempt < MAX_RETRIES:
                    wait = min(RETRY_BASE_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_WAIT_SECONDS)
                    print(f"Groq server error. Retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s.")
                    time.sleep(wait)
                    continue
                break

            response.raise_for_status()
            body = response.json()
            return body["choices"][0]["message"]["content"].strip()

        except RuntimeError:
            raise
        except requests.RequestException as exc:
            last_error = exc
            network_failures += 1
            if network_failures >= 3:
                raise RuntimeError(
                    "Could not reach Groq after 3 network attempts. Check internet/DNS. "
                    "Completed CSV rows should remain saved. "
                    f"Last error: {exc}"
                ) from exc
            if attempt < MAX_RETRIES:
                wait = min(5 * (2 ** (network_failures - 1)), 20)
                print(f"Network error ({network_failures}/3). Retrying in {wait:.1f}s: {exc}")
                time.sleep(wait)
        except (ValueError, KeyError, IndexError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                wait = min(RETRY_BASE_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_WAIT_SECONDS)
                print(f"Invalid API response. Retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s.")
                time.sleep(wait)

    raise RuntimeError(f"Groq request failed after {MAX_RETRIES} attempts: {last_error}")


def output_csv_path(name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def load_existing_rows(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path, keep_default_na=False)
    if "question_id" not in df.columns:
        return {}
    return {str(row["question_id"]): row.to_dict() for _, row in df.iterrows()}


def save_rows(path, rows):
    pd.DataFrame(list(rows.values())).to_csv(path, index=False, encoding="utf-8-sig")


def build_retriever(records):
    snippets, seen = [], set()
    for record in records:
        text = record["knowledge"]
        if text and text not in seen:
            snippets.append({"document_id": f"DOC{len(snippets) + 1:05d}", "text": text})
            seen.add(text)
    if not snippets:
        raise ValueError("No non-empty knowledge snippets available for retrieval.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([doc["text"] for doc in snippets])
    return snippets, vectorizer, matrix


def retrieve(query, snippets, vectorizer, matrix, top_k=TOP_K):
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    indices = scores.argsort()[::-1][:max(1, top_k)]
    return [(snippets[int(index)], float(scores[int(index)])) for index in indices]


def annotation_columns(prefix):
    return {
        f"{prefix}_correctness": "",
        f"{prefix}_hallucination": "",
        f"{prefix}_faithfulness": "",
        f"{prefix}_annotation_notes": "",
    }
