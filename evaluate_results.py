"""Evaluate paired baseline and RAG responses with an LLM judge.

Run from the research project folder:
    python evaluate_results.py

Inputs (kept unchanged):
    results/baseline_responses.csv
    results/rag_responses.csv

Outputs:
    results/evaluated_baseline_responses.csv
    results/evaluated_rag_responses.csv
    results/evaluation_summary.csv

Important: LLM-judge labels are preliminary annotations, not ground truth.
Manually review uncertain labels and a sample of all labels before reporting results.
"""
import json
import re
import time
from pathlib import Path

import pandas as pd

from common import (
    MODEL,
    TEMPERATURE,
    ensure_setup,
    call_llm,
    output_csv_path,
)

BASELINE_INPUT = output_csv_path("baseline_responses.csv")
RAG_INPUT = output_csv_path("rag_responses.csv")
BASELINE_OUTPUT = output_csv_path("evaluated_baseline_responses.csv")
RAG_OUTPUT = output_csv_path("evaluated_rag_responses.csv")
SUMMARY_OUTPUT = output_csv_path("evaluation_summary.csv")

# Keep prompts concise to reduce token use and rate-limit pressure.
JUDGE_SYSTEM_PROMPT = """
You are a careful research evaluator for a study comparing an LLM baseline with
retrieval-augmented generation (RAG). Evaluate only the provided record.
Do not assume a reference answer is infallible: if it conflicts with the
supplied knowledge or the evidence is insufficient, mark the issue as unclear
and explain briefly. Do not reward an answer merely for matching a phrase.
Return ONLY one valid JSON object, without Markdown fences or extra text.
"""

JUDGE_USER_TEMPLATE = """
Evaluate these two answers to the same question.

QUESTION:
{question}

REFERENCE ANSWER (may be incomplete or imperfect):
{reference_answer}

SUPPLIED KNOWLEDGE (available for post-hoc factual checking; the baseline did
NOT see this during generation):
{knowledge}

BASELINE ANSWER:
{baseline_response}

RAG RETRIEVED CONTEXT (what the RAG model could use):
{retrieved_context}

RAG ANSWER:
{rag_response}

Apply these definitions:
- correctness: "Correct", "Partially correct", "Incorrect", or
  "Unanswerable/insufficient evidence".
- hallucination: "Yes", "No", or "Unclear". Use "Yes" if the answer makes a
  material factual claim contradicted by the supplied evidence or unsupported
  by the evidence available to that system. Do not automatically label every
  wrong answer a hallucination; explain the reason.
- faithfulness: "Supported", "Partially supported", "Unsupported", or
  "Not applicable". For RAG, judge support against RAG RETRIEVED CONTEXT.
  For baseline, use supplied knowledge/reference for post-hoc checking, while
  remembering that the baseline did not receive that knowledge at generation.
- retrieval_relevance: "Relevant and sufficient", "Relevant but incomplete",
  "Irrelevant", or "No useful evidence". Judge whether the retrieved context
  contains enough relevant evidence to answer the question.

An abstention is not automatically a hallucination. If the answer abstains
even though the relevant evidence is present, mark correctness "Incorrect"
and explain the unnecessary abstention. If the evidence truly cannot settle
the question, use "Unanswerable/insufficient evidence".

Return JSON with exactly these keys:
{{
  "baseline_correctness": "...",
  "baseline_hallucination": "...",
  "baseline_faithfulness": "...",
  "baseline_annotation_notes": "...",
  "rag_correctness": "...",
  "rag_hallucination": "...",
  "rag_faithfulness": "...",
  "rag_annotation_notes": "...",
  "retrieval_relevance": "..."
}}
"""


def text(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def parse_json_object(raw):
    """Parse JSON even if a judge accidentally surrounds it with code fences."""
    raw = text(raw)
    raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Judge did not return a valid JSON object: {raw[:300]}")


def normalized_label(value, allowed, fallback="Unclear"):
    value = text(value)
    for option in allowed:
        if value.casefold() == option.casefold():
            return option
    return fallback


def load_csv(path, required_columns, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found: {path}\n"
            "Put the original CSV in your project's results folder."
        )
    df = pd.read_csv(path, keep_default_na=False)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{label} CSV is missing columns: {sorted(missing)}")
    if df["question_id"].duplicated().any():
        duplicates = df.loc[df["question_id"].duplicated(), "question_id"].tolist()
        raise ValueError(f"{label} CSV has duplicate question IDs, e.g. {duplicates[:5]}")
    return df


def annotations_complete(row, prefix):
    required = [
        f"{prefix}_correctness",
        f"{prefix}_hallucination",
        f"{prefix}_faithfulness",
        f"{prefix}_annotation_notes",
    ]
    return all(text(row.get(col, "")) for col in required)


def save_outputs(baseline_df, rag_df):
    baseline_df.to_csv(BASELINE_OUTPUT, index=False, encoding="utf-8-sig")
    rag_df.to_csv(RAG_OUTPUT, index=False, encoding="utf-8-sig")


def build_summary(baseline_df, rag_df):
    rows = []
    metrics = [
        ("Correctness", "baseline_correctness", "rag_correctness",
         {"Correct", "Partially correct", "Incorrect", "Unanswerable/insufficient evidence"}),
        ("Hallucination", "baseline_hallucination", "rag_hallucination",
         {"Yes", "No", "Unclear"}),
        ("Faithfulness", "baseline_faithfulness", "rag_faithfulness",
         {"Supported", "Partially supported", "Unsupported", "Not applicable"}),
    ]
    for metric, base_col, rag_col, allowed in metrics:
        for label in sorted(allowed):
            rows.append({
                "metric": metric,
                "label": label,
                "baseline_count": int((baseline_df[base_col].astype(str).str.casefold()
                                       == label.casefold()).sum()),
                "rag_count": int((rag_df[rag_col].astype(str).str.casefold()
                                  == label.casefold()).sum()),
                "denominator": len(baseline_df),
                "note": "Preliminary LLM-judge labels; review before reporting."
            })
    rows.extend([
        {"metric": "Responses", "label": "Non-empty", 
         "baseline_count": int(baseline_df["baseline_response"].map(text).ne("").sum()),
         "rag_count": int(rag_df["rag_response"].map(text).ne("").sum()),
         "denominator": len(baseline_df),
         "note": "Generation completeness, not a quality metric."},
        {"metric": "Retrieval", "label": "Relevant and sufficient",
         "baseline_count": "",
         "rag_count": int((rag_df["retrieval_relevance"].astype(str).str.casefold()
                           == "relevant and sufficient").sum()),
         "denominator": len(rag_df),
         "note": "Judged relevance of retrieved context."},
    ])
    pd.DataFrame(rows).to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")


def main():
    ensure_setup()
    baseline_df = load_csv(
        BASELINE_INPUT,
        ["question_id", "question", "reference_answer", "knowledge",
         "baseline_response"],
        "Baseline",
    )
    rag_df = load_csv(
        RAG_INPUT,
        ["question_id", "question", "reference_answer", "rag_response",
         "retrieved_context"],
        "RAG",
    )

    if set(baseline_df["question_id"]) != set(rag_df["question_id"]):
        raise ValueError("Baseline and RAG files must contain exactly the same question IDs.")

    # Align rows by question_id, not by their order in the CSVs.
    rag_df = rag_df.set_index("question_id").loc[baseline_df["question_id"]].reset_index()

    # Resume existing evaluated files when available, but never alter raw inputs.
    if BASELINE_OUTPUT.exists() and RAG_OUTPUT.exists():
        evaluated_base = pd.read_csv(BASELINE_OUTPUT, keep_default_na=False)
        evaluated_rag = pd.read_csv(RAG_OUTPUT, keep_default_na=False)
        if (set(evaluated_base.get("question_id", [])) == set(baseline_df["question_id"])
                and set(evaluated_rag.get("question_id", [])) == set(rag_df["question_id"])):
            baseline_df = baseline_df.drop(
                columns=[c for c in baseline_df.columns if c.startswith("baseline_")],
                errors="ignore",
            ).merge(
                evaluated_base[["question_id"] + [c for c in evaluated_base.columns
                    if c.startswith("baseline_")]],
                on="question_id", how="left",
            )
            rag_df = rag_df.drop(
                columns=[c for c in rag_df.columns if c.startswith("rag_") or c == "retrieval_relevance"],
                errors="ignore",
            ).merge(
                evaluated_rag[["question_id"] + [c for c in evaluated_rag.columns
                    if c.startswith("rag_") or c == "retrieval_relevance"]],
                on="question_id", how="left",
            )

    for col in [
        "baseline_correctness", "baseline_hallucination", "baseline_faithfulness",
        "baseline_annotation_notes"
    ]:
        if col not in baseline_df.columns:
            baseline_df[col] = ""
    for col in [
        "rag_correctness", "rag_hallucination", "rag_faithfulness",
        "rag_annotation_notes", "retrieval_relevance"
    ]:
        if col not in rag_df.columns:
            rag_df[col] = ""

    print(f"Judge model configured in .env: {MODEL}")
    print(f"Temperature: {TEMPERATURE}; paired questions: {len(baseline_df)}")
    print("Note: original response CSVs will not be overwritten.")

    base_index = baseline_df.set_index("question_id", drop=False)
    rag_index = rag_df.set_index("question_id", drop=False)
    total = len(baseline_df)

    for index, row in baseline_df.iterrows():
        qid = text(row["question_id"])
        b_row = base_index.loc[qid]
        r_row = rag_index.loc[qid]

        if (annotations_complete(b_row, "baseline")
                and annotations_complete(r_row, "rag")
                and text(r_row.get("retrieval_relevance", ""))):
            print(f"[{index + 1}/{total}] {qid}: already evaluated; skipping")
            continue

        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=text(b_row.get("question", "")),
            reference_answer=text(b_row.get("reference_answer", "")),
            knowledge=text(b_row.get("knowledge", ""))[:6000],
            baseline_response=text(b_row.get("baseline_response", ""))[:4000],
            retrieved_context=text(r_row.get("retrieved_context", ""))[:6000],
            rag_response=text(r_row.get("rag_response", ""))[:4000],
        )

        raw = ""
        try:
            raw = call_llm(JUDGE_SYSTEM_PROMPT, user_prompt)
            labels = parse_json_object(raw)
        except Exception as exc:
            # Persist completed annotations and continue so one judge failure
            # does not discard the whole evaluation.
            print(f"[{index + 1}/{total}] {qid}: evaluation failed: {exc}")
            save_outputs(baseline_df, rag_df)
            time.sleep(1)
            continue

        baseline_df.loc[index, "baseline_correctness"] = normalized_label(
            labels.get("baseline_correctness", ""),
            ["Correct", "Partially correct", "Incorrect", "Unanswerable/insufficient evidence"],
        )
        baseline_df.loc[index, "baseline_hallucination"] = normalized_label(
            labels.get("baseline_hallucination", ""), ["Yes", "No", "Unclear"]
        )
        baseline_df.loc[index, "baseline_faithfulness"] = normalized_label(
            labels.get("baseline_faithfulness", ""),
            ["Supported", "Partially supported", "Unsupported", "Not applicable"],
        )
        baseline_df.loc[index, "baseline_annotation_notes"] = text(
            labels.get("baseline_annotation_notes", "")
        )

        rag_df.loc[index, "rag_correctness"] = normalized_label(
            labels.get("rag_correctness", ""),
            ["Correct", "Partially correct", "Incorrect", "Unanswerable/insufficient evidence"],
        )
        rag_df.loc[index, "rag_hallucination"] = normalized_label(
            labels.get("rag_hallucination", ""), ["Yes", "No", "Unclear"]
        )
        rag_df.loc[index, "rag_faithfulness"] = normalized_label(
            labels.get("rag_faithfulness", ""),
            ["Supported", "Partially supported", "Unsupported", "Not applicable"],
        )
        rag_df.loc[index, "rag_annotation_notes"] = text(
            labels.get("rag_annotation_notes", "")
        )
        rag_df.loc[index, "retrieval_relevance"] = normalized_label(
            labels.get("retrieval_relevance", ""),
            ["Relevant and sufficient", "Relevant but incomplete", "Irrelevant", "No useful evidence"],
        )

        save_outputs(baseline_df, rag_df)
        print(f"[{index + 1}/{total}] {qid}: annotations saved")
        time.sleep(0.2)

    build_summary(baseline_df, rag_df)
    print("\nEvaluation pass finished.")
    print(f"Baseline annotations: {BASELINE_OUTPUT}")
    print(f"RAG annotations:      {RAG_OUTPUT}")
    print(f"Summary counts:       {SUMMARY_OUTPUT}")
    print(
        "Reminder: these are preliminary LLM-judge labels. Manually review uncertain "
        "cases and a sample of all labels before reporting research findings."
    )


if __name__ == "__main__":
    main()
