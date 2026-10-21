"""Baseline experiment: the LLM receives the question, not supplied knowledge.

This version:
- retries empty model outputs instead of treating them as completed answers;
- skips only rows with a non-empty baseline_response;
- preserves completed rows while resuming;
- saves after every successful answer;
- reports file-lock/permission errors clearly.
"""
import time

from common import (
    MODEL,
    TEMPERATURE,
    MAX_RETRIES,
    ensure_setup,
    prepare_records,
    call_llm,
    output_csv_path,
    load_existing_rows,
    save_rows,
    annotation_columns,
)

SYSTEM_PROMPT = (
    "Answer accurately and concisely using your existing knowledge. "
    "Do not claim to have consulted external documents. "
    "If unsure, say so rather than inventing details."
)

# Empty outputs are retried independently of the HTTP retry logic in common.py.
MAX_EMPTY_OUTPUT_ATTEMPTS = 3
PAUSE_BETWEEN_QUESTIONS_SECONDS = 0.2


def response_is_complete(row):
    """Only a non-empty response counts as completed."""
    return bool(str(row.get("baseline_response", "") or "").strip())


def save_safely(path, rows):
    """Save rows and give a useful message if another app locks the CSV."""
    try:
        save_rows(path, rows)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to {path}. Close the CSV in Excel or another app, "
            "then rerun this script. Existing saved results have not been "
            "intentionally deleted."
        ) from exc


def main():
    ensure_setup()
    records = prepare_records()
    path = output_csv_path("baseline_responses.csv")
    rows = load_existing_rows(path)

    print(
        f"Model={MODEL}; temperature={TEMPERATURE}; "
        f"questions={len(records)}; empty-output retries={MAX_EMPTY_OUTPUT_ATTEMPTS}"
    )

    completed = 0
    failed = 0

    for index, record in enumerate(records, start=1):
        question_id = str(record["question_id"])
        existing_row = rows.get(question_id, {})

        if response_is_complete(existing_row):
            print(f"[{index}/{len(records)}] {question_id}: already complete; skipping")
            completed += 1
            continue

        print(f"[{index}/{len(records)}] {question_id}: generating baseline response")
        answer = ""

        for attempt in range(1, MAX_EMPTY_OUTPUT_ATTEMPTS + 1):
            try:
                candidate = call_llm(SYSTEM_PROMPT, record["question"])
            except Exception as exc:
                # Save current accumulated rows before propagating the error,
                # so a transient API failure does not discard completed work.
                save_safely(path, rows)
                print(
                    f"{question_id}: API call failed on attempt "
                    f"{attempt}/{MAX_EMPTY_OUTPUT_ATTEMPTS}: {exc}"
                )
                raise

            candidate = str(candidate or "").strip()
            if candidate:
                answer = candidate
                break

            print(
                f"{question_id}: model returned an empty answer "
                f"({attempt}/{MAX_EMPTY_OUTPUT_ATTEMPTS})."
            )
            if attempt < MAX_EMPTY_OUTPUT_ATTEMPTS:
                time.sleep(min(2 ** attempt, 8))

        if not answer:
            # Store a visible failure status, but keep baseline_response empty.
            # The next run will retry it because response_is_complete() is false.
            rows[question_id] = {
                "question_id": question_id,
                "question": record["question"],
                "reference_answer": record["reference_answer"],
                "knowledge": record["knowledge"],
                "dataset_hallucinated_answer": record["dataset_hallucinated_answer"],
                "model": MODEL,
                "temperature": TEMPERATURE,
                "baseline_response": "",
                "generation_status": "failed_empty_output",
                **annotation_columns("baseline"),
            }
            save_safely(path, rows)
            failed += 1
            print(
                f"{question_id}: no non-empty response after "
                f"{MAX_EMPTY_OUTPUT_ATTEMPTS} attempts; left for retry on next run."
            )
            continue

        # Keep the existing dataset fields consistent, and mark this generation successful.
        rows[question_id] = {
            "question_id": question_id,
            "question": record["question"],
            "reference_answer": record["reference_answer"],
            "knowledge": record["knowledge"],
            "dataset_hallucinated_answer": record["dataset_hallucinated_answer"],
            "model": MODEL,
            "temperature": TEMPERATURE,
            "baseline_response": answer,
            "generation_status": "success",
            **annotation_columns("baseline"),
        }

        # Save each completed answer immediately to support safe resuming.
        save_safely(path, rows)
        completed += 1
        time.sleep(PAUSE_BETWEEN_QUESTIONS_SECONDS)

    print(f"Saved: {path}")
    print(f"Completed responses (including previously saved): {completed}")
    print(f"Empty-output failures still needing retry: {failed}")
    if failed:
        print(
            "Run this script again after checking API availability; rows with "
            "empty baseline_response will be attempted again."
        )


if __name__ == "__main__":
    main()
