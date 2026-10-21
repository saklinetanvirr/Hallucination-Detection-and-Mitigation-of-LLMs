"""RAG: retrieve top-k knowledge snippets, then answer with question + context."""
import time
from common import MODEL, TEMPERATURE, TOP_K, ensure_setup, prepare_records, call_llm, output_csv_path, load_existing_rows, save_rows, annotation_columns, build_retriever, retrieve

SYSTEM_PROMPT = ("Answer using the supplied context as evidence. Do not invent facts unsupported by the context. If the context lacks enough information, say: 'The provided documents do not state this.' Be concise.")

def main():
    ensure_setup()
    records = prepare_records()
    snippets, vectorizer, matrix = build_retriever(records)
    path = output_csv_path("rag_responses.csv")
    rows = load_existing_rows(path)
    print(f"Model={MODEL}; temperature={TEMPERATURE}; top_k={TOP_K}; questions={len(records)}; corpus_docs={len(snippets)}")
    for i, rec in enumerate(records, 1):
        qid = rec["question_id"]
        if str(rows.get(qid, {}).get("rag_response", "")).strip():
            print(f"[{i}/{len(records)}] {qid}: already complete; skipping")
            continue
        retrieved = retrieve(rec["question"], snippets, vectorizer, matrix, TOP_K)
        context = "\n\n".join(f"[{d['document_id']}] {d['text']}" for d, _ in retrieved)
        doc_ids = ";".join(d["document_id"] for d, _ in retrieved)
        scores = ";".join(f"{s:.4f}" for _, s in retrieved)
        prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{rec['question']}"
        print(f"[{i}/{len(records)}] Generating RAG response")
        answer = call_llm(SYSTEM_PROMPT, prompt)
        rows[qid] = {"question_id": qid, "question": rec["question"],
            "reference_answer": rec["reference_answer"], "knowledge": rec["knowledge"],
            "dataset_hallucinated_answer": rec["dataset_hallucinated_answer"],
            "retrieved_document_ids": doc_ids, "retrieval_similarity_scores": scores,
            "retrieved_context": context, "model": MODEL, "temperature": TEMPERATURE,
            "top_k": TOP_K, "rag_response": answer, **annotation_columns("rag")}
        save_rows(path, rows)
        time.sleep(0.15)
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
