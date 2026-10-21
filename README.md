# Reducing Large Language Model Hallucination Using Retrieval-Augmented Generation

A comparative research project evaluating a question-only baseline large language model (LLM) against a retrieval-augmented generation (RAG) pipeline using the Groq API.

## Project overview

Large language models can produce fluent answers containing unsupported or incorrect claims. This project compares two configurations:

- **Baseline LLM:** receives the question only and answers using its pretrained knowledge.
- **RAG system:** retrieves relevant knowledge snippets from the dataset using TF-IDF and cosine similarity, then receives the question together with the retrieved context.

Both systems use the configured Groq-hosted model. The experiment records the generated answers so that they can be evaluated against reference answers and available evidence.

> **Research caveat:** The baseline and RAG conditions differ in both evidence access and prompting instructions. Therefore, the experiment evaluates the overall configured RAG approach; it does not isolate the effect of retrieval alone.

## Repository structure

```text
research/
├── data/
│   └── qa_data.json
├── results/
│   ├── baseline_responses.csv
│   ├── rag_responses.csv
│   └── comparison.csv
├── common.py
├── baseline_experiment.py
├── rag_experiment.py
├── evaluate_results.py
├── compare_results.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The exact files in your local project may differ. Keep this structure description synchronized with the files you actually commit.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer recommended
- A Groq API key and access to the selected model
- The project dataset at `data/qa_data.json`

Python package requirements are listed in `requirements.txt`.

## Setup on Windows PowerShell

Run these commands from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run the following in that PowerShell window, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Configure the API

1. Copy `.env.example` to `.env`.
2. Open `.env` and enter your own Groq API key.
3. Check that the model and experiment settings match the configuration used for your run.

Example configuration (adjust values to match your actual experiment):

```dotenv
GROQ_API_KEY=put_your_actual_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-120b
SAMPLE_SIZE=100
TOP_K=2
TEMPERATURE=0
MAX_RETRIES=6
REQUEST_TIMEOUT=90
```

These values are examples, not a claim about the settings used in every experiment. Record the actual model, sample size, and other settings used to generate the results in the final report.

**Never commit `.env` or disclose your API key.** Commit `.env.example` with a placeholder only.

## Dataset

Place the dataset at:

```text
data/qa_data.json
```

The experiment code expects records containing fields such as:

- `knowledge` — evidence text available to the retrieval pipeline
- `question` — question given to the model
- `right_answer` — reference answer for evaluation
- `hallucinated_answer` — dataset-provided hallucinated answer, retained for reference where used by the code

The baseline prompt receives the question, not the record's `knowledge`, `right_answer`, or `hallucinated_answer`. The RAG pipeline retrieves context from the knowledge corpus and sends the retrieved context with the question. Reference answers and dataset-provided hallucinated answers should not be passed to the generation model if the intended experiment is to prevent answer leakage.

Dataset source: [HaluEval official repository](https://github.com/RUCAIBox/HaluEval).

Before redistributing the dataset, check the original source's licence and usage terms. If you do not have permission to redistribute the data, do not commit the full dataset; document where it can be obtained instead.

## Run the experiment

Run commands from the project root with the virtual environment activated.

### 1. Generate baseline answers

```powershell
python baseline_experiment.py
```

The baseline receives only the question. Results are saved to `results/baseline_responses.csv`.

### 2. Generate RAG answers

```powershell
python rag_experiment.py
```

The RAG system retrieves up to `TOP_K` knowledge snippets and sends them with the question. Results are saved to `results/rag_responses.csv`.

### 3. Evaluate generated answers

```powershell
python evaluate_results.py
```

Use this command if `evaluate_results.py` is present in your project and its implementation matches this workflow. Inspect the script's output and confirm which columns and evaluation method it uses before reporting its metrics.

### 4. Compare results

```powershell
python compare_results.py
```

This script produces or updates `results/comparison.csv` according to its implementation. Check its code to confirm whether it expects evaluated CSVs, manually annotated labels, or both.

**Important:** The order above assumes `evaluate_results.py` produces the evaluated data required by `compare_results.py`. If your scripts use a different input/output flow, follow the actual dependencies in those files.

## Output files

| File | Purpose |
|---|---|
| `results/baseline_responses.csv` | Baseline model answers and associated record fields |
| `results/rag_responses.csv` | RAG answers, retrieved-context details if recorded, and associated record fields |
| `results/comparison.csv` | Comparative metrics or summaries generated by the comparison script |

Some response CSVs may contain annotation columns that are initially blank. A blank annotation is **not** a negative result and must not be counted as a correct answer or as a non-hallucination. Check whether the evaluation script fills these columns or creates a separate evaluated output file.

Before using the results in a report, verify:
- the intended number of records was processed;
- no response is missing or blank;
- the same question IDs are compared across conditions;
- evaluation labels and metric denominators are correct;
- any manually reviewed or model-judged labels are described accurately;
- the saved results correspond to the stated model and configuration.

## Method overview

1. Load and prepare the selected dataset records.
2. Generate baseline answers using questions only.
3. Build a TF-IDF knowledge corpus and retrieve the top-k snippets for RAG questions.
4. Generate RAG answers using the question and retrieved context.
5. Evaluate answers against the reference answers and relevant evidence.
6. Compare the two conditions using clearly defined metrics.

The selection procedure, retrieval corpus, prompts, model identifier, sample size, evaluation protocol, and any exclusions should be documented in the research report. If the first *N* records are selected, describe the sample as a deterministic subset rather than a random sample.

## Interpreting the results

Use the metric definitions implemented in your evaluation code. Common measures include:

- **Correctness:** whether an answer accurately addresses the question in relation to the reference answer.
- **Hallucination rate:** the proportion of evaluated answers labelled as containing unsupported or fabricated claims, using the stated annotation rules.
- **Faithfulness:** the extent to which an answer is supported by the evidence supplied to the system. Define how this is measured for the baseline, which receives no retrieved context.
- **Retrieval relevance:** whether retrieved passages are relevant to the question; this is a retrieval-stage measure, not the same as answer correctness.

Report counts as well as percentages, specify the denominator for each metric, and explain how uncertain, partial, or missing labels were handled. Do not claim that RAG eliminates hallucinations based only on a small sample.

## Reproducibility and limitations

- A fixed temperature can reduce generation variability, but it does not guarantee identical outputs across all service/model versions.
- API rate limits, temporary network failures, and service availability can interrupt runs. Preserve completed outputs and resume only if the scripts correctly skip completed records.
- TF-IDF is a lexical retrieval method; semantically related passages may not share enough words to rank highly.
- If the retrieval corpus is built from the selected questions' own knowledge fields, describe that corpus construction clearly.
- Different prompts for baseline and RAG mean the comparison does not isolate retrieval as the only changing factor.
- A small or non-random subset limits how broadly the results can be generalized.
- If an LLM is used as an evaluator, describe its prompt, model, label criteria, and any human validation. Do not describe model-generated labels as human annotations.

## GitHub safety checklist

Before pushing the repository:

- [ ] `.env` is excluded from Git.
- [ ] `.venv/`, `__pycache__/`, and Python cache files are excluded.
- [ ] No API key, password, or other secret appears in committed files or logs.
- [ ] Dataset redistribution permissions have been checked.
- [ ] Result files contain no secrets or sensitive information.
- [ ] README settings match the actual run and report.
- [ ] All referenced scripts exist and their input/output paths are correct.
- [ ] The final report clearly distinguishes measured results from assumptions and limitations.

## Citation

If you use HaluEval, cite the original dataset paper and repository in your report. Add the full bibliographic details required by your university's citation style.

- Dataset repository: https://github.com/RUCAIBox/HaluEval

## Project links

- **GitHub repository:** add the repository URL after publishing.
- **Research report:** add a link if you choose to publish the report.
- **Dataset:** link to the official source above, or to an approved hosted copy if redistribution is permitted.
