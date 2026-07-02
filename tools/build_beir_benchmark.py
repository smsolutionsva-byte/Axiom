import argparse
import csv
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path


def download_beir_dataset(dataset_name: str, cache_dir: Path) -> Path:
    """Downloads and extracts a BEIR dataset."""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{dataset_name}.zip"
    dataset_dir = cache_dir / dataset_name

    if not dataset_dir.exists():
        print(f"Downloading {dataset_name} from {url}...", flush=True)
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"Extracting {zip_path}...", flush=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(cache_dir)
        except Exception as e:
            print(f"Failed to download or extract dataset: {e}", flush=True)
            raise
    else:
        print(f"Dataset {dataset_name} already exists in {dataset_dir}", flush=True)

    return dataset_dir


def safe_doc_filename(doc_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_id).strip("._-")
    if not safe:
        safe = "doc"
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:80]}__{digest}.txt"


def build_axiom_corpus_and_cases(
    dataset_dir: Path,
    dataset_name: str,
    top_n_queries: int = 100,
    *,
    corpus_scope: str = "full",
    clean: bool = True,
):
    """Converts BEIR format to Axiom format."""
    corpus_file = dataset_dir / "corpus.jsonl"
    queries_file = dataset_dir / "queries.jsonl"
    qrels_file = dataset_dir / "qrels" / "test.tsv"

    axiom_corpus_dir = Path("samples") / f"{dataset_name}_corpus"
    if clean and axiom_corpus_dir.exists():
        shutil.rmtree(axiom_corpus_dir)
    axiom_corpus_dir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...", flush=True)
    corpus = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc

    print("Loading queries...", flush=True)
    queries = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    print("Loading qrels (test split)...", flush=True)
    qrels = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            if score > 0:  # Only consider positive relevance
                qrels.setdefault(query_id, []).append(corpus_id)

    # We will only convert queries that have relevance judgments
    valid_queries = [qid for qid in qrels.keys() if qid in queries]
    
    # Cap at top_n_queries to prevent massive benchmarks unless specified
    if top_n_queries > 0:
        valid_queries = valid_queries[:top_n_queries]

    selected_doc_ids = set(corpus)
    if corpus_scope == "relevant":
        selected_doc_ids = {
            doc_id
            for query_id in valid_queries
            for doc_id in qrels.get(query_id, [])
            if doc_id in corpus
        }

    doc_filenames = {doc_id: safe_doc_filename(doc_id) for doc_id in selected_doc_ids}

    print(
        f"Building Axiom corpus ({len(selected_doc_ids)} of {len(corpus)} documents, scope={corpus_scope}) "
        f"and benchmark ({len(valid_queries)} cases)...",
        flush=True,
    )
    
    # 1. Write the Corpus files
    for doc_index, doc_id in enumerate(sorted(selected_doc_ids), start=1):
        doc = corpus[doc_id]
        file_path = axiom_corpus_dir / doc_filenames[doc_id]
        title = doc.get("title", "")
        text = doc.get("text", "")
        content = f"{title}\n\n{text}" if title else text
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        if doc_index % 1000 == 0:
            print(f"Wrote {doc_index}/{len(selected_doc_ids)} corpus documents...", flush=True)

    # 2. Write the Benchmark cases
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    benchmark_file = benchmarks_dir / f"{dataset_name}_eval.jsonl"
    
    cases = []
    for idx, query_id in enumerate(valid_queries):
        expected_doc_ids = qrels[query_id]
        expected_sources = [
            doc_filenames[doc_id]
            for doc_id in expected_doc_ids
            if doc_id in doc_filenames
        ]
        
        # Create a synthetic reference answer from the expected documents for Ragas Context Recall
        ref_texts = []
        for doc_id in expected_doc_ids:
            if doc_id in corpus:
                text = corpus[doc_id].get("text", "")
                # Truncate to avoid context window blowups in Ragas
                ref_texts.append(text[:500])
        
        synthetic_reference = " ".join(ref_texts)
        if not synthetic_reference.strip():
            synthetic_reference = queries[query_id]

        cases.append({
            "id": f"{dataset_name}-{idx:04d}",
            "question": queries[query_id],
            "expected_sources": expected_sources,
            "expected_terms": [], # BEIR does not define specific expected terms
            "reference": synthetic_reference,
            "notes": f"query_id:{query_id}"
        })

    with open(benchmark_file, "w", encoding="utf-8", newline="\n") as f:
        for case in cases:
            f.write(json.dumps(case, sort_keys=True) + "\n")

    print(f"Done! Corpus written to {axiom_corpus_dir}/", flush=True)
    print(f"Benchmark cases written to {benchmark_file}", flush=True)
    print("\nNext steps on Colab:", flush=True)
    print("Fast retrieval smoke test first:", flush=True)
    print(f"!python -m axiom benchmark \\")
    print(f"  --db data/{dataset_name}_eval.sqlite \\")
    print(f"  --corpus {axiom_corpus_dir} \\")
    print(f"  --dataset {benchmark_file} \\")
    print(f"  --modes vector,hiverag \\")
    print(f"  --case-limit 10 \\")
    print(f"  --out exports/benchmarks \\")
    print(f"  --label {dataset_name}_smoke")
    print("\nThen run the slower official evaluator only after the smoke test moves:", flush=True)
    print(f"!python -m axiom benchmark \\")
    print(f"  --db data/{dataset_name}_eval.sqlite \\")
    print(f"  --dataset {benchmark_file} \\")
    print(f"  --modes vector,hiverag \\")
    print(f"  --evaluator ollama \\")
    print(f"  --evaluator-model qwen2.5:7b \\")
    print(f"  --out exports/benchmarks \\")
    print(f"  --label {dataset_name}_showdown")


def main():
    parser = argparse.ArgumentParser(description="Build an Axiom benchmark from a BEIR dataset")
    parser.add_argument("--dataset", type=str, default="scifact", help="Name of the BEIR dataset (e.g. scifact, nfcorpus, fiqa, etc.)")
    parser.add_argument("--limit", type=int, default=100, help="Max number of queries to include in the benchmark (0 for all)")
    parser.add_argument(
        "--corpus-scope",
        choices=["full", "relevant"],
        default="full",
        help="Use full BEIR corpus for real runs, or only judged-relevant docs for a quick smoke test.",
    )
    parser.add_argument("--no-clean", action="store_true", help="Do not clear the generated Axiom corpus folder before writing.")
    args = parser.parse_args()

    cache_dir = Path("tmp") / "beir_cache"
    dataset_dir = download_beir_dataset(args.dataset, cache_dir)
    build_axiom_corpus_and_cases(
        dataset_dir,
        args.dataset,
        top_n_queries=args.limit,
        corpus_scope=args.corpus_scope,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
