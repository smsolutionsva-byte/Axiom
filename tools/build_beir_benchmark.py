import argparse
import csv
import json
import os
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
        print(f"Downloading {dataset_name} from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(cache_dir)
        except Exception as e:
            print(f"Failed to download or extract dataset: {e}")
            raise
    else:
        print(f"Dataset {dataset_name} already exists in {dataset_dir}")

    return dataset_dir


def build_axiom_corpus_and_cases(dataset_dir: Path, dataset_name: str, top_n_queries: int = 100):
    """Converts BEIR format to Axiom format."""
    corpus_file = dataset_dir / "corpus.jsonl"
    queries_file = dataset_dir / "queries.jsonl"
    qrels_file = dataset_dir / "qrels" / "test.tsv"

    axiom_corpus_dir = Path("samples") / f"{dataset_name}_corpus"
    axiom_corpus_dir.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    corpus = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc

    print("Loading queries...")
    queries = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    print("Loading qrels (test split)...")
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

    print(f"Building Axiom corpus ({len(corpus)} documents) and benchmark ({len(valid_queries)} cases)...")
    
    # 1. Write the Corpus files
    for doc_id, doc in corpus.items():
        file_path = axiom_corpus_dir / f"{doc_id}.txt"
        title = doc.get("title", "")
        text = doc.get("text", "")
        content = f"{title}\n\n{text}" if title else text
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    # 2. Write the Benchmark cases
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    benchmark_file = benchmarks_dir / f"{dataset_name}_eval.jsonl"
    
    cases = []
    for idx, query_id in enumerate(valid_queries):
        expected_doc_ids = qrels[query_id]
        expected_sources = [f"{doc_id}.txt" for doc_id in expected_doc_ids]
        
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

    print(f"Done! Corpus written to {axiom_corpus_dir}/")
    print(f"Benchmark cases written to {benchmark_file}")
    print("\nNext steps on Colab:")
    print(f"!python -m axiom benchmark \\")
    print(f"  --db data/{dataset_name}_eval.sqlite \\")
    print(f"  --corpus {axiom_corpus_dir} \\")
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
    args = parser.parse_args()

    cache_dir = Path("tmp") / "beir_cache"
    dataset_dir = download_beir_dataset(args.dataset, cache_dir)
    build_axiom_corpus_and_cases(dataset_dir, args.dataset, top_n_queries=args.limit)


if __name__ == "__main__":
    main()
