# HiveRAG: Biologically Optimized Retrieval-Augmented Generation Through Hexagonal Memory, Spherical Concept Clustering, Web Signal Propagation, and Hebbian Retrieval Plasticity

Shivansh Mukhia

Draft date: 30 June 2026

## Abstract

Retrieval-Augmented Generation (RAG) systems commonly rely on vector search, hybrid lexical-vector search, graph traversal, or agentic retry loops. These methods improve retrieval quality, but they often treat retrieval structures as independent tools rather than as a coordinated adaptive memory system. This paper proposes HiveRAG, a biologically inspired retrieval architecture that maps retrieval roles to optimization principles found in natural systems. HiveRAG uses hexagonal tiling for local embedding-space partitioning, spherical concept clusters for high-level memory, hierarchical tree routing for progressive narrowing, spider-web signal propagation for decentralized cross-domain relevance flow, and Hebbian-style adaptive edge plasticity so frequently useful retrieval paths strengthen while stale paths decay. A local implementation in Axiom demonstrates an end-to-end HiveRAG prototype with strict citation support, layer auditability, and benchmark tooling. Preliminary smoke-test results show that HiveRAG matches strong local retrieval baselines on a small multimodal corpus, while incurring higher latency due to layered routing and signal propagation. Larger evaluations against BEIR, HotpotQA, QASPER, Natural Questions, RAGTruth, and custom long-PDF corpora are proposed to measure whether HiveRAG improves recall and evidence coverage under realistic multi-document workloads.

## 1. Introduction

RAG systems improve language-model answers by retrieving external evidence before generation. A basic vector RAG system retrieves semantically similar chunks. Hybrid RAG adds lexical matching and rank fusion. Tree-based RAG systems organize documents into hierarchical summaries. GraphRAG uses entities, relations, and community summaries to support global questions across a corpus. Agentic RAG systems add planning, critique, and iterative retrieval.

These approaches are useful, but each has a characteristic failure mode. Vector search can miss exact identifiers and numeric details. Hybrid search can remain static top-k retrieval. Tree retrieval can over-trust summaries. Graph retrieval can over-expand through noisy edges. Agentic retrieval can become slow or unstable.

HiveRAG asks a different question: can retrieval be organized as an adaptive memory ecology where each structure has a specific optimization role?

The core contributions are:

1. A layered HiveRAG architecture with explicit responsibilities for sphere, tree, hex, spider-web, adaptive growth, and energy-budget layers.
2. A concrete implementation in Axiom using versioned HiveRAG v2 indexes.
3. A benchmark harness that compares HiveRAG against vector, hybrid, tree, graph-lite, and AVTR-style baselines.
4. A research protocol for scaling evaluation to public benchmarks and custom long-document corpora.

## 2. Related Work

Vector RAG retrieves nearest chunks using dense embeddings. Hybrid RAG combines dense retrieval with lexical retrieval, often using reciprocal rank fusion. RAPTOR introduced tree-organized retrieval with recursive abstractive summaries. GraphRAG introduced graph-based local and global retrieval over entity and community structures. Self-RAG and Corrective RAG introduced retrieval critique, correction, and self-reflection. Contextual Retrieval improves chunk retrieval by prepending document-specific context to chunks.

HiveRAG is related to all of these methods, but it differs in its unifying contract. HiveRAG does not merely combine retrieval methods. It assigns each method to a biologically motivated optimization principle and uses an energy controller to decide how much retrieval is justified.

## 3. HiveRAG Architecture

HiveRAG has six primary layers.

### 3.1 Sphere Layer

The sphere layer stores high-level concept memory. Each sphere is a concept cluster in embedding space. A sphere stores:

- centroid vector
- support chunks
- topic terms
- document IDs
- radius
- shell mean
- shell standard deviation
- density
- dimensionality

The sphere layer is used for coarse routing. It should not be used as final proof. Its purpose is to cheaply identify likely topic regions before expensive retrieval.

The high-dimensional sphere analogy matters because embedding clusters are not flat bags of text. Distance from centroid, shell concentration, and density can influence whether a query belongs to the core of a concept or near its boundary.

### 3.2 Tree Layer

The tree layer organizes knowledge hierarchically:

- corpus node
- document node
- section or parent chunk node
- child evidence chunks

Tree routing narrows search progressively. The tree layer is useful when documents have natural structure or when a question needs broad-to-specific descent. Like the sphere layer, tree summaries route retrieval but do not prove final claims.

### 3.3 Hex Layer

The hex layer is based on optimal packing. Hexagonal tiling partitions a plane with uniform neighbor distance and fewer directional artifacts than square grids. HiveRAG projects high-dimensional vectors into a stable 2D plane, assigns chunks to axial hex-grid cells, and retrieves from same-cell and adjacent-cell neighborhoods.

In Axiom HiveRAG v2:

- each chunk vector is projected to `(x, y)`
- each point is rounded to axial hex coordinates `(q, r)`
- chunks are assigned to `biorag_hex_cells`
- each chunk receives a `biorag_chunk_cells` assignment
- retrieval explores same-cell and adjacent-cell chunks before broader fallback

This layer is not a replacement for ANN search. It is a local exploration structure designed to reduce edge artifacts and support efficient neighborhood retrieval.

### 3.4 Spider/Web Layer

The spider/web layer models distributed cross-domain connectivity. A strict tree has a single path from root to leaf; a web allows multiple routes. When a query activates one node, relevance should vibrate outward through connected nodes rather than staying isolated.

HiveRAG v2 implements this using damped signal propagation:

```text
signal(target) += signal(source) * damping * normalized_edge_weight
```

Normalization reduces domination by high-degree hubs. Damping prevents uncontrolled graph spread.

### 3.5 Adaptive Growth Layer

The adaptive growth layer applies Hebbian-style plasticity to retrieval edges:

```text
if evidence_a and evidence_b are selected together:
    weight(a, b) += learning_rate
else:
    weight decays over time
```

This is implemented through `biorag_edge_weights`. Edges store weight, pheromone, coactivation count, and decay timestamps.

The research hypothesis is that retrieval paths repeatedly useful for verified answers should become easier to traverse, while unused paths should weaken. This is potentially novel when applied directly to graph-based RAG retrieval edges rather than only to neural network weights or final ranker parameters.

### 3.6 Energy Budget Layer

HiveRAG should not use every layer for every query. A query complexity estimate controls:

- seed candidate count
- sphere limit
- tree limit
- hex frontier
- web propagation iterations
- maximum candidates
- subquery budget

This prevents simple questions from paying the full cost of deep graph traversal.

## 4. Implementation

HiveRAG is implemented in Axiom as a local-first retrieval engine.

### 4.1 Storage

The implementation adds these persistent structures:

| Table | Purpose |
| --- | --- |
| `sphere_summaries` | Concept clusters with centroid, radius, density, and support chunks |
| `biorag_tree_nodes` | Corpus, document, and section routing hierarchy |
| `biorag_hex_cells` | Axial hex-grid cells in projected embedding space |
| `biorag_chunk_cells` | Chunk-to-hex-cell assignments |
| `hex_neighbors` | Bounded local semantic and hex-cell neighbors |
| `biorag_edge_weights` | Hebbian web edge weights and decay state |
| `adaptive_path_stats` | Query-signature path statistics |
| `biorag_retrieval_runs` | Budget and trace audit for each retrieval |

### 4.2 Retrieval

The HiveRAG retrieval pipeline is:

1. Estimate query energy budget.
2. Retrieve dense vector seeds.
3. Retrieve lexical seeds.
4. Route through concept spheres.
5. Route through tree nodes.
6. Expand through hex-grid local neighborhoods.
7. Propagate signal through spider/web edges.
8. Apply adaptive growth boosts.
9. Rerank and select a coverage-aware evidence set.
10. Save retrieval trace.
11. Reinforce selected evidence pairs with Hebbian edge updates.

### 4.3 Citation Contract

Internal routing nodes do not prove final claims. Final answers must cite leaf-level chunks. This follows the evidence discipline used in Axiom and AVTR-HRAG: internal structures route; leaves prove.

## 5. Baselines

The current benchmark harness compares:

| Mode | Description |
| --- | --- |
| Vector | Dense vector retrieval only |
| Hybrid | Dense plus lexical retrieval with fusion |
| Tree | Sphere and tree routing without full HiveRAG |
| Graph | GraphRAG-lite signal propagation over links |
| AVTR | Agentic Verifiable Tree-Reasoning Hybrid RAG style retrieval |
| HiveRAG | Full HiveRAG v2 pipeline |

Future comparisons should include external implementations where possible:

- Microsoft GraphRAG for graph/community retrieval
- RAPTOR for tree retrieval
- BM25 plus dense hybrid retrieval
- Self-RAG or Corrective RAG for agentic/critic retrieval
- ColBERT or reranker-augmented hybrid retrieval

## 6. Evaluation Protocol

### 6.1 Retrieval Metrics

The benchmark harness computes:

- Hit@k
- MRR
- source recall
- expected-term recall
- average latency
- returned evidence count

Publication-grade experiments should add:

- Recall@k
- nDCG@k
- evidence precision
- subclaim coverage
- contradiction recall
- citation reconstruction accuracy
- cost per verified answer
- p50/p95 latency

### 6.2 Generation Metrics

For answer generation, evaluate:

- factuality
- citation support rate
- unsupported claim rate
- abstention quality
- answer completeness
- hallucination rate

RAGTruth or a similar hallucination benchmark should be used for generation faithfulness.

## 7. Datasets

The full paper should use multiple dataset families:

| Dataset | Purpose |
| --- | --- |
| Local Axiom smoke set | Reproducible implementation sanity check |
| BEIR | Broad retrieval evaluation |
| MTEB retrieval tasks | Embedding retrieval comparison |
| HotpotQA | Multi-hop QA |
| QASPER | Question answering over research papers |
| Natural Questions | Open-domain QA |
| RAGTruth | Hallucination and faithfulness |
| Custom 50-PDF corpus | Long-document, citation, table, OCR, and multi-document stress test |

The custom corpus is important because public datasets usually do not test the exact target: many large PDFs, strict leaf citations, OCR/table issues, and cross-document evidence trails.

## 8. Preliminary Local Smoke Benchmark

The first reproducible smoke benchmark uses six local Axiom cases from the `samples` folder. This benchmark is intentionally small and should not be interpreted as a final research result.

Command:

```powershell
python -m axiom --db data\hiverag_benchmark.sqlite benchmark --corpus samples --dataset benchmarks\local_axiom_eval.jsonl --modes vector,hybrid,tree,graph,avtr,hiverag --top-k 5 --out exports\benchmarks --label local_smoke_hiverag
```

Results:

| Mode | Hit@5 | MRR | Source Recall | Term Recall | Context Precision Proxy | Faithfulness Proxy | Avg Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector | 1.0000 | 1.0000 | 0.8333 | 0.6667 | 0.8333 | 0.6667 | 3.260 |
| Hybrid | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6667 | 5.497 |
| Tree | 1.0000 | 1.0000 | 0.8333 | 0.7222 | 0.8333 | 0.7222 | 11.711 |
| Graph-lite | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6667 | 16.952 |
| AVTR | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6667 | 15.164 |
| HiveRAG v2 | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.8667 | 0.6667 | 29.055 |

Interpretation:

- All methods are strong on this tiny corpus.
- HiveRAG does not show a quality advantage on this smoke set because the queries are easy.
- HiveRAG is slower because it performs layered routing, hex exploration, web propagation, and adaptive updates.
- The real test is whether HiveRAG improves recall and evidence coverage on larger multi-document and multi-hop corpora.

## 8.1 Evaluator Framework Benchmark Tracks

The updated benchmark harness now records RAGAS-style, TruLens-style, and DeepEval-style evaluation tracks. In this local run, the official evaluator packages were not installed, so the paper reports deterministic offline proxy metrics mapped to each framework's RAG metric family. These values are useful for repeatable smoke testing, but they are not a substitute for final LLM-judge evaluation.

Official evaluator availability in this run:

| Framework | Official package available | Local track | Metric family represented |
| --- | --- | --- | --- |
| RAGAS | No | offline proxy | context precision, context recall, faithfulness, answer relevancy |
| TruLens | No | offline proxy | context relevance, groundedness, answer relevance |
| DeepEval | No | offline proxy | contextual precision, contextual recall, contextual relevancy, faithfulness, answer relevancy |

RAGAS-style proxy scores:

| Mode | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | 0.8333 | 0.6667 | 0.6667 | 0.5880 | 0.6887 |
| Hybrid | 0.8333 | 0.6667 | 0.6667 | 0.6065 | 0.6933 |
| Tree | 0.8333 | 0.7222 | 0.7222 | 0.6065 | 0.7210 |
| Graph-lite | 0.8333 | 0.6667 | 0.6667 | 0.6065 | 0.6933 |
| AVTR | 0.8333 | 0.6667 | 0.6667 | 0.6065 | 0.6933 |
| HiveRAG v2 | 0.8667 | 0.6667 | 0.6667 | 0.6065 | 0.7016 |

TruLens-style RAG triad proxy scores:

| Mode | Context Relevance | Groundedness | Answer Relevance | RAG Triad |
| --- | ---: | ---: | ---: | ---: |
| Vector | 0.7107 | 0.6667 | 0.5880 | 0.6551 |
| Hybrid | 0.7199 | 0.6667 | 0.6065 | 0.6644 |
| Tree | 0.7199 | 0.7222 | 0.6065 | 0.6829 |
| Graph-lite | 0.7199 | 0.6667 | 0.6065 | 0.6644 |
| AVTR | 0.7199 | 0.6667 | 0.6065 | 0.6644 |
| HiveRAG v2 | 0.7366 | 0.6667 | 0.6065 | 0.6699 |

DeepEval-style proxy scores:

| Mode | Contextual Precision | Contextual Recall | Contextual Relevancy | Faithfulness | Answer Relevancy | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector | 0.8333 | 0.6667 | 0.6960 | 0.6667 | 0.5880 | 0.6901 |
| Hybrid | 0.8333 | 0.6667 | 0.7022 | 0.6667 | 0.6065 | 0.6951 |
| Tree | 0.8333 | 0.7222 | 0.7207 | 0.7222 | 0.6065 | 0.7210 |
| Graph-lite | 0.8333 | 0.6667 | 0.7022 | 0.6667 | 0.6065 | 0.6951 |
| AVTR | 0.8333 | 0.6667 | 0.7022 | 0.6667 | 0.6065 | 0.6951 |
| HiveRAG v2 | 0.8667 | 0.6667 | 0.7133 | 0.6667 | 0.6065 | 0.7040 |

The proxy definitions are:

- context precision proxy: fraction of returned contexts that match expected source names or expected evidence terms
- context recall proxy: expected-term recall from retrieved contexts
- faithfulness or groundedness proxy: fraction of expected answer/evidence terms supported by retrieved contexts
- answer relevancy proxy: fraction of meaningful query terms represented in retrieved contexts
- context relevancy proxy: blend of context precision and answer relevancy

Official RAGAS, TruLens, and DeepEval runs can be added through the optional `eval` dependency group:

```powershell
python -m pip install -e ".[eval]"
```

For a final paper, these official tools should be run with a configured evaluator LLM and embeddings, then reported separately from retrieval-only metrics.

## 9. Expected Research Questions

The final paper should answer:

1. Does HiveRAG improve evidence recall on long multi-document corpora?
2. Does hex-grid local search reduce neighborhood artifacts compared with kNN alone?
3. Does sphere geometry improve coarse routing over plain topic summaries?
4. Does web signal propagation retrieve useful bridge evidence missed by top-k search?
5. Does Hebbian edge reinforcement improve future retrieval without causing popularity bias?
6. Does the energy budget reduce latency without harming recall?
7. Does HiveRAG outperform GraphRAG or TreeRAG on any task class?
8. Which HiveRAG layer contributes unique evidence?

## 10. Ablation Plan

Run these variants:

| Variant | Removed component |
| --- | --- |
| HiveRAG-full | none |
| no-sphere | sphere routing disabled |
| no-tree | tree routing disabled |
| no-hex | hex-cell expansion disabled |
| no-web | signal propagation disabled |
| no-hebbian | edge reinforcement disabled |
| no-energy | fixed budget for every query |

The ablation table is essential. Without it, HiveRAG cannot prove that each biological layer contributes measurable value.

## 11. Threats to Validity

1. Small local benchmark results do not prove generalization.
2. Synthetic or easy queries can hide differences between retrieval systems.
3. Expected-term recall can undercount valid paraphrases.
4. Source recall can overcount when a document is returned but the exact evidence span is weak.
5. Local hashed embeddings are weaker than production embedding models.
6. HiveRAG has more moving parts and can lose on latency.
7. Adaptive edge weights may reinforce historical bias.
8. Graph propagation can over-rank hubs if normalization is insufficient.

## 12. Production Evaluation Plan

1. Build dataset adapters for BEIR, HotpotQA, QASPER, Natural Questions, and RAGTruth.
2. Build a custom long-PDF dataset with gold evidence spans.
3. Run each baseline with the same chunking and embedding model.
4. Report retrieval and generation metrics separately.
5. Run ablations.
6. Measure latency and memory.
7. Publish exact commands, configs, and result JSON files.

## 13. Conclusion

HiveRAG proposes a retrieval architecture that treats memory as a layered adaptive system. Hexagonal tiling handles local embedding-space neighborhoods. Spheres handle coarse high-dimensional concept memory. Trees handle hierarchical narrowing. Spider-web signal propagation handles distributed cross-domain evidence flow. Hebbian plasticity adapts retrieval paths over time. The current Axiom implementation demonstrates that the architecture can be built end-to-end with citations and trace auditability. The next step is large-scale evaluation against public and custom benchmarks to determine whether the added complexity produces measurable gains over simpler RAG systems.

## References

- GraphRAG: From Local to Global Graph Retrieval-Augmented Generation. https://arxiv.org/abs/2404.16130
- RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. https://arxiv.org/abs/2401.18059
- BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. https://arxiv.org/abs/2104.08663
- MTEB: Massive Text Embedding Benchmark. https://arxiv.org/abs/2210.07316
- HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. https://arxiv.org/abs/1809.09600
- QASPER: A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. https://arxiv.org/abs/2105.03011
- Natural Questions. https://ai.google.com/research/NaturalQuestions
- RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. https://arxiv.org/abs/2401.00396
- RAGAS: Evaluation framework for Retrieval-Augmented Generation. https://docs.ragas.io/
- TruLens: RAG evaluation and observability with the RAG triad. https://www.trulens.org/
- DeepEval: Evaluation framework for LLM applications and RAG metrics. https://docs.confident-ai.com/
- Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. https://arxiv.org/abs/2310.11511
- Corrective Retrieval Augmented Generation. https://arxiv.org/abs/2401.15884
- Anthropic Contextual Retrieval. https://www.anthropic.com/engineering/contextual-retrieval

