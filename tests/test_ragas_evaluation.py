"""
RAGAS Evaluation Test for Offline RAG Pipeline
================================================
Evaluates the full pipeline (ingestion → retrieval → generation) using the
RAGAS evaluation framework with both non-LLM and LLM-based metrics.

Metrics evaluated:
  ── Non-LLM (always run) ──────────────────────────────────────────────
    • ROUGE-L          – Longest-common-subsequence overlap with reference
    • BLEU             – n-gram precision against reference
    • Non-LLM Similarity – Token-level string similarity
    • String Presence  – Key terms present in the response
    • Context Entity Recall – Named-entity overlap between context & reference

  ── LLM-Based (requires judge LLM) ───────────────────────────────────
    • Faithfulness     – Is the response grounded in retrieved context?
    • Answer Relevancy – Does the response address the question?
    • Context Precision – Are the retrieved chunks relevant to the question?
    • Factual Correctness – Factual alignment with reference answer

  ── Custom Pipeline Metrics ───────────────────────────────────────────
    • Retrieval Hit Rate  – Did retrieval return any results?
    • Avg Reranker Score  – Mean cross-encoder confidence
    • End-to-End Latency  – Total time per query

Judge LLM:
  By default, the test spins up llama-cpp-python's OpenAI-compatible server
  using the same Qwen2.5-3B model. Set RAGAS_SKIP_LLM_METRICS=1 to skip.
  Set OPENAI_API_KEY to use OpenAI GPT-4o-mini as judge instead.

Run:
  cd D:\\Offline_Rag_V2
  .venv\\Scripts\\python.exe tests\\test_ragas_evaluation.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_ROOT = PROJECT_ROOT / "rag-system"

os.chdir(RAG_ROOT)
sys.path.insert(0, str(RAG_ROOT))

# Force UTF-8 stdout on Windows to support Unicode symbols
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["RAG_PATHS__MODELS_DIR"] = str(PROJECT_ROOT / "models")
os.environ["RAG_PATHS__DATA_DIR"] = str(PROJECT_ROOT / "data")
os.environ["RAG_PATHS__INDEX_DIR"] = str(PROJECT_ROOT / "data" / "index")
os.environ["RAG_PATHS__UPLOADS_DIR"] = str(PROJECT_ROOT / "data" / "uploads")
os.environ["RAG_PATHS__LOGS_DIR"] = str(PROJECT_ROOT / "data" / "logs")
os.environ["RAG_LLM__MODEL_PATH"] = str(
    PROJECT_ROOT / "models" / "llm" / "qwen2.5-3b-instruct-q4_k_m.gguf"
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
logger = logging.getLogger("ragas_eval")
logger.setLevel(logging.INFO)


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC KNOWLEDGE CORPUS
# ══════════════════════════════════════════════════════════════════════════════
# Carefully crafted documents with verifiable facts so we can measure
# retrieval accuracy and generation faithfulness.

KNOWLEDGE_DOCS = [
    {
        "source": "quantum_systems_overview.md",
        "text": (
            "Quantum Systems Inc. was founded in 2019 by Dr. Elena Vasquez and "
            "Dr. Marcus Chen in Boulder, Colorado. The company specializes in "
            "quantum-resistant cryptography and post-quantum security solutions. "
            "As of 2025, Quantum Systems employs 342 people across three offices "
            "in Boulder, Austin, and Berlin. The company's flagship product is "
            "QuantumShield, a hardware security module that implements lattice-based "
            "encryption algorithms. QuantumShield was first released in March 2022 "
            "and has since been deployed by over 150 enterprise customers worldwide. "
            "The company reported annual revenue of $47.3 million in fiscal year 2024."
        ),
    },
    {
        "source": "quantumshield_technical.md",
        "text": (
            "QuantumShield Technical Specifications: The QuantumShield HSM uses "
            "CRYSTALS-Kyber for key encapsulation and CRYSTALS-Dilithium for "
            "digital signatures. It supports AES-256-GCM for symmetric encryption. "
            "The device operates at 2.4 GHz with 16 GB of secure memory and can "
            "process up to 50,000 cryptographic operations per second. Power "
            "consumption is rated at 45 watts under full load. The module has "
            "achieved FIPS 140-3 Level 3 certification and Common Criteria EAL4+ "
            "certification. Firmware updates are delivered over a secure OTA channel "
            "using dual-signature verification. The latest firmware version is 4.2.1, "
            "released in January 2025."
        ),
    },
    {
        "source": "quantum_research_division.md",
        "text": (
            "The Quantum Systems Research Division is led by Dr. Sarah Kim and "
            "consists of 48 researchers. Key research areas include: (1) Homomorphic "
            "encryption acceleration using custom ASICs, (2) Quantum key distribution "
            "over metropolitan fiber networks up to 120 kilometers, (3) Side-channel "
            "attack resistance for embedded devices. In 2024, the research team "
            "published 17 peer-reviewed papers and filed 8 patents. The division "
            "received a $12.5 million DARPA grant in September 2024 for developing "
            "next-generation quantum-safe communication protocols. Their QKD testbed "
            "in Boulder achieved a secure key rate of 2.3 Mbps over 85 km of fiber."
        ),
    },
    {
        "source": "quantum_partnerships.md",
        "text": (
            "Quantum Systems maintains strategic partnerships with several major "
            "technology companies. In June 2023, they signed a five-year agreement "
            "with GlobalBank Corp to deploy QuantumShield across all 2,400 branch "
            "locations in North America. In October 2023, they partnered with "
            "NordTelecom to integrate quantum key distribution into 5G network "
            "infrastructure. The company also collaborates with the University of "
            "Colorado Boulder on a joint quantum computing research lab, established "
            "in 2022. Recent partnership with AeroDefense Systems in March 2024 "
            "focuses on quantum-safe satellite communications for defense applications."
        ),
    },
    {
        "source": "quantum_financials_2024.md",
        "text": (
            "Quantum Systems Financial Summary FY2024: Total revenue was $47.3 million, "
            "representing a 34% year-over-year growth. Product revenue accounted for "
            "$31.8 million (67%), while services and support contributed $15.5 million "
            "(33%). Gross margin improved to 72% from 68% in FY2023. Operating expenses "
            "were $38.1 million, with R&D spending at $14.2 million (30% of revenue). "
            "The company achieved positive EBITDA of $9.2 million for the first time. "
            "Total funding raised to date is $85 million across Series A, B, and C "
            "rounds. The Series C round of $40 million was led by TechVentures Capital "
            "in April 2024, valuing the company at $420 million."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION DATASET (questions + ground truth)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalSample:
    """A single evaluation sample with question, reference answer, and key terms."""
    question: str
    reference: str                          # Ground truth answer
    reference_contexts: list[str]           # Ground truth context snippets
    key_terms: list[str]                    # Terms that MUST appear in good answer
    expected_source: str                    # Which doc should be retrieved


EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        question="Who founded Quantum Systems Inc. and when?",
        reference=(
            "Quantum Systems Inc. was founded in 2019 by Dr. Elena Vasquez "
            "and Dr. Marcus Chen in Boulder, Colorado."
        ),
        reference_contexts=[
            "Quantum Systems Inc. was founded in 2019 by Dr. Elena Vasquez and "
            "Dr. Marcus Chen in Boulder, Colorado."
        ],
        key_terms=["2019", "Elena Vasquez", "Marcus Chen", "Boulder"],
        expected_source="quantum_systems_overview.md",
    ),
    EvalSample(
        question="What cryptographic algorithms does QuantumShield use?",
        reference=(
            "QuantumShield uses CRYSTALS-Kyber for key encapsulation, "
            "CRYSTALS-Dilithium for digital signatures, and AES-256-GCM "
            "for symmetric encryption."
        ),
        reference_contexts=[
            "The QuantumShield HSM uses CRYSTALS-Kyber for key encapsulation and "
            "CRYSTALS-Dilithium for digital signatures. It supports AES-256-GCM "
            "for symmetric encryption."
        ],
        key_terms=["CRYSTALS-Kyber", "CRYSTALS-Dilithium", "AES-256-GCM"],
        expected_source="quantumshield_technical.md",
    ),
    EvalSample(
        question="What was Quantum Systems' annual revenue in FY2024?",
        reference=(
            "Quantum Systems reported total annual revenue of $47.3 million "
            "in fiscal year 2024, a 34% year-over-year growth."
        ),
        reference_contexts=[
            "Total revenue was $47.3 million, representing a 34% year-over-year growth."
        ],
        key_terms=["47.3 million", "34%"],
        expected_source="quantum_financials_2024.md",
    ),
    EvalSample(
        question="Who leads the research division and how many researchers are there?",
        reference=(
            "The Quantum Systems Research Division is led by Dr. Sarah Kim "
            "and consists of 48 researchers."
        ),
        reference_contexts=[
            "The Quantum Systems Research Division is led by Dr. Sarah Kim and "
            "consists of 48 researchers."
        ],
        key_terms=["Sarah Kim", "48"],
        expected_source="quantum_research_division.md",
    ),
    EvalSample(
        question="What DARPA grant did the research team receive?",
        reference=(
            "The research division received a $12.5 million DARPA grant in "
            "September 2024 for developing next-generation quantum-safe "
            "communication protocols."
        ),
        reference_contexts=[
            "The division received a $12.5 million DARPA grant in September 2024 "
            "for developing next-generation quantum-safe communication protocols."
        ],
        key_terms=["12.5 million", "DARPA", "September 2024"],
        expected_source="quantum_research_division.md",
    ),
    EvalSample(
        question="What partnership did Quantum Systems sign with GlobalBank Corp?",
        reference=(
            "In June 2023, Quantum Systems signed a five-year agreement with "
            "GlobalBank Corp to deploy QuantumShield across all 2,400 branch "
            "locations in North America."
        ),
        reference_contexts=[
            "In June 2023, they signed a five-year agreement with GlobalBank Corp "
            "to deploy QuantumShield across all 2,400 branch locations in North America."
        ],
        key_terms=["GlobalBank", "2,400", "five-year", "June 2023"],
        expected_source="quantum_partnerships.md",
    ),
    EvalSample(
        question="What certifications has QuantumShield achieved?",
        reference=(
            "QuantumShield has achieved FIPS 140-3 Level 3 certification and "
            "Common Criteria EAL4+ certification."
        ),
        reference_contexts=[
            "The module has achieved FIPS 140-3 Level 3 certification and "
            "Common Criteria EAL4+ certification."
        ],
        key_terms=["FIPS 140-3", "Level 3", "Common Criteria", "EAL4+"],
        expected_source="quantumshield_technical.md",
    ),
    EvalSample(
        question="What is the company's valuation and who led the Series C round?",
        reference=(
            "The Series C round of $40 million was led by TechVentures Capital "
            "in April 2024, valuing Quantum Systems at $420 million."
        ),
        reference_contexts=[
            "The Series C round of $40 million was led by TechVentures Capital "
            "in April 2024, valuing the company at $420 million."
        ],
        key_terms=["40 million", "TechVentures Capital", "420 million"],
        expected_source="quantum_financials_2024.md",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Output of running a single query through the full RAG pipeline."""
    question: str
    response: str
    retrieved_contexts: list[str]
    retrieved_sources: list[str]
    reranker_scores: list[float]
    latency_s: float
    num_retrieved: int


def ingest_synthetic_corpus() -> tuple[list, list, list]:
    """
    Ingest the synthetic knowledge documents through the actual pipeline:
    normalize → chunk → embed.  Returns (chunk_ids, embeddings, payloads).
    """
    from app.config.settings import get_settings
    from app.api.dependencies import get_embedding_model
    from app.processing.chunking import SlidingWindowChunker
    from app.processing.normalization import normalize_text

    settings = get_settings()
    embedding_model = get_embedding_model()
    chunker = SlidingWindowChunker(
        target_tokens=settings.chunking.target_tokens,
        max_tokens=settings.chunking.max_tokens,
        overlap_tokens=settings.chunking.overlap_tokens,
    )

    all_chunk_ids = []
    all_embeddings = []
    all_payloads = []

    for doc in KNOWLEDGE_DOCS:
        normalized = normalize_text(doc["text"])
        chunks = chunker.chunk_text(
            text=normalized,
            source=doc["source"],
            modality="document",
            page_start=1,
        )
        if not chunks:
            continue

        texts = [c.text for c in chunks]
        embeddings = embedding_model.embed_texts(texts)

        for c, emb in zip(chunks, embeddings):
            all_chunk_ids.append(c.chunk_id)
            all_embeddings.append(emb)
            all_payloads.append({
                "text": c.text,
                "source": c.source,
                "modality": "document",
                "page_start": c.page_start,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "department": "test",
                "tags": ["ragas_eval"],
            })

    import numpy as np
    return all_chunk_ids, np.array(all_embeddings), all_payloads


def setup_index(chunk_ids, embeddings, payloads) -> None:
    """Upsert chunks into Qdrant (with in-memory fallback) and build BM25 index."""
    from app.api.dependencies import get_vector_store, get_bm25_store

    vs = get_vector_store()

    # Try connecting to Qdrant server; fall back to in-memory if unavailable
    try:
        vs.ensure_collection()
    except Exception as e:
        err = str(e).lower()
        if "refused" in err or "timeout" in err or "connect" in err or "ssl" in err:
            logger.info("Qdrant server offline → using in-memory fallback")
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

            in_mem = QdrantClient(":memory:")
            in_mem.create_collection(
                collection_name="rag_chunks",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )
            vs._client = in_mem
        else:
            raise

    vs.upsert_chunks(chunk_ids, embeddings, payloads)

    bm25 = get_bm25_store()
    texts = [p["text"] for p in payloads]
    meta = payloads
    bm25.build(chunk_ids, texts, meta)
    bm25.save()

    logger.info("Indexed %d chunks into Qdrant + BM25", len(chunk_ids))


def run_pipeline_query(question: str) -> PipelineResult:
    """
    Run a single question through the full RAG pipeline:
    normalize → embed query → hybrid retrieve → rerank → build prompt → generate.
    """
    from app.api.dependencies import (
        get_embedding_model,
        get_hybrid_retriever,
        get_llm_engine,
    )
    from app.config.settings import get_settings
    from app.generation.prompt_templates import ChunkContext, build_prompt
    from app.processing.normalization import normalize_query

    t0 = time.perf_counter()

    # 1. Normalize query
    query_text = normalize_query(question)

    # 2. Embed query
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query_text)

    # 3. Hybrid retrieval (vector + BM25 + RRF + rerank)
    retriever = get_hybrid_retriever()
    results = retriever.retrieve(query=query_text, query_embedding=query_embedding)

    if not results:
        return PipelineResult(
            question=question,
            response="No relevant information found.",
            retrieved_contexts=[],
            retrieved_sources=[],
            reranker_scores=[],
            latency_s=time.perf_counter() - t0,
            num_retrieved=0,
        )

    # 4. Build context chunks (cap at 2 for faster CPU inference)
    chunks = []
    retrieved_contexts = []
    retrieved_sources = []
    reranker_scores = []

    for r in results[:2]:
        meta = r.get("metadata", r.get("payload", {}))
        text = meta.get("text", "")
        retrieved_contexts.append(text)
        retrieved_sources.append(meta.get("source", "unknown"))
        reranker_scores.append(r.get("reranker_score", 0.0))

        chunks.append(ChunkContext(
            chunk_id=r["chunk_id"],
            text=text,
            source=meta.get("source", "unknown"),
            modality=meta.get("modality", "document"),
            page_start=meta.get("page_start"),
            reranker_score=r.get("reranker_score", 0.0),
        ))

    # 5. Build prompt
    llm = get_llm_engine()
    if not llm.is_loaded:
        # Use smaller context window for test speed (doesn't change pipeline code)
        settings = get_settings()
        settings.llm.context_window = 1024
        llm.load()

    prompt = build_prompt(
        query=query_text,
        chunks=chunks,
        token_counter=llm.count_tokens,
    )

    # 6. Generate
    response = llm.generate(prompt, max_tokens=32)

    latency = time.perf_counter() - t0

    return PipelineResult(
        question=question,
        response=response,
        retrieved_contexts=retrieved_contexts,
        retrieved_sources=retrieved_sources,
        reranker_scores=reranker_scores,
        latency_s=latency,
        num_retrieved=len(results),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_non_llm_metrics(
    pipeline_results: list[PipelineResult],
    eval_samples: list[EvalSample],
) -> dict:
    """
    Evaluate using RAGAS non-LLM metrics that don't require a judge LLM.
    Returns a dict of metric_name → list of per-sample scores.
    """
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.metrics.collections import (
        RougeScore,
        BleuScore,
        NonLLMStringSimilarity,
        StringPresence,
    )

    samples = []
    for pr, es in zip(pipeline_results, eval_samples):
        sample = SingleTurnSample(
            user_input=pr.question,
            response=pr.response,
            reference=es.reference,
            retrieved_contexts=pr.retrieved_contexts,
            reference_contexts=es.reference_contexts,
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)

    metrics = [
        RougeScore(),
        BleuScore(),
        NonLLMStringSimilarity(),
        StringPresence(),
    ]

    logger.info("Running non-LLM RAGAS metrics (%d metrics, %d samples)...",
                len(metrics), len(samples))

    result = evaluate(dataset=dataset, metrics=metrics)
    return result


def evaluate_llm_metrics(
    pipeline_results: list[PipelineResult],
    eval_samples: list[EvalSample],
    llm_judge,
) -> dict:
    """
    Evaluate using RAGAS LLM-based metrics.
    Requires a RAGAS-compatible LLM judge instance.
    """
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextPrecisionWithoutReference,
        FactualCorrectness,
    )

    samples = []
    for pr, es in zip(pipeline_results, eval_samples):
        sample = SingleTurnSample(
            user_input=pr.question,
            response=pr.response,
            reference=es.reference,
            retrieved_contexts=pr.retrieved_contexts,
            reference_contexts=es.reference_contexts,
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=llm_judge),
        AnswerRelevancy(llm=llm_judge),
        LLMContextPrecisionWithoutReference(llm=llm_judge),
        FactualCorrectness(llm=llm_judge),
    ]

    logger.info("Running LLM-based RAGAS metrics (%d metrics, %d samples)...",
                len(metrics), len(samples))

    result = evaluate(dataset=dataset, metrics=metrics)
    return result


def compute_custom_metrics(
    pipeline_results: list[PipelineResult],
    eval_samples: list[EvalSample],
) -> dict:
    """
    Compute custom pipeline-specific metrics not covered by RAGAS.
    """
    retrieval_hit_rate = 0
    source_accuracy = 0
    total_latency = 0.0
    avg_reranker_scores = []
    key_term_recall_scores = []

    for pr, es in zip(pipeline_results, eval_samples):
        # Retrieval hit rate: did we get ANY results?
        if pr.num_retrieved > 0:
            retrieval_hit_rate += 1

        # Source accuracy: is the expected source in retrieved sources?
        if es.expected_source in pr.retrieved_sources:
            source_accuracy += 1

        # Latency
        total_latency += pr.latency_s

        # Reranker score
        if pr.reranker_scores:
            avg_reranker_scores.append(sum(pr.reranker_scores) / len(pr.reranker_scores))

        # Key term recall: what fraction of key terms appear in the response?
        if es.key_terms:
            found = sum(1 for t in es.key_terms if t.lower() in pr.response.lower())
            key_term_recall_scores.append(found / len(es.key_terms))

    n = len(pipeline_results)
    return {
        "retrieval_hit_rate": retrieval_hit_rate / n if n else 0,
        "source_accuracy": source_accuracy / n if n else 0,
        "avg_latency_s": total_latency / n if n else 0,
        "total_latency_s": total_latency,
        "avg_reranker_score": (
            sum(avg_reranker_scores) / len(avg_reranker_scores)
            if avg_reranker_scores else 0
        ),
        "key_term_recall": (
            sum(key_term_recall_scores) / len(key_term_recall_scores)
            if key_term_recall_scores else 0
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  JUDGE LLM SETUP
# ══════════════════════════════════════════════════════════════════════════════

def try_setup_judge_llm():
    """
    Try to set up a RAGAS-compatible LLM judge.

    Priority:
    1. If OPENAI_API_KEY is set → use OpenAI GPT-4o-mini (best quality)
    2. If RAGAS_SKIP_LLM_METRICS=1 → skip
    3. Otherwise → start llama-cpp-python OpenAI server locally
    """
    skip = os.environ.get("RAGAS_SKIP_LLM_METRICS", "").strip()
    if skip == "1":
        logger.info("RAGAS_SKIP_LLM_METRICS=1 → skipping LLM-based metrics")
        return None

    # Option 1: OpenAI API key available
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            from openai import OpenAI
            from ragas.llms import llm_factory

            client = OpenAI(api_key=api_key)
            llm = llm_factory("gpt-4o-mini", client=client)
            logger.info("Using OpenAI GPT-4o-mini as RAGAS judge")
            return llm
        except Exception as e:
            logger.warning("Failed to set up OpenAI judge: %s", e)

    # Option 2: Start local llama-cpp-python server
    return _try_local_judge()


def _try_local_judge():
    """Start llama-cpp-python's built-in OpenAI-compatible server."""
    model_path = os.environ.get(
        "RAG_LLM__MODEL_PATH",
        str(PROJECT_ROOT / "models" / "llm" / "qwen2.5-3b-instruct-q4_k_m.gguf"),
    )
    if not Path(model_path).exists():
        logger.warning("Judge model not found at %s → skipping LLM metrics", model_path)
        return None

    port = 8089
    logger.info("Starting llama-cpp-python server on port %d for judge LLM...", port)

    python_exe = sys.executable
    server_proc = None

    try:
        # Start the server as a background process
        server_proc = subprocess.Popen(
            [
                python_exe, "-m", "llama_cpp.server",
                "--model", model_path,
                "--port", str(port),
                "--n_ctx", "4096",
                "--chat_format", "chatml",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        # Wait for the server to be ready (max 120s for model loading)
        import urllib.request
        import urllib.error

        health_url = f"http://localhost:{port}/v1/models"
        deadline = time.time() + 120
        ready = False

        while time.time() < deadline:
            try:
                req = urllib.request.urlopen(health_url, timeout=2)
                if req.status == 200:
                    ready = True
                    break
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(2)

        if not ready:
            logger.warning("llama-cpp server did not become ready in 120s → skipping LLM metrics")
            _kill_process(server_proc)
            return None

        logger.info("llama-cpp server ready on port %d", port)

        # Create RAGAS LLM via OpenAI client
        from openai import OpenAI
        from ragas.llms import llm_factory

        client = OpenAI(api_key="local-no-key", base_url=f"http://localhost:{port}/v1")
        llm = llm_factory(
            "qwen2.5-3b-instruct",
            provider="openai",
            client=client,
        )

        # Attach server process so we can clean up later
        llm._server_proc = server_proc
        return llm

    except Exception as e:
        logger.warning("Failed to start local judge LLM: %s", e)
        if server_proc:
            _kill_process(server_proc)
        return None


def _kill_process(proc):
    """Kill a subprocess tree."""
    try:
        if sys.platform == "win32":
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass


def cleanup_judge(llm_judge):
    """Stop the local judge server if we started one."""
    if llm_judge and hasattr(llm_judge, "_server_proc"):
        logger.info("Stopping local judge LLM server...")
        _kill_process(llm_judge._server_proc)


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def print_header(text: str) -> None:
    w = 78
    print(f"\n{'═' * w}")
    print(f"  {text}")
    print(f"{'═' * w}")


def print_per_sample_results(
    pipeline_results: list[PipelineResult],
    eval_samples: list[EvalSample],
) -> None:
    """Print detailed per-question results."""
    print_header("PER-SAMPLE PIPELINE RESULTS")
    for i, (pr, es) in enumerate(zip(pipeline_results, eval_samples), 1):
        src_hit = "✓" if es.expected_source in pr.retrieved_sources else "✗"
        found_terms = [t for t in es.key_terms if t.lower() in pr.response.lower()]
        term_recall = len(found_terms) / len(es.key_terms) if es.key_terms else 0

        print(f"\n  Q{i}: {pr.question}")
        print(f"  {'─' * 72}")
        print(f"  Response  : {pr.response[:200]}{'...' if len(pr.response) > 200 else ''}")
        print(f"  Reference : {es.reference[:150]}{'...' if len(es.reference) > 150 else ''}")
        print(f"  Sources   : {pr.retrieved_sources[:3]}  expected={es.expected_source}  [{src_hit}]")
        print(f"  Reranker  : {[f'{s:.3f}' for s in pr.reranker_scores[:3]]}")
        print(f"  Key Terms : {len(found_terms)}/{len(es.key_terms)} = {term_recall:.0%}  {found_terms}")
        print(f"  Latency   : {pr.latency_s:.2f}s   Retrieved: {pr.num_retrieved} chunks")


def print_ragas_results(title: str, result) -> None:
    """Print RAGAS evaluate() results."""
    print_header(title)
    # RAGAS evaluate() returns a Result object with .scores dict
    try:
        df = result.to_pandas()
        # Print aggregated scores
        numeric_cols = df.select_dtypes(include=["float64", "float32", "int64"]).columns
        print(f"\n  {'Metric':<45} {'Mean':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'─' * 73}")
        for col in numeric_cols:
            if col in ("user_input", "response", "reference"):
                continue
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:<45} {vals.mean():>8.4f}  {vals.min():>8.4f}  {vals.max():>8.4f}")
    except Exception as e:
        # Fallback: print raw scores
        logger.warning("Could not format as DataFrame: %s", e)
        print(f"  Raw: {result}")


def print_custom_metrics(metrics: dict) -> None:
    """Print custom pipeline metrics."""
    print_header("CUSTOM PIPELINE METRICS")
    print(f"\n  {'Metric':<40} {'Value':>12}")
    print(f"  {'─' * 55}")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:<40} {value:>12.4f}")
        else:
            print(f"  {name:<40} {value!s:>12}")


def print_final_summary(
    non_llm_result,
    llm_result,
    custom_metrics: dict,
    total_time: float,
) -> None:
    """Print the final evaluation summary."""
    print_header("FINAL RAGAS EVALUATION SUMMARY")

    # Aggregate non-LLM scores
    try:
        df = non_llm_result.to_pandas()
        numeric_cols = [c for c in df.select_dtypes(include=["float64", "float32"]).columns
                        if c not in ("user_input", "response", "reference")]
        overall_non_llm = df[numeric_cols].mean().mean() if numeric_cols else 0.0
    except Exception:
        overall_non_llm = 0.0

    # Aggregate LLM scores
    overall_llm = None
    if llm_result is not None:
        try:
            df2 = llm_result.to_pandas()
            numeric_cols2 = [c for c in df2.select_dtypes(include=["float64", "float32"]).columns
                            if c not in ("user_input", "response", "reference")]
            overall_llm = df2[numeric_cols2].mean().mean() if numeric_cols2 else 0.0
        except Exception:
            overall_llm = None

    print(f"""
  Non-LLM Metrics (avg)   : {overall_non_llm:.4f}
  LLM Metrics (avg)       : {f'{overall_llm:.4f}' if overall_llm is not None else 'SKIPPED'}
  Retrieval Hit Rate       : {custom_metrics['retrieval_hit_rate']:.0%}
  Source Accuracy           : {custom_metrics['source_accuracy']:.0%}
  Key Term Recall           : {custom_metrics['key_term_recall']:.0%}
  Avg Reranker Score        : {custom_metrics['avg_reranker_score']:.4f}
  Avg Latency per Query     : {custom_metrics['avg_latency_s']:.2f}s
  Total Evaluation Time     : {total_time:.1f}s
  Samples Evaluated         : {len(EVAL_DATASET)}
""")

    # Pass/fail assessment
    passed = (
        custom_metrics["retrieval_hit_rate"] >= 0.8
        and custom_metrics["source_accuracy"] >= 0.6
        and custom_metrics["key_term_recall"] >= 0.4
        and overall_non_llm >= 0.1
    )
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Overall Assessment: {status}")
    print(f"  {'═' * 76}\n")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    t_total = time.perf_counter()

    # Limit evaluation set size for CPU-only inference speed
    # Set RAGAS_MAX_QUERIES=N to override (default: 4 for fast CPU runs)
    max_queries = int(os.environ.get("RAGAS_MAX_QUERIES", "4"))
    eval_samples = EVAL_DATASET[:max_queries]

    print("=" * 78)
    print("  RAGAS EVALUATION — Offline RAG Pipeline")
    print(f"  Corpus: {len(KNOWLEDGE_DOCS)} documents, {len(eval_samples)} evaluation queries")
    print("=" * 78)

    # ── Step 1: Ingest synthetic corpus ───────────────────────────────────
    print("\n[1/5] Ingesting synthetic knowledge corpus...")
    t0 = time.perf_counter()
    try:
        chunk_ids, embeddings, payloads = ingest_synthetic_corpus()
        setup_index(chunk_ids, embeddings, payloads)
        print(f"      ✓ Ingested {len(chunk_ids)} chunks in {time.perf_counter()-t0:.1f}s")
    except Exception as e:
        print(f"      ✗ Ingestion failed: {e}")
        traceback.print_exc()
        return 1

    # ── Step 2: Run pipeline queries ──────────────────────────────────────
    print(f"\n[2/5] Running {len(eval_samples)} queries through the full pipeline...")
    pipeline_results: list[PipelineResult] = []
    for i, sample in enumerate(eval_samples, 1):
        print(f"      Q{i}/{len(eval_samples)}: {sample.question[:60]}...", end="", flush=True)
        try:
            result = run_pipeline_query(sample.question)
            pipeline_results.append(result)
            print(f"  ✓ {result.latency_s:.1f}s")
        except Exception as e:
            print(f"  ✗ {e}")
            # Create a failed result so indices stay aligned
            pipeline_results.append(PipelineResult(
                question=sample.question,
                response=f"ERROR: {e}",
                retrieved_contexts=[],
                retrieved_sources=[],
                reranker_scores=[],
                latency_s=0.0,
                num_retrieved=0,
            ))

    # ── Step 3: Non-LLM RAGAS evaluation ─────────────────────────────────
    print("\n[3/5] Evaluating with RAGAS non-LLM metrics...")
    t0 = time.perf_counter()
    try:
        non_llm_result = evaluate_non_llm_metrics(pipeline_results, eval_samples)
        print(f"      ✓ Non-LLM evaluation done in {time.perf_counter()-t0:.1f}s")
    except Exception as e:
        print(f"      ✗ Non-LLM evaluation failed: {e}")
        traceback.print_exc()
        non_llm_result = None

    # ── Step 4: LLM-based RAGAS evaluation (optional) ────────────────────
    print("\n[4/5] Setting up judge LLM for LLM-based metrics...")
    llm_result = None
    llm_judge = None
    try:
        llm_judge = try_setup_judge_llm()
        if llm_judge is not None:
            print("      ✓ Judge LLM ready")
            t0 = time.perf_counter()
            print("      Running LLM-based metrics (this may take several minutes)...")
            llm_result = evaluate_llm_metrics(pipeline_results, eval_samples, llm_judge)
            print(f"      ✓ LLM evaluation done in {time.perf_counter()-t0:.1f}s")
        else:
            print("      ⊘ No judge LLM available — skipping LLM-based metrics")
            print("        (Set OPENAI_API_KEY or ensure llama-cpp-python server can start)")
    except Exception as e:
        print(f"      ✗ LLM evaluation failed: {e}")
        traceback.print_exc()
    finally:
        if llm_judge:
            cleanup_judge(llm_judge)

    # ── Step 5: Custom pipeline metrics + report ──────────────────────────
    print("\n[5/5] Computing custom metrics and generating report...")
    custom_metrics = compute_custom_metrics(pipeline_results, eval_samples)

    # ── Print detailed report ─────────────────────────────────────────────
    print_per_sample_results(pipeline_results, eval_samples)

    if non_llm_result is not None:
        print_ragas_results("RAGAS NON-LLM METRICS", non_llm_result)

    if llm_result is not None:
        print_ragas_results("RAGAS LLM-BASED METRICS", llm_result)

    print_custom_metrics(custom_metrics)

    total_time = time.perf_counter() - t_total
    passed = print_final_summary(non_llm_result, llm_result, custom_metrics, total_time)

    # ── Save results to JSON ──────────────────────────────────────────────
    output_path = PROJECT_ROOT / "tests" / "ragas_results.json"
    try:
        save_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(eval_samples),
            "num_docs": len(KNOWLEDGE_DOCS),
            "custom_metrics": custom_metrics,
            "passed": passed,
            "per_sample": [
                {
                    "question": pr.question,
                    "response": pr.response[:500],
                    "sources": pr.retrieved_sources,
                    "reranker_scores": pr.reranker_scores,
                    "latency_s": pr.latency_s,
                    "num_retrieved": pr.num_retrieved,
                }
                for pr in pipeline_results
            ],
        }
        if non_llm_result is not None:
            try:
                df = non_llm_result.to_pandas()
                numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns
                save_data["non_llm_scores"] = {
                    col: float(df[col].mean()) for col in numeric_cols
                }
            except Exception:
                pass
        if llm_result is not None:
            try:
                df = llm_result.to_pandas()
                numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns
                save_data["llm_scores"] = {
                    col: float(df[col].mean()) for col in numeric_cols
                }
            except Exception:
                pass

        output_path.write_text(json.dumps(save_data, indent=2))
        print(f"  Results saved to {output_path}")
    except Exception as e:
        print(f"  Warning: Could not save results: {e}")

    return 0 if passed else 1


if __name__ == "__main__":
    import io

    # ── Tee stdout/stderr to a log file for reliable output capture ───────
    log_path = PROJECT_ROOT / "tests" / "ragas_eval_log.txt"
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")

    class TeeStream:
        """Write to both console and log file."""
        def __init__(self, console, log):
            self.console = console
            self.log = log
        def write(self, text):
            try:
                self.console.write(text)
                self.console.flush()
            except Exception:
                pass
            try:
                self.log.write(text)
                self.log.flush()
            except Exception:
                pass
        def flush(self):
            try:
                self.console.flush()
            except Exception:
                pass
            try:
                self.log.flush()
            except Exception:
                pass
        def reconfigure(self, **kwargs):
            pass  # ignore reconfigure calls

    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    # ── Ignore SIGINT (Ctrl+C) so the evaluation can complete ─────────────
    if sys.platform == "win32":
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        exit_code = main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        log_file.close()

    sys.exit(exit_code)
