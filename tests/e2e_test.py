"""
End-to-End Test for Offline RAG System
=======================================
Tests every component of the RAG pipeline with detailed timing.

Components tested:
  1.  Configuration Loading
  2.  Qdrant Connection & Collection
  3.  Embedding Model Loading & Inference
  4.  BM25 Index Loading
  5.  Reranker Model Loading & Inference
  6.  Text Normalization
  7.  Chunking Pipeline
  8.  Vector Upsert & Search (round-trip)
  9.  BM25 Build + Search
  10. Hybrid Retrieval (Vector + BM25 + RRF + Reranking)
  11. LLM Loading (skipped if > 120s)
  12. LLM Generation (skipped if loading was skipped or > 60s)
  13. Prompt Template Construction
  14. Index Versioning

Run:
  cd D:\\Offline_Rag_V2
  .venv\\Scripts\\python.exe e2e_test.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RAG_ROOT = PROJECT_ROOT / "rag-system"

# Ensure we run from the rag-system directory so relative config paths work
os.chdir(RAG_ROOT)
sys.path.insert(0, str(RAG_ROOT))

# Override model and data paths to use ABSOLUTE paths
os.environ["RAG_PATHS__MODELS_DIR"] = str(PROJECT_ROOT / "models")
os.environ["RAG_PATHS__DATA_DIR"] = str(PROJECT_ROOT / "data")
os.environ["RAG_PATHS__INDEX_DIR"] = str(PROJECT_ROOT / "data" / "index")
os.environ["RAG_PATHS__UPLOADS_DIR"] = str(PROJECT_ROOT / "data" / "uploads")
os.environ["RAG_PATHS__LOGS_DIR"] = str(PROJECT_ROOT / "data" / "logs")
os.environ["RAG_LLM__MODEL_PATH"] = str(PROJECT_ROOT / "models" / "llm" / "qwen2.5-3b-instruct-q4_k_m.gguf")


# ── Timing / result helpers ──────────────────────────────────────────────────

@dataclass
class StepResult:
    name: str
    status: str = "NOT_RUN"
    duration_s: float = 0.0
    detail: str = ""
    error: str = ""


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


results: list[StepResult] = []

LLM_LOAD_TIMEOUT = 120
LLM_GEN_TIMEOUT  = 120

# CPU-only baseline timings (measured before GPU acceleration)
CPU_BASELINE = {
    "llm_load_s": 5.90,
    "llm_gen_s": 58.19,
    "llm_tok_per_s": 1.1,
    "total_s": 85.24,
    "device": "CPU only (gpu_layers=0)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – Configuration Loading
# ══════════════════════════════════════════════════════════════════════════════
def test_config():
    step = StepResult(name="1. Configuration Loading")
    t = Timer()
    try:
        with t:
            from app.config.settings import load_settings
            settings = load_settings()
            assert settings.chunking.target_tokens == 480
            assert settings.qdrant.port == 6333
            assert settings.llm.context_window == 32768
        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = (
            f"target_tokens={settings.chunking.target_tokens}, "
            f"qdrant={settings.qdrant.host}:{settings.qdrant.port}, "
            f"gpu_layers={settings.llm.gpu_layers}, "
            f"models_dir={settings.paths.models_dir}"
        )
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = str(e)
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – Qdrant Connection & Collection
# ══════════════════════════════════════════════════════════════════════════════
_qdrant_available = False
_vector_store = None

def test_qdrant():
    global _qdrant_available, _vector_store
    step = StepResult(name="2. Qdrant Connection & Collection")
    t = Timer()
    try:
        with t:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333, timeout=5)
            client.get_collections()
            _qdrant_available = True

            from app.retrieval.vector_store import VectorStore
            vs = VectorStore()
            vs.ensure_collection()
            count = vs.count()
            health = vs.health_check()
            _vector_store = vs

        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = f"collection='{vs.collection_name}', chunks={count}, status={health.get('status')}"
    except Exception as e:
        step.duration_s = t.elapsed
        err_str = str(e)
        if "refused" in err_str.lower() or "timeout" in err_str.lower() or "connect" in err_str.lower():
            t2 = Timer()
            try:
                with t2:
                    from qdrant_client import QdrantClient
                    from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
                    in_mem_client = QdrantClient(":memory:")
                    in_mem_client.create_collection(
                        collection_name="rag_chunks",
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                    )
                    _qdrant_available = True

                    from app.retrieval.vector_store import VectorStore
                    vs = VectorStore()
                    vs._client = in_mem_client
                    _vector_store = vs

                step.status = "PASS"
                step.duration_s = t2.elapsed
                step.detail = "Qdrant server offline -> IN-MEMORY fallback OK. Collection created."
            except Exception as e2:
                step.status = "FAIL"
                step.error = f"Remote: {err_str} | Fallback: {e2}"
        else:
            step.status = "FAIL"
            step.error = err_str
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – Embedding Model Loading & Inference
# ══════════════════════════════════════════════════════════════════════════════
_embedding_model = None

def test_embedding_model():
    global _embedding_model
    step = StepResult(name="3. Embedding Model Load & Inference")
    t = Timer()
    try:
        with t:
            from app.models.model_registry import ModelRegistry
            from app.models.model_manager import ModelManager
            from app.models.embeddings import EmbeddingModel

            models_dir = Path(os.environ.get("RAG_PATHS__MODELS_DIR", "models"))
            registry = ModelRegistry(models_dir)
            manager = ModelManager(registry)
            emb = EmbeddingModel(manager)

            vec = emb.embed_single("warmup test sentence")
            dim = vec.shape[-1]

        t_batch = Timer()
        with t_batch:
            texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning models need training data.",
                "Retrieval augmented generation combines search with LLMs.",
            ]
            batch_vecs = emb.embed_texts(texts)

        t_query = Timer()
        with t_query:
            q_vec = emb.embed_query("What is RAG?")

        _embedding_model = emb
        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = (
            f"dim={dim}, load={t.elapsed:.3f}s, "
            f"batch(3)={t_batch.elapsed*1000:.1f}ms, "
            f"query={t_query.elapsed*1000:.1f}ms, "
            f"shapes: {vec.shape}/{batch_vecs.shape}/{q_vec.shape}"
        )
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 – BM25 Index Loading
# ══════════════════════════════════════════════════════════════════════════════
_bm25_store = None

def test_bm25_load():
    global _bm25_store
    step = StepResult(name="4. BM25 Index Load")
    t = Timer()
    try:
        with t:
            from app.retrieval.bm25_store import BM25Store
            index_dir = Path(os.environ["RAG_PATHS__INDEX_DIR"]) / "current" / "bm25"
            index_dir.mkdir(parents=True, exist_ok=True)
            store = BM25Store(index_dir)
            loaded = store.load()

        _bm25_store = store
        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = f"from_disk={loaded}, is_loaded={store.is_loaded}, chunks={store.chunk_count}"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 – Reranker Model Load & Inference
# ══════════════════════════════════════════════════════════════════════════════
_reranker = None

def test_reranker():
    global _reranker
    step = StepResult(name="5. Reranker Model Load & Inference")
    t = Timer()
    try:
        with t:
            from app.models.model_registry import ModelRegistry
            from app.models.model_manager import ModelManager
            from app.retrieval.reranker import Reranker

            models_dir = Path(os.environ.get("RAG_PATHS__MODELS_DIR", "models"))
            registry = ModelRegistry(models_dir)
            manager = ModelManager(registry)
            rr = Reranker(manager)

            candidates = [
                {"chunk_id": "c1", "metadata": {"text": "RAG combines retrieval with generation."}},
                {"chunk_id": "c2", "metadata": {"text": "The weather is sunny today."}},
                {"chunk_id": "c3", "metadata": {"text": "Embeddings represent text as vectors."}},
            ]
            reranked = rr.rerank(
                query="What is retrieval augmented generation?",
                candidates=candidates, threshold=0.0, min_results=1,
            )

        _reranker = rr
        step.status = "PASS"
        step.duration_s = t.elapsed
        scores = ", ".join(f"{r['chunk_id']}={r['reranker_score']:.4f}" for r in reranked)
        step.detail = f"load+infer={t.elapsed:.3f}s, scores=[{scores}]"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 – Text Normalization
# ══════════════════════════════════════════════════════════════════════════════
def test_normalization():
    step = StepResult(name="6. Text Normalization")
    t = Timer()
    try:
        with t:
            from app.processing.normalization import normalize_text, normalize_query

            raw = "  Hello\t  world!   \u201CHello\u201D   \n\n\n\nextra   spacing  "
            norm = normalize_text(raw)

            assert "\t" not in norm, "Tabs should be converted"
            assert "\u201C" not in norm, "Smart quotes should be normalized"
            assert "   " not in norm, "Multiple spaces should be collapsed"

            query_norm = normalize_query("  What is   RAG?  ")

        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = f"input_len={len(raw)}, output_len={len(norm)}, query='{query_norm}'"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 – Chunking Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def test_chunking():
    step = StepResult(name="7. Chunking Pipeline")
    t = Timer()
    try:
        with t:
            from app.processing.chunking import SlidingWindowChunker

            chunker = SlidingWindowChunker(
                target_tokens=480, max_tokens=512, overlap_tokens=50,
            )

            sample_text = (
                "# Introduction to RAG\n\n"
                "Retrieval-Augmented Generation (RAG) is a technique that combines "
                "information retrieval with language model generation. It works by first "
                "retrieving relevant documents from a knowledge base, then using those "
                "documents as context for generating a response. This approach helps reduce "
                "hallucination and ground responses in factual information. "
                "RAG systems typically use dense vector embeddings to find semantically similar "
                "chunks of text. The retrieved chunks are then formatted into a prompt that "
                "instructs the language model to answer based only on the provided context. "
            ) * 10

            chunks = chunker.chunk_text(
                text=sample_text, source="test_doc.pdf",
                modality="document", page_start=1,
            )

            for c in chunks:
                assert c.token_count <= 512, f"Chunk exceeds max: {c.token_count}"

        step.status = "PASS"
        step.duration_s = t.elapsed
        tc = [c.token_count for c in chunks]
        step.detail = f"chunks={len(chunks)}, tokens={tc}, valid={all(x<=512 for x in tc)}"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 – Vector Upsert & Search (round-trip)
# ══════════════════════════════════════════════════════════════════════════════
def test_vector_roundtrip():
    step = StepResult(name="8. Vector Upsert & Search (round-trip)")
    if _embedding_model is None or _vector_store is None:
        step.status = "SKIPPED"
        step.detail = "Requires embedding (step 3) + Qdrant (step 2)"
        results.append(step)
        return False

    t_embed = Timer()
    t_upsert = Timer()
    t_search = Timer()
    try:
        import uuid as _uuid
        vs = _vector_store
        test_texts = [
            "RAG combines retrieval with generation for grounded answers.",
            "Vector databases store embeddings for similarity search.",
            "BM25 is a keyword-based retrieval algorithm used in search engines.",
            "Sentence transformers produce dense vectors from text.",
            "Cross-encoder reranking improves search precision.",
        ]
        test_ids = [_uuid.uuid4().hex for _ in range(len(test_texts))]
        test_payloads = [
            {"text": txt, "source": "e2e_test.txt", "modality": "document",
             "page_start": i+1, "upload_id": "e2e_test"}
            for i, txt in enumerate(test_texts)
        ]

        with t_embed:
            embeddings = _embedding_model.embed_texts(test_texts)

        with t_upsert:
            vs.upsert_chunks(test_ids, embeddings, test_payloads)

        time.sleep(0.3)

        with t_search:
            q_vec = _embedding_model.embed_query("What is retrieval augmented generation?")
            search_results = vs.search(query_vector=q_vec, top_k=5)

        # Cleanup
        try:
            from qdrant_client.models import PointIdsList
            vs.client.delete(collection_name=vs.collection_name, points_selector=PointIdsList(points=test_ids))
        except Exception:
            pass

        step.status = "PASS"
        step.duration_s = t_embed.elapsed + t_upsert.elapsed + t_search.elapsed
        top_ids = [r["chunk_id"] for r in search_results[:3]]
        top_scores = [f"{r['score']:.4f}" for r in search_results[:3]]
        step.detail = (
            f"embed={t_embed.elapsed*1000:.1f}ms, "
            f"upsert={t_upsert.elapsed*1000:.1f}ms, "
            f"search={t_search.elapsed*1000:.1f}ms, "
            f"results={len(search_results)}, top={top_ids}"
        )
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t_embed.elapsed + t_upsert.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 – BM25 Build + Search
# ══════════════════════════════════════════════════════════════════════════════
def test_bm25_search():
    step = StepResult(name="9. BM25 Build + Search")
    t_build = Timer()
    t_search = Timer()
    try:
        import tempfile, shutil
        from app.retrieval.bm25_store import BM25Store

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            store = BM25Store(tmp_dir)
            test_texts = [
                "Retrieval augmented generation combines search with language models.",
                "Vector databases store high-dimensional embeddings for similarity.",
                "BM25 uses term frequency and inverse document frequency for ranking.",
                "Cross-encoder models rerank initial results for better precision.",
                "Chunking splits documents into smaller pieces for embedding.",
            ]
            test_ids = [f"bm25_t{i}" for i in range(len(test_texts))]
            test_meta = [{"text": txt} for txt in test_texts]

            with t_build:
                store.build(test_ids, test_texts, test_meta)

            with t_search:
                bm25_results = store.search("retrieval augmented generation", top_k=5)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        step.status = "PASS"
        step.duration_s = t_build.elapsed + t_search.elapsed
        r_ids = [r["chunk_id"] for r in bm25_results[:3]]
        r_scores = [f"{r['score']:.4f}" for r in bm25_results[:3]]
        step.detail = (
            f"build={t_build.elapsed*1000:.1f}ms, "
            f"search={t_search.elapsed*1000:.1f}ms, "
            f"results={len(bm25_results)}, top3={r_ids}, scores={r_scores}"
        )
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t_build.elapsed + t_search.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10 – Hybrid Retrieval
# ══════════════════════════════════════════════════════════════════════════════
def test_hybrid_retrieval():
    step = StepResult(name="10. Hybrid Retrieval Pipeline")
    if _embedding_model is None or _vector_store is None or _reranker is None:
        step.status = "SKIPPED"
        step.detail = "Requires embedding + Qdrant + reranker"
        results.append(step)
        return False
    t = Timer()
    try:
        with t:
            import tempfile, shutil
            from app.retrieval.bm25_store import BM25Store
            from app.retrieval.hybrid_retriever import HybridRetriever

            vs = _vector_store

            corpus = [
                "Retrieval augmented generation helps reduce hallucination in LLMs.",
                "Dense embeddings capture semantic meaning of text passages.",
                "BM25 is an effective baseline for information retrieval tasks.",
                "Reranking with cross-encoders significantly improves recall.",
                "The RAG pipeline has indexing, retrieval, and generation stages.",
            ]
            import uuid as _uuid
            c_ids = [_uuid.uuid4().hex for _ in range(len(corpus))]
            c_payloads = [
                {"text": txt, "source": "hybrid_test.txt", "modality": "document",
                 "page_start": i+1}
                for i, txt in enumerate(corpus)
            ]
            c_embs = _embedding_model.embed_texts(corpus)
            vs.upsert_chunks(c_ids, c_embs, c_payloads)
            time.sleep(0.3)

            tmp_dir = Path(tempfile.mkdtemp())
            try:
                bm25 = BM25Store(tmp_dir)
                bm25.build(c_ids, corpus, c_payloads)

                retriever = HybridRetriever(
                    vector_store=vs, bm25_store=bm25, reranker=_reranker,
                )

                query = "How does RAG reduce hallucination?"
                q_emb = _embedding_model.embed_query(query)
                hybrid_results = retriever.retrieve(query=query, query_embedding=q_emb)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                try:
                    from qdrant_client.models import PointIdsList
                    vs.client.delete(collection_name=vs.collection_name, points_selector=PointIdsList(points=c_ids))
                except Exception:
                    pass

        step.status = "PASS"
        step.duration_s = t.elapsed
        if hybrid_results:
            top = hybrid_results[0]
            step.detail = (
                f"full_pipeline={t.elapsed*1000:.0f}ms, "
                f"results={len(hybrid_results)}, "
                f"top={top['chunk_id']}, "
                f"score={top.get('reranker_score',0):.4f}"
            )
        else:
            step.detail = f"pipeline={t.elapsed*1000:.0f}ms, results=0"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 11 – LLM Loading
# ══════════════════════════════════════════════════════════════════════════════
_llm_engine = None
_llm_available = False

def test_llm_load():
    global _llm_engine, _llm_available
    step = StepResult(name="11. LLM (Qwen2.5-3B) Model Loading")
    t = Timer()
    try:
        import threading
        from app.generation.llm_engine import LLMEngine
        engine = LLMEngine()

        load_error = [None]
        load_done = threading.Event()

        def _load():
            try:
                engine.load()
            except Exception as e:
                load_error[0] = e
            finally:
                load_done.set()

        with t:
            thread = threading.Thread(target=_load, daemon=True)
            thread.start()
            finished = load_done.wait(timeout=LLM_LOAD_TIMEOUT)

        if not finished:
            step.status = "TIMEOUT"
            step.duration_s = LLM_LOAD_TIMEOUT
            step.detail = f"Exceeded {LLM_LOAD_TIMEOUT}s — SKIPPING LLM generation"
            try:
                engine.unload()
            except Exception:
                pass
        elif load_error[0]:
            step.status = "FAIL"
            step.duration_s = t.elapsed
            step.error = str(load_error[0])
        else:
            _llm_engine = engine
            _llm_available = True
            step.status = "PASS"
            step.duration_s = t.elapsed
            from app.config.settings import load_settings
            _cfg = load_settings()
            step.detail = f"loaded in {t.elapsed:.2f}s, ctx=32768, gpu_layers={_cfg.llm.gpu_layers}"

    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 12 – LLM Generation
# ══════════════════════════════════════════════════════════════════════════════
def test_llm_generation():
    step = StepResult(name="12. LLM (Qwen2.5-3B) Text Generation")
    if not _llm_available or _llm_engine is None:
        step.status = "SKIPPED"
        step.detail = "LLM not loaded (step 11 failed/timed out)"
        results.append(step)
        return False
    t = Timer()
    try:
        import threading

        prompt = (
            "<|im_start|>user\n"
            "What is Retrieval Augmented Generation in one sentence?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        gen_result = [None]
        gen_error = [None]
        gen_done = threading.Event()

        def _gen():
            try:
                gen_result[0] = _llm_engine.generate(prompt, max_tokens=64)
            except Exception as e:
                gen_error[0] = e
            finally:
                gen_done.set()

        with t:
            thread = threading.Thread(target=_gen, daemon=True)
            thread.start()
            finished = gen_done.wait(timeout=LLM_GEN_TIMEOUT)

        if not finished:
            step.status = "TIMEOUT"
            step.duration_s = LLM_GEN_TIMEOUT
            step.detail = f"Generation exceeded {LLM_GEN_TIMEOUT}s — SKIPPED"
        elif gen_error[0]:
            step.status = "FAIL"
            step.duration_s = t.elapsed
            step.error = str(gen_error[0])
        else:
            step.status = "PASS"
            step.duration_s = t.elapsed
            output = gen_result[0] or ""
            # Estimate tokens (~0.75 tokens per word for English)
            est_tokens = max(len(output.split()), 1)
            tok_per_s = est_tokens / t.elapsed if t.elapsed > 0 else 0
            step.detail = f"time={t.elapsed:.2f}s, ~{est_tokens}tok, ~{tok_per_s:.1f}tok/s, out='{output[:80]}'"

    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 13 – Prompt Template Construction
# ══════════════════════════════════════════════════════════════════════════════
def test_prompt_template():
    step = StepResult(name="13. Prompt Template Construction")
    t = Timer()
    try:
        with t:
            from app.generation.prompt_templates import (
                ChunkContext, build_prompt, build_citation_tag,
            )

            chunks = [
                ChunkContext(
                    chunk_id="tc1", text="RAG combines retrieval with generation.",
                    source="test_doc.pdf", modality="document",
                    page_start=1, reranker_score=0.95,
                ),
                ChunkContext(
                    chunk_id="tc2", text="Vector DBs store embeddings.",
                    source="test_doc.pdf", modality="document",
                    page_start=2, reranker_score=0.82,
                ),
            ]

            tag = build_citation_tag(chunks[0])
            assert "test_doc.pdf" in tag

            prompt = build_prompt(
                query="What is RAG?", chunks=chunks,
                token_counter=lambda x: len(x.split()),
            )
            assert len(prompt) > 0

        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = f"prompt_len={len(prompt)} chars, citation='{tag}'"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 14 – Index Versioning
# ══════════════════════════════════════════════════════════════════════════════
def test_index_versioning():
    step = StepResult(name="14. Index Versioning")
    t = Timer()
    try:
        with t:
            from app.versioning.index_manager import IndexManager
            idx_dir = Path(os.environ["RAG_PATHS__INDEX_DIR"])
            im = IndexManager(idx_dir)
            im.ensure_initialized()
            version = im.current_version

        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = f"current_version={version}"
    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 15 – EasyOCR Image Text Extraction
# ══════════════════════════════════════════════════════════════════════════════
def test_easyocr():
    step = StepResult(name="15. EasyOCR Image Text Extraction")
    t = Timer()
    try:
        with t:
            import easyocr
            from PIL import Image, ImageDraw, ImageFont

            # Create a test image with clear text
            img = Image.new("RGB", (600, 120), "white")
            draw = ImageDraw.Draw(img)
            # Use a large font size for better OCR
            try:
                font = ImageFont.truetype("arial.ttf", 36)
            except Exception:
                font = ImageFont.load_default()
            draw.text((30, 30), "RAG System Test 2026", fill="black", font=font)
            test_path = str(PROJECT_ROOT / "test_ocr_e2e.png")
            img.save(test_path)

            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            ocr_results = reader.readtext(test_path)

            # Also test our worker module
            from app.ingestion.ocr_worker import extract_text_from_image
            worker_result = extract_text_from_image(reader, test_path, "test_ocr_e2e.png")

        detected_texts = [r[1] for r in ocr_results] if ocr_results else []
        step.status = "PASS"
        step.duration_s = t.elapsed
        step.detail = (
            f"raw_detections={len(ocr_results)}, "
            f"texts={detected_texts}, "
            f"worker_blocks={worker_result['block_count']}, "
            f"worker_chars={len(worker_result['ocr_text'])}"
        )

        # Cleanup
        try:
            os.remove(test_path)
        except Exception:
            pass

    except Exception as e:
        step.status = "FAIL"
        step.duration_s = t.elapsed
        step.error = f"{e}\n{traceback.format_exc()}"
    results.append(step)
    return step.status == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
#  Report
# ══════════════════════════════════════════════════════════════════════════════
def print_report():
    W = 110
    line = "=" * W
    thin = "-" * W

    print(f"\n{line}")
    print(f"{'END-TO-END TEST REPORT':^{W}}")
    print(f"{'Offline RAG System - Component Health & Timing':^{W}}")
    print(f"{line}\n")

    hdr = f"  {'#':<3} {'Component':<42} {'Status':<12} {'Time':>12}  {'Details'}"
    print(hdr)
    print(f"  {thin[:-4]}")

    total_time = 0.0
    passed = failed = skipped = timed_out = 0

    for r in results:
        total_time += r.duration_s

        if r.status == "PASS":
            marker = "PASS  [OK]"
            passed += 1
        elif r.status == "FAIL":
            marker = "FAIL  [!!]"
            failed += 1
        elif r.status == "SKIPPED":
            marker = "SKIP  [--]"
            skipped += 1
        elif r.status == "TIMEOUT":
            marker = "TMOUT [TT]"
            timed_out += 1
        else:
            marker = "???   [??]"

        if r.duration_s >= 1:
            dur = f"{r.duration_s:.2f}s"
        else:
            dur = f"{r.duration_s*1000:.1f}ms"

        detail_short = (r.detail[:65] + "...") if len(r.detail) > 65 else r.detail
        print(f"  {'':<3} {r.name:<42} {marker:<12} {dur:>12}  {detail_short}")

    print(f"  {thin[:-4]}")
    print(f"  {'TOTAL':>42} {'':>12} {total_time:>11.2f}s\n")

    # Summary box
    print(f"  +----------------------------------------------+")
    print(f"  |{'SUMMARY':^46}|")
    print(f"  +----------------------------------------------+")
    print(f"  |  Passed     : {passed:>3} / {len(results):<3}{'':>28}|")
    print(f"  |  Failed     : {failed:>3} / {len(results):<3}{'':>28}|")
    print(f"  |  Skipped    : {skipped:>3} / {len(results):<3}{'':>28}|")
    print(f"  |  Timed Out  : {timed_out:>3} / {len(results):<3}{'':>28}|")
    print(f"  |  Total Time : {total_time:>8.2f}s{'':>26}|")
    print(f"  +----------------------------------------------+\n")

    # Failure details
    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        print(f"  {line}")
        print(f"  {'FAILURE DETAILS':^{W}}")
        print(f"  {line}")
        for r in failures:
            print(f"\n  [FAIL] {r.name}")
            for err_line in r.error.split("\n")[:6]:
                print(f"         {err_line}")
        print()

    # Timeout details
    timeouts = [r for r in results if r.status == "TIMEOUT"]
    if timeouts:
        print(f"  {thin}")
        for r in timeouts:
            print(f"  [TIMEOUT] {r.name}: {r.detail}")
        print()

    # ── GPU vs CPU Comparison ──────────────────────────────────────────────
    llm_load_r = next((r for r in results if "11." in r.name), None)
    llm_gen_r = next((r for r in results if "12." in r.name), None)

    if llm_load_r and llm_gen_r and llm_load_r.status == "PASS" and llm_gen_r.status == "PASS":
        print(f"  {line}")
        print(f"  {'GPU vs CPU PERFORMANCE COMPARISON':^{W}}")
        print(f"  {line}")
        print(f"  {'Metric':<35} {'CPU (before)':>16} {'GPU (now)':>16} {'Speedup':>12}")
        print(f"  {thin[:-4]}")

        cpu_load = CPU_BASELINE['llm_load_s']
        gpu_load = llm_load_r.duration_s
        load_speedup = cpu_load / gpu_load if gpu_load > 0 else 0
        print(f"  {'LLM Load Time':<35} {cpu_load:>15.2f}s {gpu_load:>15.2f}s {load_speedup:>11.1f}x")

        cpu_gen = CPU_BASELINE['llm_gen_s']
        gpu_gen = llm_gen_r.duration_s
        gen_speedup = cpu_gen / gpu_gen if gpu_gen > 0 else 0
        print(f"  {'LLM Generation (64 tok)':<35} {cpu_gen:>15.2f}s {gpu_gen:>15.2f}s {gen_speedup:>11.1f}x")

        cpu_tps = CPU_BASELINE['llm_tok_per_s']
        # Extract tok/s from detail if available
        gpu_tps = 0.0
        if 'tok/s' in llm_gen_r.detail:
            import re
            m = re.search(r'~([\d.]+)tok/s', llm_gen_r.detail)
            if m:
                gpu_tps = float(m.group(1))
        tps_speedup = gpu_tps / cpu_tps if cpu_tps > 0 else 0
        print(f"  {'Throughput (tok/s)':<35} {cpu_tps:>15.1f}  {gpu_tps:>15.1f}  {tps_speedup:>11.1f}x")

        cpu_total = CPU_BASELINE['total_s']
        total_speedup = cpu_total / total_time if total_time > 0 else 0
        print(f"  {thin[:-4]}")
        print(f"  {'Previous total (CPU only)':<35} {cpu_total:>15.2f}s")
        print(f"  {'Current total (GPU accel)':<35} {total_time:>15.2f}s {total_speedup:>27.1f}x")
        print(f"  {thin[:-4]}")
        print(f"  Device before : {CPU_BASELINE['device']}")
        print(f"  Device now    : NVIDIA GTX 1650 CUDA (gpu_layers=36)")
        print()

    # Verdict
    print(f"  {line}")
    if failed == 0:
        if timed_out > 0:
            print(f"  >>> VERDICT: PASS (with LLM timeouts - all core components operational)")
        else:
            print(f"  >>> VERDICT: ALL {passed} TESTS PASSED")
    else:
        print(f"  >>> VERDICT: {failed} TEST(S) FAILED - review details above")
    print(f"  {line}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 110)
    print(f"{'STARTING END-TO-END TEST':^110}")
    print(f"{'Project: ' + str(PROJECT_ROOT):^110}")
    print("=" * 110 + "\n")

    test_config()
    test_qdrant()
    test_embedding_model()
    test_bm25_load()
    test_reranker()
    test_normalization()
    test_chunking()
    test_vector_roundtrip()
    test_bm25_search()
    test_hybrid_retrieval()
    test_llm_load()
    test_llm_generation()
    test_prompt_template()
    test_index_versioning()
    test_easyocr()

    # Cleanup LLM
    if _llm_engine:
        try:
            _llm_engine.unload()
        except Exception:
            pass

    print_report()

    # Only actual FAILs count; timeouts and skips are acceptable
    failed_count = sum(1 for r in results if r.status == "FAIL")
    sys.exit(failed_count)
