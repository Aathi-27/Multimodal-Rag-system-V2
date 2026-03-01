"""
Pydantic schemas for API request/response models.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadMetadata(BaseModel):
    department: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    upload_date: Optional[str] = None


class UploadResponse(BaseModel):
    upload_id: str
    status: str = "processing"
    estimated_time: str = "30s"
    filename: str = ""
    modality: str = ""


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryFilters(BaseModel):
    modality: Optional[list[str]] = None       # ["document", "image", "audio"]
    department: Optional[str] = None
    date_range: Optional[list[str]] = None     # ["2025-01-01", "2025-12-31"]


class QueryRequest(BaseModel):
    query: str
    filters: Optional[QueryFilters] = None
    max_tokens: int = 512
    debug: bool = False            # When True, return retrieval debug info via SSE


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    speaker: Optional[str] = None
    timestamp: Optional[str] = None
    modality: str = "document"


class QueryResponse(BaseModel):
    """Non-streaming query response."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    query_id: str = ""


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    qdrant: str = "unknown"
    bm25: str = "unknown"
    llm: str = "unknown"
    embeddings: str = "unknown"
    reranker: str = "unknown"
    clip: str = "unknown"
    whisper: str = "unknown"
    corpus_size: int = 0
    uptime: str = ""


# ── Status ────────────────────────────────────────────────────────────────────

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    filename: Optional[str] = None
    modality: Optional[str] = None
    error: Optional[str] = None


# ── Knowledge Base ────────────────────────────────────────────────────────────

class DocumentSummary(BaseModel):
    """Aggregated metadata for a single source document."""
    source: str
    modality: str = "document"
    department: str = ""
    tags: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    total_tokens: int = 0
    upload_id: str = ""


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary] = Field(default_factory=list)
    total: int = 0


class ChunkDetail(BaseModel):
    """Single chunk within a document."""
    chunk_id: str
    text: str
    chunk_index: int = 0
    page_start: Optional[int] = None
    token_count: int = 0


class DocumentChunksResponse(BaseModel):
    source: str
    chunks: list[ChunkDetail] = Field(default_factory=list)
    total_chunks: int = 0


class DeleteDocumentResponse(BaseModel):
    source: str
    deleted_chunks: int = 0
    status: str = "deleted"


# ── Index Health ──────────────────────────────────────────────────────────────

class IndexHealthResponse(BaseModel):
    total_chunks: int = 0
    total_documents: int = 0
    avg_tokens_per_chunk: float = 0.0
    largest_document: str = ""
    largest_document_chunks: int = 0
    embedding_dimension: int = 384
    qdrant_collection: str = ""
    bm25_chunk_count: int = 0
    bm25_vocab_size: int = 0
    total_tokens: int = 0


# ── Analytics ─────────────────────────────────────────────────────────────────

class DocumentAnalytics(BaseModel):
    source: str
    retrieval_count: int = 0
    last_queried: float = 0           # Unix timestamp
    avg_reranker_score: float = 0.0
    avg_rank_position: float = 0.0


class AnalyticsResponse(BaseModel):
    documents: list[DocumentAnalytics] = Field(default_factory=list)


# ── Re-Index ──────────────────────────────────────────────────────────────────

class ReindexResponse(BaseModel):
    source: str
    status: str = "reindexing"
    message: str = ""


# ── Query History ─────────────────────────────────────────────────────────────

class QueryHistoryItem(BaseModel):
    query_id: str
    query: str
    timestamp: float = 0.0
    chunk_count: int = 0
    answer: str = ""
    retrieval_latency: float = 0.0
    rerank_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0
    debug_enabled: bool = False
    error: Optional[str] = None
    token_count: int = 0


class QueryHistoryResponse(BaseModel):
    queries: list[QueryHistoryItem] = Field(default_factory=list)
    total: int = 0


class QueryDetailResponse(BaseModel):
    query_id: str
    query: str
    timestamp: float = 0.0
    answer: str = ""
    retrieved_chunks: list[dict] = Field(default_factory=list)
    chunk_count: int = 0
    retrieval_latency: float = 0.0
    rerank_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0
    debug_enabled: bool = False
    debug_info: Optional[dict] = None
    error: Optional[str] = None
    token_count: int = 0


class QuerySummaryResponse(BaseModel):
    total_queries: int = 0
    avg_latency: float = 0.0
    avg_retrieval_latency: float = 0.0
    avg_generation_latency: float = 0.0
    avg_chunks_per_query: float = 0.0
    error_count: int = 0


# ── Index Versions ────────────────────────────────────────────────────────────

class VersionInfo(BaseModel):
    version: str
    is_active: bool = False
    size_mb: float = 0.0


class VersionListResponse(BaseModel):
    versions: list[VersionInfo] = Field(default_factory=list)
    current_version: Optional[str] = None


class VersionSwitchResponse(BaseModel):
    version: str
    status: str = "switched"
    message: str = ""


# ── Metrics / Resources ──────────────────────────────────────────────────────

class LatencyBreakdown(BaseModel):
    avg: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    count: int = 0


class MetricsResponse(BaseModel):
    uploads_total: int = 0
    upload_errors: int = 0
    queries_total: int = 0
    query_latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    retrieval_latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    rerank_latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    generation_latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    corpus_size: int = 0


class ResourceStatus(BaseModel):
    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_percent: float = 0.0
    gpu_name: str = ""
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_percent: float = 0.0
    status: str = "reindexing"
    message: str = ""


# ── Settings ──────────────────────────────────────────────────────────────────

class RetrievalSettingsRequest(BaseModel):
    """PATCH body for runtime retrieval parameter overrides."""
    vector_top_k: Optional[int] = None
    bm25_top_k: Optional[int] = None
    rrf_k: Optional[int] = None
    rerank_count: Optional[int] = None
    rerank_threshold: Optional[float] = None
    rerank_min_results: Optional[int] = None


class RetrievalSettingsResponse(BaseModel):
    """Current effective retrieval settings."""
    vector_top_k: int = 50
    bm25_top_k: int = 50
    rrf_k: int = 60
    rerank_count: int = 15
    rerank_threshold: float = 0.15
    rerank_min_results: int = 5
    overrides: dict = Field(default_factory=dict)


# ── Recall Validation ─────────────────────────────────────────────────────────

class ChunkRelevanceAnnotation(BaseModel):
    """Single chunk relevance annotation."""
    chunk_id: str
    relevant: bool


class RecallValidationRequest(BaseModel):
    """POST body for recall validation annotations."""
    annotations: list[ChunkRelevanceAnnotation] = Field(default_factory=list)


class RecallMetrics(BaseModel):
    """Computed recall metrics for a query."""
    query_id: str
    total_retrieved: int = 0
    annotated_count: int = 0
    relevant_count: int = 0
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    mrr: Optional[float] = None
    annotations: list[ChunkRelevanceAnnotation] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# ── Research Lab Models ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


# ── Survival Tracking ─────────────────────────────────────────────────────────

class SurvivalStageInfo(BaseModel):
    """Presence info for a single pipeline stage."""
    present: bool = False
    rank: Optional[int] = None
    score: Optional[float] = None
    origin: Optional[str] = None
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    entity_match: Optional[str] = None


class ChunkSurvivalEntry(BaseModel):
    """Per-chunk survival record across all pipeline stages."""
    chunk_id: str
    source: str = "unknown"
    stages: dict[str, SurvivalStageInfo] = Field(default_factory=dict)
    survived: bool = False
    dropped_at: Optional[str] = None
    dropped_reason: Optional[str] = None


class SurvivalSummary(BaseModel):
    """Aggregate survival statistics for a query."""
    total_chunks_seen: int = 0
    survived_count: int = 0
    dropped_count: int = 0
    drop_reasons: dict[str, int] = Field(default_factory=dict)
    stage_counts: dict[str, int] = Field(default_factory=dict)
    survival_rates: dict[str, float] = Field(default_factory=dict)


class SurvivalResponse(BaseModel):
    """Full survival analysis for a query."""
    query_id: str
    query: str = ""
    survival_log: list[dict] = Field(default_factory=list)
    summary: Optional[dict] = None


# ── Failure Diagnosis ─────────────────────────────────────────────────────────

class DiagnosisResponse(BaseModel):
    """Root-cause diagnosis for a query."""
    query_id: str
    root_cause: str
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    secondary_causes: list[str] = Field(default_factory=list)


class BatchDiagnosisResponse(BaseModel):
    """Aggregate diagnosis across multiple queries."""
    total_analyzed: int = 0
    cause_distribution: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    common_recommendations: list[dict] = Field(default_factory=list)
    diagnoses: list[dict] = Field(default_factory=list)


# ── Ground Truth ──────────────────────────────────────────────────────────────

class GroundTruthRequest(BaseModel):
    """Tag a query with ground truth chunk IDs for evaluation."""
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)


# ── Experiment Comparison ─────────────────────────────────────────────────────

class ExperimentParams(BaseModel):
    """Retrieval parameter set for an experiment."""
    vector_top_k: Optional[int] = None
    bm25_top_k: Optional[int] = None
    rrf_k: Optional[int] = None
    rerank_count: Optional[int] = None
    rerank_threshold: Optional[float] = None
    rerank_min_results: Optional[int] = None


class CompareRequest(BaseModel):
    """Side-by-side comparison request."""
    query: str
    params_a: ExperimentParams
    params_b: ExperimentParams
    label_a: str = "A"
    label_b: str = "B"


class BatchEvalQuery(BaseModel):
    """Single query in a batch evaluation."""
    query: str
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)
    expected_source: Optional[str] = None


class BatchEvalRequest(BaseModel):
    """Batch evaluation request."""
    queries: list[BatchEvalQuery]
    overrides: Optional[ExperimentParams] = None


# ── Corpus Coverage ───────────────────────────────────────────────────────────

class SourceCoverage(BaseModel):
    """Coverage stats for a single document source."""
    source: str
    total_chunks: int = 0
    retrieved_chunks: int = 0
    coverage_rate: float = 0.0
    never_retrieved: int = 0


class CorpusCoverageResponse(BaseModel):
    """Full corpus coverage analysis."""
    total_corpus_chunks: int = 0
    total_queries_analyzed: int = 0
    coverage_rate: float = 0.0
    retrieved_chunk_count: int = 0
    never_retrieved_count: int = 0
    never_retrieved_sample: list[str] = Field(default_factory=list)
    hotspot_chunks: list[dict] = Field(default_factory=list)
    coldspot_chunks: list[dict] = Field(default_factory=list)
    per_source_coverage: list[dict] = Field(default_factory=list)


# ── Embedding Quality ─────────────────────────────────────────────────────────

class EmbeddingQualityResponse(BaseModel):
    """Embedding quality diagnostics."""
    health_status: str = "unknown"
    issues: list[str] = Field(default_factory=list)
    total_vectors: int = 0
    sample_analyzed: int = 0
    embedding_dim: Optional[int] = None
    norm_stats: Optional[dict] = None
    cosine_stats: Optional[dict] = None
    per_source: list[dict] = Field(default_factory=list)


# ── Version Metadata ──────────────────────────────────────────────────────────

class VersionMetadataRequest(BaseModel):
    """Metadata to associate with an index version."""
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None
    rerank_threshold: Optional[float] = None
    rrf_k: Optional[int] = None
    entity_limit: Optional[int] = None
    description: Optional[str] = None
    chunk_count: Optional[int] = None


class VersionDetailResponse(BaseModel):
    """Version info with full metadata."""
    version: str
    is_active: bool = False
    size_mb: float = 0.0
    metadata: dict = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# ── Phase 4: Image-as-Query / Audio-as-Query Models ──────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


class ImageQueryResponse(BaseModel):
    """Response for POST /query/image — image-as-query retrieval."""
    query_id: str = ""
    retrieval_mode: str = "image"
    results: list[dict] = Field(default_factory=list)
    result_count: int = 0
    modality_contribution: dict = Field(default_factory=dict)
    latency_split: dict = Field(default_factory=dict)
    ocr_text: Optional[str] = None
    answer: Optional[str] = None
    citations: list[dict] = Field(default_factory=list)
    debug_info: Optional[dict] = None


class AudioQueryResponse(BaseModel):
    """Response for POST /query/audio — audio-as-query (Whisper→text)."""
    query_id: str = ""
    retrieval_mode: str = "audio"
    transcript: str = ""
    answer: str = ""
    citations: list[dict] = Field(default_factory=list)
    transcription_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
