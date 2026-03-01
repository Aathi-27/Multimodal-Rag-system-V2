/* ─── Upload ─────────────────────────────────────────────────────────────── */

export interface UploadResponse {
  upload_id: string;
  status: string;
  estimated_time: string;
  filename: string;
  modality: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: 'processing' | 'completed' | 'failed' | 'unknown';
  filename?: string;
  modality?: string;
  error?: string;
}

/* ─── Query / Chat ───────────────────────────────────────────────────────── */

export interface QueryFilters {
  modality?: string[];
  department?: string;
  date_range?: string[];
}

export interface QueryRequest {
  query: string;
  filters?: QueryFilters;
  max_tokens?: number;
}

/** Individual SSE event coming from the backend */
export type SSEEventType = 'status' | 'token' | 'citation' | 'done' | 'error' | 'debug' | 'meta';

export interface ConfidenceInfo {
  score: number;
  level: 'high' | 'medium' | 'low';
  signals: Record<string, number>;
  source_count: number;
  grounding: string;
}

export interface QueryCostInfo {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  retrieval_time_ms: number;
  generation_time_ms: number;
  total_time_ms: number;
  estimated_cost_usd: number;
}

export interface HallucinationInfo {
  grounded_ratio: number;
  risk_level: 'low' | 'medium' | 'high' | 'unknown';
  total_sentences: number;
  grounded_sentences: number;
  ungrounded_claims: string[];
}

export interface DebugResult {
  chunk_id: string;
  source: string;
  page: number | null;
  score: number;
  rank: number | null;
  reranker_score: number | null;
  stage: string;
  origin?: 'vector' | 'bm25' | 'both' | 'entity' | null;
  vector_rank?: number | null;
  bm25_rank?: number | null;
}

export interface EffectiveParams {
  vector_top_k: number;
  bm25_top_k: number;
  rrf_k: number;
  rerank_count: number;
  rerank_threshold: number;
  rerank_min_results: number;
}

export interface EntityInjectedChunk {
  chunk_id: string;
  entity_match: string;
  source: string;
}

export interface EntityInfo {
  extracted_entities: string[];
  injected_count: number;
  injected_chunks: EntityInjectedChunk[];
}

export interface DebugInfo {
  vector_results: DebugResult[];
  bm25_results: DebugResult[];
  rrf_fused: DebugResult[];
  reranked: DebugResult[];
  dropped: DebugResult[];
  effective_params?: EffectiveParams | null;
  entity_info?: EntityInfo | null;
  retrieval_mode?: 'text' | 'image' | 'audio' | null;
  image_branch_info?: ImageBranchInfo | null;
}

export interface SSEEvent {
  type: SSEEventType;
  content?: string;
  source?: string;
  page?: number | null;
  speaker?: string | null;
  modality?: string;
  image_path?: string | null;
  timestamp_start?: number | null;
  timestamp_end?: number | null;
  file_id?: string | null;
  // debug fields (when type === 'debug')
  vector_results?: DebugResult[];
  bm25_results?: DebugResult[];
  rrf_fused?: DebugResult[];
  reranked?: DebugResult[];
  dropped?: DebugResult[];
  effective_params?: EffectiveParams | null;
  entity_info?: EntityInfo | null;
  retrieval_mode?: 'text' | 'image' | 'audio' | null;
  image_branch_info?: ImageBranchInfo | null;
  // meta fields (when type === 'meta')
  confidence?: ConfidenceInfo;
  cost?: QueryCostInfo;
  hallucination?: HallucinationInfo;
  cached?: boolean;
}

export interface Citation {
  source: string;
  page?: number | null;
  speaker?: string | null;
  modality: string;
  image_path?: string | null;
  timestamp_start?: number | null;
  timestamp_end?: number | null;
  file_id?: string | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  isStreaming?: boolean;
  error?: string;
  timestamp: number;
  confidence?: ConfidenceInfo;
  cost?: QueryCostInfo;
  hallucination?: HallucinationInfo;
  cached?: boolean;
}

/* ─── Health ─────────────────────────────────────────────────────────────── */

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'error';
  qdrant: string;
  bm25: string;
  llm: string;
  embeddings: string;
  reranker: string;
  clip: string;
  whisper: string;
  corpus_size: number;
  uptime: string;
}

/* ─── Auth ───────────────────────────────────────────────────────────────── */

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  token: string;
}

/* ─── Index Health ────────────────────────────────────────────────────────── */

export interface IndexHealthResponse {
  total_chunks: number;
  total_documents: number;
  avg_tokens_per_chunk: number;
  largest_document: string;
  largest_document_chunks: number;
  embedding_dimension: number;
  qdrant_collection: string;
  bm25_chunk_count: number;
  bm25_vocab_size: number;
  total_tokens: number;
}

/* ─── Analytics ───────────────────────────────────────────────────────────── */

export interface DocumentAnalytics {
  source: string;
  retrieval_count: number;
  last_queried: number;
  avg_reranker_score: number;
  avg_rank_position: number;
}

/* ─── Query History ──────────────────────────────────────────────────────── */

export interface QueryHistoryItem {
  query_id: string;
  query: string;
  timestamp: number;
  chunk_count: number;
  answer: string;
  retrieval_latency: number;
  rerank_latency: number;
  generation_latency: number;
  total_latency: number;
  debug_enabled: boolean;
  error?: string | null;
  token_count: number;
}

export interface QueryHistoryResponse {
  queries: QueryHistoryItem[];
  total: number;
}

export interface QueryDetailResponse {
  query_id: string;
  query: string;
  timestamp: number;
  answer: string;
  retrieved_chunks: Array<{ chunk_id: string; source: string; score: number }>;
  chunk_count: number;
  retrieval_latency: number;
  rerank_latency: number;
  generation_latency: number;
  total_latency: number;
  debug_enabled: boolean;
  debug_info?: Record<string, unknown> | null;
  error?: string | null;
  token_count: number;
  recall_validation?: RecallValidation | null;
}

/* ─── Recall Validation ──────────────────────────────────────────────────── */

export interface ChunkRelevanceAnnotation {
  chunk_id: string;
  relevant: boolean;
}

export interface RecallValidation {
  annotations: ChunkRelevanceAnnotation[];
  recall_at_5: number | null;
  recall_at_10: number | null;
  mrr: number | null;
  relevant_count: number;
  annotated_count: number;
}

export interface RecallMetrics {
  query_id: string;
  total_retrieved: number;
  annotated_count: number;
  relevant_count: number;
  recall_at_5: number | null;
  recall_at_10: number | null;
  mrr: number | null;
  annotations: ChunkRelevanceAnnotation[];
}

export interface QuerySummaryResponse {
  total_queries: number;
  avg_latency: number;
  avg_retrieval_latency: number;
  avg_generation_latency: number;
  avg_chunks_per_query: number;
  error_count: number;
}

/* ─── Index Versions ─────────────────────────────────────────────────────── */

export interface VersionInfo {
  version: string;
  is_active: boolean;
  size_mb: number;
}

export interface VersionListResponse {
  versions: VersionInfo[];
  current_version: string | null;
}

/* ─── Metrics ────────────────────────────────────────────────────────────── */

export interface LatencyBreakdown {
  avg: number;
  p95: number;
  count: number;
}

export interface MetricsResponse {
  uploads_total: number;
  upload_errors: number;
  queries_total: number;
  query_latency: LatencyBreakdown;
  retrieval_latency: LatencyBreakdown;
  rerank_latency: LatencyBreakdown;
  generation_latency: LatencyBreakdown;
  corpus_size: number;
}

/* ─── Resources ──────────────────────────────────────────────────────────── */

export interface ResourceStatus {
  cpu_percent: number;
  ram_used_mb: number;
  ram_total_mb: number;
  ram_percent: number;
  gpu_name: string;
  gpu_memory_used_mb: number;
  gpu_memory_total_mb: number;
  gpu_memory_percent: number;
  gpu_utilization: number;
  disk_used_gb: number;
  disk_total_gb: number;
  disk_percent: number;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ─── Research Lab Types ─────────────────────────────────────────────────── */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* ─── Survival Tracking ──────────────────────────────────────────────────── */

export interface SurvivalStageInfo {
  present: boolean;
  rank?: number | null;
  score?: number | null;
  origin?: string | null;
  vector_rank?: number | null;
  bm25_rank?: number | null;
  entity_match?: string | null;
}

export interface ChunkSurvivalEntry {
  chunk_id: string;
  source: string;
  stages: Record<string, SurvivalStageInfo>;
  survived: boolean;
  dropped_at?: string | null;
  dropped_reason?: string | null;
}

export interface SurvivalSummary {
  total_chunks_seen: number;
  survived_count: number;
  dropped_count: number;
  drop_reasons: Record<string, number>;
  stage_counts: Record<string, number>;
  survival_rates: Record<string, number>;
}

export interface SurvivalResponse {
  query_id: string;
  query: string;
  survival_log: ChunkSurvivalEntry[];
  summary?: SurvivalSummary | null;
}

/* ─── Failure Diagnosis ──────────────────────────────────────────────────── */

export interface DiagnosisResponse {
  query_id: string;
  root_cause: string;
  confidence: number;
  evidence: string[];
  recommendations: string[];
  secondary_causes: string[];
}

export interface BatchDiagnosisResponse {
  total_analyzed: number;
  cause_distribution: Record<string, number>;
  avg_confidence: number;
  common_recommendations: Array<{ recommendation: string; frequency: number }>;
  diagnoses: DiagnosisResponse[];
}

/* ─── Experiments ────────────────────────────────────────────────────────── */

export interface ExperimentParams {
  vector_top_k?: number | null;
  bm25_top_k?: number | null;
  rrf_k?: number | null;
  rerank_count?: number | null;
  rerank_threshold?: number | null;
  rerank_min_results?: number | null;
}

export interface ExperimentChunk {
  chunk_id: string;
  source: string;
  page_start?: number | null;
  reranker_score: number;
  rank: number;
  text_preview: string;
}

export interface ExperimentResult {
  experiment_id: string;
  query: string;
  params: ExperimentParams;
  chunks: ExperimentChunk[];
  chunk_count: number;
  retrieval_latency: number;
  survival_log: ChunkSurvivalEntry[];
}

export interface CompareAnalysis {
  overlap_count: number;
  unique_to_a: number;
  unique_to_b: number;
  jaccard_similarity: number;
  rank_differences: Array<{ chunk_id: string; rank_a: number; rank_b: number; rank_diff: number }>;
  score_comparison: Array<{ chunk_id: string; score_a: number; score_b: number; score_diff: number }>;
  latency_diff: number;
}

export interface CompareResponse {
  query: string;
  experiment_a: { label: string; params: ExperimentParams; result: ExperimentResult };
  experiment_b: { label: string; params: ExperimentParams; result: ExperimentResult };
  analysis: CompareAnalysis;
}

export interface BatchEvalMetrics {
  avg_recall_at_5: number | null;
  avg_recall_at_10: number | null;
  avg_mrr: number | null;
  avg_latency: number;
  p95_latency: number;
  min_latency: number;
  max_latency: number;
  avg_chunks_returned: number;
}

export interface BatchEvalResponse {
  evaluation_id: string;
  total_queries: number;
  params: ExperimentParams;
  metrics: BatchEvalMetrics;
  per_query: Array<Record<string, unknown>>;
}

/* ─── Corpus Coverage ────────────────────────────────────────────────────── */

export interface SourceCoverage {
  source: string;
  total_chunks: number;
  retrieved_chunks: number;
  coverage_rate: number;
  never_retrieved: number;
}

export interface CorpusCoverageResponse {
  total_corpus_chunks: number;
  total_queries_analyzed: number;
  coverage_rate: number;
  retrieved_chunk_count: number;
  never_retrieved_count: number;
  never_retrieved_sample: string[];
  hotspot_chunks: Array<{ chunk_id: string; retrieval_count: number; avg_rank: number; avg_score: number }>;
  coldspot_chunks: Array<{ chunk_id: string; retrieval_count: number; avg_rank: number; avg_score: number }>;
  per_source_coverage: SourceCoverage[];
}

/* ─── Embedding Quality ──────────────────────────────────────────────────── */

export interface NormStats {
  avg: number;
  std: number;
  min: number;
  max: number;
  outlier_count: number;
}

export interface CosineStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  p25: number;
  p75: number;
  pairs_analyzed: number;
}

export interface EmbeddingQualityResponse {
  health_status: 'good' | 'warning' | 'critical';
  issues: string[];
  total_vectors: number;
  sample_analyzed: number;
  embedding_dim?: number | null;
  norm_stats?: NormStats | null;
  cosine_stats?: CosineStats | null;
  per_source: Array<{ source: string; count: number; avg_norm: number; norm_std: number }>;
}

/* ─── Version Metadata ───────────────────────────────────────────────────── */

export interface VersionDetailResponse {
  version: string;
  is_active: boolean;
  size_mb: number;
  metadata: Record<string, unknown>;
}

/* ─── Image / Audio Query ────────────────────────────────────────────────── */

export interface ImageQueryResultItem {
  chunk_id: string;
  source: string;
  page?: number | null;
  score: number;
  origin: 'clip_image' | 'linked_expansion' | 'text_assisted';
  clip_score?: number | null;
  reranker_score?: number | null;
  text_preview?: string;
}

export interface ModalityContribution {
  image_results: number;
  linked_text_results: number;
  text_assisted_results: number;
}

export interface LatencySplit {
  image_embedding_ms: number;
  image_search_ms: number;
  linked_expansion_ms: number;
  text_search_ms?: number;
  rerank_ms: number;
  total_ms: number;
}

export interface ImageQueryResponse {
  query_id: string;
  retrieval_mode: 'image';
  results: ImageQueryResultItem[];
  result_count: number;
  modality_contribution: ModalityContribution;
  latency_split: LatencySplit;
  ocr_text?: string | null;
  answer?: string | null;
  citations?: Array<{ source: string; page?: number | null; modality?: string }>;
  debug_info?: Record<string, unknown> | null;
}

export interface AudioQueryResponse {
  query_id: string;
  retrieval_mode: 'audio';
  transcript: string;
  answer: string;
  citations: Citation[];
  transcription_latency_ms: number;
  total_latency_ms: number;
}

/* ─── Image Branch Info (debug SSE) ──────────────────────────────────────── */

export interface ImageBranchInfo {
  image_chunk_count: number;
  linked_text_count: number;
  image_branch_contribution: number;
}
