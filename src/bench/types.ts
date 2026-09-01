/**
 * Types for the QMD benchmark harness.
 *
 * A benchmark fixture defines queries with expected results.
 * The harness runs each query through multiple search backends
 * and measures precision, recall, MRR, and latency.
 */

export interface BenchmarkQuery {
  /** Unique identifier for the query */
  id: string;
  /** The search query text */
  query: string;
  /** Query difficulty/type for grouping results */
  type: "exact" | "semantic" | "topical" | "cross-domain" | "alias";
  /** Human-readable description of what this tests */
  description: string;
  /** File paths (relative to collection) that should appear in results */
  expected_files: string[];
  /** How many of expected_files should appear in top-k results */
  expected_in_top_k: number;
}

export interface BenchmarkFixture {
  /** Description of the benchmark */
  description: string;
  /** Fixture format version */
  version: number;
  /** Optional collection to search within */
  collection?: string;
  /** The test queries */
  queries: BenchmarkQuery[];
}

export interface BackendResult {
  /** Whether the backend returned a result set that can be scored. */
  status: "measured" | "not_measured";
  /** Present when status is not_measured; never interpret placeholder zeros as scores. */
  not_measured_reason?: string;
  /** Fraction of top-k results that are relevant */
  precision_at_k: number;
  /** Fraction of expected files found anywhere in results */
  recall: number;
  /** Fraction of expected files found in the first result */
  recall_at_1: number;
  /** Fraction of expected files found in the top 3 results */
  recall_at_3: number;
  /** Fraction of expected files found in the top 5 results */
  recall_at_5: number;
  recall_at_10: number;
  recall_at_40: number;
  recall_at_100: number;
  /** Binary-relevance normalized discounted cumulative gain at 10. */
  ndcg_at_10: number;
  /** Reciprocal rank of first relevant result (1/rank, 0 if not found) */
  mrr: number;
  /** Harmonic mean of precision_at_k and recall */
  f1: number;
  /** Number of expected files found in top-k */
  hits_at_k: number;
  /** Total expected files */
  total_expected: number;
  /** Wall-clock latency in milliseconds */
  latency_ms: number;
  /** Top result file paths (for inspection) */
  top_files: string[];
  /** Expected files that were found anywhere in the returned result set */
  matched_files: string[];
  /** Expected files missing from the returned result set */
  unmatched_expected_files: string[];
}

export interface QueryResult {
  id: string;
  query: string;
  type: string;
  backends: Record<string, BackendResult>;
}

export interface BenchmarkResult {
  timestamp: string;
  fixture: string;
  /** SHA-256 of exact fixture bytes: freezes relevance input for comparisons. */
  fixture_sha256: string;
  environment: {
    node: string;
    platform: string;
    arch: string;
    /** Main SQLite database plus any live WAL/journal sidecars. */
    index_size_bytes?: number;
    index_files?: Record<string, number>;
    peak_rss_bytes: number;
    embedding_provider: "local" | "openai";
    embedding_identity: import("../store.js").EmbeddingIdentity;
    embedding_fingerprint: string;
  };
  results: QueryResult[];
  summary: Record<string, {
    avg_precision: number;
    avg_recall: number;
    avg_recall_at_1: number;
    avg_recall_at_3: number;
    avg_recall_at_5: number;
    avg_recall_at_10: number;
    avg_recall_at_40: number;
    avg_recall_at_100: number;
    avg_ndcg_at_10: number;
    avg_mrr: number;
    avg_f1: number;
    avg_latency_ms: number;
    search_queries_per_second: number;
  }>;
}
