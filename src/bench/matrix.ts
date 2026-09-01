import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { relative, resolve } from "node:path";
import Database from "better-sqlite3";
import YAML from "yaml";
import { runBenchmark } from "./bench.js";
import type { BenchmarkResult } from "./types.js";

export type BenchmarkMode = "vector-only" | "hybrid-no-rerank" | "full";

export interface FrozenModel {
  id: string;
  provider: "local" | "openai";
  model: string;
  identity_reference: string;
  revision: string;
  artifact_sha256?: string;
  artifact_size_bytes?: number;
  control?: boolean;
  full_supported?: boolean;
  full_not_measured_reason?: string;
}

export interface FrozenMatrixManifest {
  version: 1;
  name: string;
  frozen_at: string;
  fixture: { path: string; sha256: string };
  corpus: { path: string; sha256_tree_v1: string };
  retrieval: { cutoffs: [10, 40, 100]; max_results: 100; modes: BenchmarkMode[] };
  full_pipeline: {
    generator: { model: string; revision: string; artifact_sha256: string };
    reranker: { model: string; revision: string; artifact_sha256: string };
  };
  models: FrozenModel[];
}

export interface LocalModelRun {
  db_path?: string;
  config_path?: string;
  artifact_path?: string;
  not_measured_reason?: string;
  mode_not_measured?: Partial<Record<BenchmarkMode, string>>;
  indexing?: {
    duration_ms: number;
    documents: number;
    chunks: number;
    peak_rss_bytes?: number;
  };
  full_pipeline?: {
    generator_artifact_path: string;
    reranker_artifact_path: string;
  };
}

export interface LocalRunSpec {
  version: 1;
  runs: Record<string, LocalModelRun>;
}

export interface MatrixCell {
  model_id: string;
  provider: "local" | "openai";
  embedding_model: string;
  embedding_revision: string;
  mode: BenchmarkMode;
  status: "measured" | "not_measured";
  not_measured_reason?: string;
  metrics?: {
    recall_at_10: number;
    recall_at_40: number;
    recall_at_100: number;
    mrr: number;
    ndcg_at_10: number;
    avg_latency_ms: number;
    search_queries_per_second: number;
  };
  peak_rss_bytes?: number;
  index_size_bytes?: number;
  indexing_throughput?: {
    documents_per_second: number;
    chunks_per_second: number;
    duration_ms: number;
    peak_rss_bytes?: number;
  };
  indexing_not_measured_reason?: string;
  embedding_identity?: BenchmarkResult["environment"]["embedding_identity"];
  embedding_fingerprint?: string;
}

export interface MatrixResult {
  schema: 1;
  manifest_name: string;
  manifest_sha256: string;
  fixture_sha256: string;
  corpus_sha256_tree_v1: string;
  generated_at: string;
  cells: MatrixCell[];
  notes: string[];
}

export function sha256File(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

/** Hash sorted relative paths and file hashes, so corpus identity is host-independent. */
export function sha256Tree(root: string): string {
  const absoluteRoot = resolve(root);
  const files: string[] = [];
  const visit = (directory: string): void => {
    for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = resolve(directory, entry.name);
      if (entry.isDirectory()) visit(path);
      else if (entry.isFile()) files.push(path);
    }
  };
  visit(absoluteRoot);
  const hash = createHash("sha256");
  for (const path of files) {
    hash.update(relative(absoluteRoot, path).replaceAll("\\", "/"));
    hash.update("\0");
    hash.update(sha256File(path));
    hash.update("\n");
  }
  return hash.digest("hex");
}

export function loadFrozenManifest(path: string): FrozenMatrixManifest {
  const manifest = JSON.parse(readFileSync(path, "utf8")) as FrozenMatrixManifest;
  if (manifest.version !== 1 || !manifest.models?.length) throw new Error("Unsupported or empty matrix manifest");
  if (JSON.stringify(manifest.retrieval.cutoffs) !== "[10,40,100]" || manifest.retrieval.max_results !== 100) {
    throw new Error("Frozen matrix must use Recall cutoffs 10/40/100 and max_results 100");
  }
  return manifest;
}

export function verifyFrozenInputs(manifest: FrozenMatrixManifest, baseDir: string): void {
  const fixturePath = resolve(baseDir, manifest.fixture.path);
  const corpusPath = resolve(baseDir, manifest.corpus.path);
  if (!existsSync(fixturePath)) throw new Error(`Frozen fixture is missing: ${fixturePath}`);
  if (!existsSync(corpusPath)) throw new Error(`Frozen corpus is missing: ${corpusPath}`);
  const fixtureHash = sha256File(fixturePath);
  if (fixtureHash !== manifest.fixture.sha256) throw new Error(`Fixture hash mismatch: expected ${manifest.fixture.sha256}, got ${fixtureHash}`);
  const corpusHash = sha256Tree(corpusPath);
  if (corpusHash !== manifest.corpus.sha256_tree_v1) throw new Error(`Corpus hash mismatch: expected ${manifest.corpus.sha256_tree_v1}, got ${corpusHash}`);
}

function verifyPreparedInputs(
  manifest: FrozenMatrixManifest,
  baseDir: string,
  model: FrozenModel,
  run: LocalModelRun,
): string | undefined {
  try {
    const config = YAML.parse(readFileSync(run.config_path!, "utf8")) as {
    collections?: Record<string, { path?: string }>;
    models?: { embed?: string; generate?: string; rerank?: string };
    embedding?: { provider?: string; identity?: { revision?: string }; openai?: { model?: string } };
  };
    const fixture = JSON.parse(readFileSync(resolve(baseDir, manifest.fixture.path), "utf8")) as { collection?: string };
    const collection = fixture.collection;
    if (!collection) return "frozen fixture does not declare a collection";
    const configuredCorpus = config.collections?.[collection]?.path;
    const frozenCorpus = resolve(baseDir, manifest.corpus.path);
    if (!configuredCorpus || resolve(configuredCorpus) !== frozenCorpus) {
      return `config collection ${collection} must point to frozen corpus ${frozenCorpus}`;
    }
    const provider = config.embedding?.provider ?? "local";
    if (provider !== model.provider) return `config embedding provider mismatch: expected ${model.provider}, got ${provider}`;
    if (config.embedding?.identity?.revision !== model.revision) {
      return `config embedding revision mismatch: expected ${model.revision}, got ${config.embedding?.identity?.revision ?? "unspecified"}`;
    }
    if (model.provider === "local" && resolve(config.models?.embed ?? "") !== resolve(run.artifact_path!)) {
      return "config models.embed must be the exact local artifact_path";
    }
    if (model.provider === "openai" && config.embedding?.openai?.model !== model.model) {
      return `config OpenAI model mismatch: expected ${model.model}, got ${config.embedding?.openai?.model ?? "unspecified"}`;
    }

    const db = new Database(run.db_path!, { readonly: true, fileMustExist: true });
    try {
      const rows = db.prepare("SELECT path, hash FROM documents WHERE collection = ? AND active = 1 ORDER BY path").all(collection) as Array<{ path: string; hash: string }>;
      const corpusFiles = readdirSync(frozenCorpus, { withFileTypes: true })
        .filter(entry => entry.isFile())
        .map(entry => ({ path: entry.name, hash: sha256File(resolve(frozenCorpus, entry.name)) }))
        .sort((a, b) => a.path.localeCompare(b.path));
      if (JSON.stringify(rows) !== JSON.stringify(corpusFiles)) {
        return "prepared index active document paths/content hashes do not match the frozen corpus";
      }
    } finally {
      db.close();
    }
  } catch (error) {
    return `prepared index could not be verified: ${error instanceof Error ? error.message : String(error)}`;
  }
  return undefined;
}

function verifyVectorCoverage(dbPath: string, collection: string, fingerprint: string): string | undefined {
  try {
    const db = new Database(dbPath, { readonly: true, fileMustExist: true });
    try {
      const expected = db.prepare("SELECT COUNT(DISTINCT hash) AS count FROM documents WHERE collection = ? AND active = 1").get(collection) as { count: number };
      const embedded = db.prepare(`
        SELECT COUNT(DISTINCT d.hash) AS count
        FROM documents d
        JOIN content_vectors v ON v.hash = d.hash
        WHERE d.collection = ? AND d.active = 1 AND v.embed_fingerprint = ?
      `).get(collection, fingerprint) as { count: number };
      if (embedded.count !== expected.count) {
        return `prepared index has active vectors for ${embedded.count}/${expected.count} frozen documents under embedding fingerprint ${fingerprint}`;
      }
    } finally {
      db.close();
    }
  } catch (error) {
    return `prepared vector coverage could not be verified: ${error instanceof Error ? error.message : String(error)}`;
  }
  return undefined;
}

function indexingFields(run: LocalModelRun): Pick<MatrixCell, "indexing_throughput" | "indexing_not_measured_reason"> {
  const indexing = run.indexing;
  if (!indexing) return { indexing_not_measured_reason: "indexing counters were not supplied in the local run spec" };
  if (!(indexing.duration_ms > 0)) return { indexing_not_measured_reason: "indexing duration_ms must be greater than zero" };
  const seconds = indexing.duration_ms / 1000;
  return {
    indexing_throughput: {
      documents_per_second: indexing.documents / seconds,
      chunks_per_second: indexing.chunks / seconds,
      duration_ms: indexing.duration_ms,
      ...(indexing.peak_rss_bytes === undefined ? {} : { peak_rss_bytes: indexing.peak_rss_bytes }),
    },
  };
}

function unavailableCells(model: FrozenModel, modes: BenchmarkMode[], reason: string, run: LocalModelRun = {}): MatrixCell[] {
  return modes.map(mode => ({
    model_id: model.id,
    provider: model.provider,
    embedding_model: model.model,
    embedding_revision: model.revision,
    mode,
    status: "not_measured",
    not_measured_reason: run.mode_not_measured?.[mode] ?? reason,
    ...indexingFields(run),
  }));
}

function fullModeUnavailableReason(manifest: FrozenMatrixManifest, model: FrozenModel, run: LocalModelRun): string | undefined {
  if (model.full_supported === false) return model.full_not_measured_reason ?? "the frozen manifest does not support full mode for this provider";
  const paths = run.full_pipeline;
  if (!paths) return "full pipeline artifact paths were not supplied in the local run spec";
  if (!existsSync(paths.generator_artifact_path)) return `full pipeline generator is missing: ${paths.generator_artifact_path}`;
  if (!existsSync(paths.reranker_artifact_path)) return `full pipeline reranker is missing: ${paths.reranker_artifact_path}`;
  let config: { models?: { generate?: string; rerank?: string } };
  try {
    config = YAML.parse(readFileSync(run.config_path!, "utf8")) as typeof config;
  } catch (error) {
    return `full pipeline config could not be read: ${error instanceof Error ? error.message : String(error)}`;
  }
  if (resolve(config.models?.generate ?? "") !== resolve(paths.generator_artifact_path)) {
    return "config models.generate must be the exact full_pipeline generator_artifact_path";
  }
  if (resolve(config.models?.rerank ?? "") !== resolve(paths.reranker_artifact_path)) {
    return "config models.rerank must be the exact full_pipeline reranker_artifact_path";
  }
  const generatorHash = sha256File(paths.generator_artifact_path);
  if (generatorHash !== manifest.full_pipeline.generator.artifact_sha256) {
    return `generator SHA-256 mismatch: expected ${manifest.full_pipeline.generator.artifact_sha256}, got ${generatorHash}`;
  }
  const rerankerHash = sha256File(paths.reranker_artifact_path);
  if (rerankerHash !== manifest.full_pipeline.reranker.artifact_sha256) {
    return `reranker SHA-256 mismatch: expected ${manifest.full_pipeline.reranker.artifact_sha256}, got ${rerankerHash}`;
  }
  return undefined;
}

const BACKEND_BY_MODE: Record<BenchmarkMode, string> = {
  "vector-only": "vector",
  "hybrid-no-rerank": "hybrid-no-rerank",
  full: "full",
};

export async function runFrozenMatrix(
  manifestPath: string,
  localSpec?: LocalRunSpec,
): Promise<MatrixResult> {
  const absoluteManifest = resolve(manifestPath);
  const baseDir = resolve(absoluteManifest, "..");
  const manifest = loadFrozenManifest(absoluteManifest);
  if (localSpec && localSpec.version !== 1) throw new Error(`Unsupported local run spec version: ${localSpec.version}`);
  verifyFrozenInputs(manifest, baseDir);
  const fixturePath = resolve(baseDir, manifest.fixture.path);
  const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as { collection: string };
  const cells: MatrixCell[] = [];

  for (const model of manifest.models) {
    const run = localSpec?.runs[model.id];
    if (!run) {
      cells.push(...unavailableCells(model, manifest.retrieval.modes, "no local run spec was supplied for this model"));
      continue;
    }
    if (run.not_measured_reason) {
      cells.push(...unavailableCells(model, manifest.retrieval.modes, run.not_measured_reason, run));
      continue;
    }
    if (!run.db_path || !existsSync(run.db_path)) {
      cells.push(...unavailableCells(model, manifest.retrieval.modes, `prepared index is missing: ${run.db_path ?? "db_path not supplied"}`, run));
      continue;
    }
    if (!run.config_path || !existsSync(run.config_path)) {
      cells.push(...unavailableCells(model, manifest.retrieval.modes, `frozen config is missing: ${run.config_path ?? "config_path not supplied"}`, run));
      continue;
    }
    if (model.provider === "local") {
      if (!run.artifact_path || !existsSync(run.artifact_path)) {
        cells.push(...unavailableCells(model, manifest.retrieval.modes, `model artifact is missing: ${run.artifact_path ?? "artifact_path not supplied"}`, run));
        continue;
      }
      const artifactHash = sha256File(run.artifact_path);
      if (artifactHash !== model.artifact_sha256) {
        cells.push(...unavailableCells(model, manifest.retrieval.modes, `model artifact SHA-256 mismatch: expected ${model.artifact_sha256}, got ${artifactHash}`, run));
        continue;
      }
    }
    const preparedInputReason = verifyPreparedInputs(manifest, baseDir, model, run);
    if (preparedInputReason) {
      cells.push(...unavailableCells(model, manifest.retrieval.modes, preparedInputReason, run));
      continue;
    }

    for (const mode of manifest.retrieval.modes) {
      const forcedReason = run.mode_not_measured?.[mode]
        ?? (mode === "full" ? fullModeUnavailableReason(manifest, model, run) : undefined);
      if (forcedReason) {
        cells.push(...unavailableCells(model, [mode], forcedReason, run));
        continue;
      }
      let result: BenchmarkResult;
      try {
        result = await runBenchmark(fixturePath, {
          quiet: true,
          dbPath: run.db_path,
          configPath: run.config_path,
          backends: [BACKEND_BY_MODE[mode]],
          maxResults: manifest.retrieval.max_results,
        });
      } catch (error) {
        cells.push(...unavailableCells(model, [mode], `benchmark could not start or close cleanly: ${error instanceof Error ? error.message : String(error)}`, run));
        continue;
      }
      const backend = BACKEND_BY_MODE[mode];
      const summary = result.summary[backend];
      const failures = result.results.flatMap(row => {
        const value = row.backends[backend];
        return value?.status === "not_measured" ? [value.not_measured_reason ?? "backend failed"] : [];
      });
      const base: MatrixCell = {
        model_id: model.id,
        provider: model.provider,
        embedding_model: model.model,
        embedding_revision: model.revision,
        mode,
        status: summary && failures.length === 0 ? "measured" : "not_measured",
        ...indexingFields(run),
        peak_rss_bytes: result.environment.peak_rss_bytes,
        index_size_bytes: result.environment.index_size_bytes,
        embedding_identity: result.environment.embedding_identity,
        embedding_fingerprint: result.environment.embedding_fingerprint,
      };
      const identityMismatch = result.environment.embedding_provider !== model.provider
        || result.environment.embedding_identity.model.reference !== model.identity_reference
        || result.environment.embedding_identity.model.revision !== model.revision;
      if (identityMismatch) {
        base.status = "not_measured";
        base.not_measured_reason = `embedding identity mismatch: expected ${model.provider}/${model.identity_reference}@${model.revision}, got ${result.environment.embedding_provider}/${result.environment.embedding_identity.model.reference}@${result.environment.embedding_identity.model.revision}`;
      } else {
        const coverageReason = verifyVectorCoverage(run.db_path, fixture.collection, result.environment.embedding_fingerprint);
        if (coverageReason) {
          base.status = "not_measured";
          base.not_measured_reason = coverageReason;
        } else if (!summary || failures.length > 0) {
          base.not_measured_reason = [...new Set(failures)].join("; ") || "backend produced no measured queries";
        } else {
          base.metrics = {
            recall_at_10: summary.avg_recall_at_10,
            recall_at_40: summary.avg_recall_at_40,
            recall_at_100: summary.avg_recall_at_100,
            mrr: summary.avg_mrr,
            ndcg_at_10: summary.avg_ndcg_at_10,
            avg_latency_ms: summary.avg_latency_ms,
            search_queries_per_second: summary.search_queries_per_second,
          };
        }
      }
      cells.push(base);
    }
  }

  return {
    schema: 1,
    manifest_name: manifest.name,
    manifest_sha256: sha256File(absoluteManifest),
    fixture_sha256: manifest.fixture.sha256,
    corpus_sha256_tree_v1: manifest.corpus.sha256_tree_v1,
    generated_at: new Date().toISOString(),
    cells,
    notes: [
      "Scores are comparable only when every frozen identity and host condition is controlled.",
      "peak_rss_bytes is the runner process high-water mark; run models in separate processes for clean RAM comparisons.",
      "A missing measurement is never represented as a zero score.",
    ],
  };
}
