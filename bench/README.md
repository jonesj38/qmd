# Frozen embedding benchmark matrix

`model-matrix.json` freezes the small checked-in corpus, relevance fixture,
retrieval depths, pipeline modes, model repository revisions, artifact hashes, and
the local full-pipeline controls. It contains no benchmark scores.

The runner never downloads or indexes models. With no local run spec it emits all
15 model/mode cells as `not_measured`, which is the honest baseline:

```sh
node node_modules/tsx/dist/cli.mjs scripts/run-model-benchmark.ts \
  --manifest bench/model-matrix.json > matrix.json
```

To measure a model, copy `model-matrix.local.example.json` outside version control
and replace that model's reason with:

```json
{
  "db_path": "/absolute/path/to/model-index.sqlite",
  "config_path": "/absolute/path/to/frozen-model-config.yml",
  "artifact_path": "/absolute/path/to/the-exact.gguf",
  "full_pipeline": {
    "generator_artifact_path": "/absolute/path/to/qmd-query-expansion-1.7B-q4_k_m.gguf",
    "reranker_artifact_path": "/absolute/path/to/qwen3-reranker-0.6b-q8_0.gguf"
  },
  "indexing": {
    "duration_ms": 12345,
    "documents": 6,
    "chunks": 6,
    "peak_rss_bytes": 123456789
  }
}
```

The config must select the local `artifact_path` (not a mutable Hub alias), set
`embedding.identity.revision` to the manifest revision, and point its collection at
the frozen corpus. For OpenAI it must select `text-embedding-3-small`, declare the
manifest revision string, and use an already-built independent index. Keep one
index per embedding fingerprint; vectors are not cross-compatible.

Then run:

```sh
node node_modules/tsx/dist/cli.mjs scripts/run-model-benchmark.ts \
  --manifest bench/model-matrix.json \
  --runs /absolute/path/to/model-matrix.local.json > matrix.json
```

The runner verifies fixture, corpus, and local GGUF hashes before querying. It
records Recall@10/40/100, MRR, binary nDCG@10, mean search latency, search
throughput, process peak RSS, on-disk SQLite/WAL size, and QMD's canonical embedding
identity/fingerprint. Indexing throughput is reported only when actual duration and
counters are supplied; otherwise it has an explicit reason. Backend failures and
unsupported modes are also `not_measured`, never fabricated zero scores.

`hybrid-no-rerank` is deliberately literal-query BM25 + vector RRF (equal weights,
RRF constant 60). It does not invoke query expansion or the reranker. `full` keeps
QMD's normal full search behavior and therefore requires the two pinned control
artifacts. OpenAI full mode is not comparable in this frozen matrix because QMD's
remote expansion/rerank models are mutable deployments; vector-only and
hybrid-no-rerank remain measurable with a prepared OpenAI index and credentials.

Peak RSS is a process high-water mark. For clean RAM comparisons, prepare a local
spec containing one measurable model at a time and run each in a fresh process,
then merge cells only when the manifest hash matches.
