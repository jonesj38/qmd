import { afterEach, describe, expect, test } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { setEmbeddingConfig } from "../src/llm.js";
import {
  createStore,
  formatDocForEmbedding,
  formatQueryForEmbedding,
  getEmbeddingFingerprint,
  getEmbeddingIdentity,
  getHashesNeedingEmbedding,
  insertContent,
  insertDocument,
  maybeAdoptLegacyEmbeddingFingerprint,
} from "../src/store.js";

const model = "compatible-embed-model";
const localIdentity = {
  revision: "0123456789abcdef",
  artifact: "compatible-embed-model-q8_0.gguf",
  quantization: "Q8_0",
  pooling: "mean",
  normalization: "l2",
  dimensions: 3,
  tokenizer: "compatible-tokenizer",
  tokenizerRevision: "4.2.0",
};

function useLocal(identity = localIdentity): void {
  setEmbeddingConfig({ provider: "local", identity });
}

afterEach(() => setEmbeddingConfig({ provider: "local" }));

describe("canonical embedding identity", () => {
  test("is stable, excludes credentials and machine-local cache parents", () => {
    setEmbeddingConfig({
      provider: "openai",
      identity: { revision: "2026-08-15", dimensions: 1536 },
      openai: {
        apiKey: "first-secret",
        baseURL: "https://user:password@example.test/v1/?api_key=also-secret",
        embedModel: "service-model",
      },
    });
    const first = getEmbeddingIdentity("/home/alice/.cache/models/artifact.gguf");
    const firstFingerprint = getEmbeddingFingerprint("/home/alice/.cache/models/artifact.gguf");

    setEmbeddingConfig({
      provider: "openai",
      identity: { revision: "2026-08-15", dimensions: 1536 },
      openai: {
        apiKey: "different-secret",
        baseURL: "https://other:credentials@example.test/v1#token",
        embedModel: "service-model",
      },
    });
    const second = getEmbeddingIdentity("/var/tmp/another-cache/artifact.gguf");

    expect(second).toEqual(first);
    expect(getEmbeddingFingerprint("/var/tmp/another-cache/artifact.gguf")).toBe(firstFingerprint);
    const serialized = JSON.stringify(first);
    expect(serialized).not.toContain("secret");
    expect(serialized).not.toContain("password");
    expect(serialized).not.toContain("/home/alice");
    expect(serialized).not.toContain("/var/tmp");
    expect(first.backend).toBe("openai-compatible:https://example.test/v1");
    expect(first.model.reference).toBe("file:artifact.gguf");
  });

  test("changes for every compatibility-affecting semantic field", () => {
    useLocal();
    const baseline = getEmbeddingFingerprint(model);
    const cases = [
      { provider: "openai" as const, identity: localIdentity, openai: { apiKey: "ignored", baseURL: "https://example.test/v1" } },
      { provider: "local" as const, identity: { ...localIdentity, backend: "another-backend" } },
      { provider: "local" as const, identity: { ...localIdentity, revision: "different" } },
      { provider: "local" as const, identity: { ...localIdentity, artifact: "different.gguf" } },
      { provider: "local" as const, identity: { ...localIdentity, quantization: "Q4_K_M" } },
      { provider: "local" as const, identity: { ...localIdentity, pooling: "cls" } },
      { provider: "local" as const, identity: { ...localIdentity, normalization: "none" } },
      { provider: "local" as const, identity: { ...localIdentity, dimensions: 4 } },
      { provider: "local" as const, identity: { ...localIdentity, tokenizer: "other-tokenizer" } },
      { provider: "local" as const, identity: { ...localIdentity, tokenizerRevision: "5.0.0" } },
      { provider: "local" as const, identity: { ...localIdentity, queryPrompt: "query: {{query}}" } },
      { provider: "local" as const, identity: { ...localIdentity, documentPrompt: "title={{title}} text={{text}}" } },
    ];

    for (const config of cases) {
      setEmbeddingConfig(config);
      expect(getEmbeddingFingerprint(model)).not.toBe(baseline);
    }
    useLocal();
    expect(getEmbeddingFingerprint("different-model")).not.toBe(baseline);
  });

  test("configured prompts are both applied and represented by the identity", () => {
    setEmbeddingConfig({
      provider: "local",
      identity: {
        ...localIdentity,
        queryPrompt: "Q={{query}}",
        documentPrompt: "T={{title}};D={{text}}",
      },
    });
    expect(formatQueryForEmbedding("needle", model)).toBe("Q=needle");
    expect(formatDocForEmbedding("body", "Heading", model)).toBe("T=Heading;D=body");
    const identity = getEmbeddingIdentity(model);
    expect(identity.preprocessing.query).toContain("Q=__qmd_embedding_query_probe__");
    expect(identity.preprocessing.document).toContain("T=__qmd_embedding_title_probe__");
  });
});

describe("embedding identity vector safety", () => {
  test("filters stale identities and rejects incompatible query dimensions", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-identity-"));
    const store = createStore(join(dir, "index.sqlite"));
    const now = new Date().toISOString();
    try {
      useLocal();
      store.ensureVecTable(3);
      insertContent(store.db, "stale-hash", "stale body", now);
      insertDocument(store.db, "docs", "stale.md", "Stale", "stale-hash", now, now);
      insertContent(store.db, "current-hash", "current body", now);
      insertDocument(store.db, "docs", "current.md", "Current", "current-hash", now, now);

      setEmbeddingConfig({ provider: "local", identity: { ...localIdentity, revision: "old" } });
      const staleFingerprint = getEmbeddingFingerprint(model);
      useLocal();
      const currentFingerprint = getEmbeddingFingerprint(model);
      store.insertEmbedding("stale-hash", 0, 0, new Float32Array([1, 0, 0]), model, now, 1, staleFingerprint);
      store.insertEmbedding("current-hash", 0, 0, new Float32Array([0.8, 0.6, 0]), model, now, 1, currentFingerprint);

      const results = await store.searchVec("ignored", model, 10, undefined, undefined, [1, 0, 0]);
      expect(results.map(result => result.hash)).toEqual(["current-hash"]);
      await expect(store.searchVec("ignored", model, 10, undefined, undefined, [1, 0])).rejects.toThrow(/dimension mismatch|declares 3d/i);
    } finally {
      store.close();
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("legacy empty fingerprints stay stale unless the compatibility probe can verify them", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-legacy-identity-"));
    const store = createStore(join(dir, "index.sqlite"));
    const now = new Date().toISOString();
    try {
      useLocal();
      insertContent(store.db, "legacy-hash", "legacy body", now);
      insertDocument(store.db, "docs", "legacy.md", "Legacy", "legacy-hash", now, now);
      store.db.prepare(`INSERT INTO content_vectors
        (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
        VALUES (?, 0, 0, ?, '', 1, ?)`
      ).run("legacy-hash", model, now);

      expect(getHashesNeedingEmbedding(store.db, undefined, model)).toBe(1);
      const adoption = await maybeAdoptLegacyEmbeddingFingerprint(store, model);
      expect(adoption.adopted).toBe(0);
      expect(adoption.reason).toMatch(/vectors_vec table is missing/i);
      expect(store.db.prepare(`SELECT embed_fingerprint FROM content_vectors WHERE hash = ?`).get("legacy-hash"))
        .toEqual({ embed_fingerprint: "" });
    } finally {
      store.close();
      await rm(dir, { recursive: true, force: true });
    }
  });
});
