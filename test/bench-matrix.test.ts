import { describe, expect, test } from "vitest";
import { resolve } from "node:path";
import { loadFrozenManifest, runFrozenMatrix, sha256Tree, verifyFrozenInputs } from "../src/bench/matrix.js";
import { computeSummary } from "../src/bench/bench.js";

const manifestPath = resolve("bench/model-matrix.json");

describe("frozen model benchmark matrix", () => {
  test("fixture and corpus match their frozen hashes", () => {
    const manifest = loadFrozenManifest(manifestPath);
    expect(() => verifyFrozenInputs(manifest, resolve("bench"))).not.toThrow();
    expect(sha256Tree(resolve("test/eval-docs"))).toBe(manifest.corpus.sha256_tree_v1);
  });

  test("emits every cell as explicitly not measured without a local run spec", async () => {
    const result = await runFrozenMatrix(manifestPath);
    expect(result.cells).toHaveLength(5 * 3);
    expect(result.cells.every(cell => cell.status === "not_measured")).toBe(true);
    expect(result.cells.every(cell => !!cell.not_measured_reason)).toBe(true);
    expect(result.cells.every(cell => cell.metrics === undefined)).toBe(true);
  });

  test("does not average unavailable backend placeholders into quality scores", () => {
    const summary = computeSummary([{
      id: "q",
      query: "query",
      type: "semantic",
      backends: {
        vector: {
          status: "not_measured",
          not_measured_reason: "model unavailable",
          precision_at_k: 0,
          recall: 0,
          recall_at_1: 0,
          recall_at_3: 0,
          recall_at_5: 0,
          recall_at_10: 0,
          recall_at_40: 0,
          recall_at_100: 0,
          ndcg_at_10: 0,
          mrr: 0,
          f1: 0,
          hits_at_k: 0,
          total_expected: 1,
          latency_ms: 1,
          top_files: [],
          matched_files: [],
          unmatched_expected_files: ["missing.md"],
        },
      },
    }]);
    expect(summary).toEqual({});
  });
});
