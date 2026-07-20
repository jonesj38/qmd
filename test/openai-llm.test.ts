import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";

const openAiMocks = vi.hoisted(() => {
  const embeddingsCreate = vi.fn();
  const chatCreate = vi.fn();
  const OpenAIConstructor = vi.fn(() => ({
    embeddings: { create: embeddingsCreate },
    chat: { completions: { create: chatCreate } },
  }));
  return { embeddingsCreate, chatCreate, OpenAIConstructor };
});

vi.mock("openai", () => ({
  default: openAiMocks.OpenAIConstructor,
}));

import {
  DEFAULT_OPENAI_QUERY_TIMEOUT_MS,
  OpenAIEmbedding,
} from "../src/openai-llm.js";

type CapturedRequestOptions = {
  timeout?: number;
  maxRetries?: number;
  signal?: AbortSignal | null;
};

function expectQueryRequestOptions(options: unknown, timeout: number) {
  const requestOptions = options as CapturedRequestOptions | undefined;
  expect(requestOptions).toMatchObject({ timeout, maxRetries: 0 });
  expect(requestOptions?.signal).toBeInstanceOf(AbortSignal);
  return requestOptions!;
}

describe("OpenAI query-stage timeouts", () => {
  const originalQueryTimeout = process.env.QMD_OPENAI_QUERY_TIMEOUT_MS;
  let warnSpy: ReturnType<typeof vi.spyOn>;
  let errorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    openAiMocks.embeddingsCreate.mockReset();
    openAiMocks.chatCreate.mockReset();
    openAiMocks.OpenAIConstructor.mockClear();
    warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    delete process.env.QMD_OPENAI_QUERY_TIMEOUT_MS;
  });

  afterEach(() => {
    vi.useRealTimers();
    warnSpy.mockRestore();
    errorSpy.mockRestore();
    if (originalQueryTimeout === undefined) delete process.env.QMD_OPENAI_QUERY_TIMEOUT_MS;
    else process.env.QMD_OPENAI_QUERY_TIMEOUT_MS = originalQueryTimeout;
  });

  test("applies timeout and disables SDK retries for query embeddings", async () => {
    process.env.QMD_OPENAI_QUERY_TIMEOUT_MS = "1234";
    openAiMocks.embeddingsCreate.mockResolvedValue({
      data: [{ embedding: [0.1, 0.2, 0.3] }],
    });

    const llm = new OpenAIEmbedding({ apiKey: "test-key" });
    const result = await llm.embed("query text", { isQuery: true });

    expect(result?.embedding).toEqual([0.1, 0.2, 0.3]);
    expect(openAiMocks.embeddingsCreate).toHaveBeenCalledWith(
      { model: "text-embedding-3-small", input: "query text" },
      expect.objectContaining({ timeout: 1234, maxRetries: 0 }),
    );
    expectQueryRequestOptions(openAiMocks.embeddingsCreate.mock.calls[0]?.[1], 1234);
  });

  test("does not shorten maintenance document embedding or embedBatch calls", async () => {
    openAiMocks.embeddingsCreate.mockResolvedValueOnce({
      data: [{ embedding: [0.4, 0.5, 0.6] }],
    });
    openAiMocks.embeddingsCreate.mockResolvedValueOnce({
      data: [{ embedding: [0.7, 0.8, 0.9] }],
    });

    const llm = new OpenAIEmbedding({ apiKey: "test-key" });
    await llm.embed("document text", { isQuery: false });
    await llm.embedBatch(["document batch"]);

    expect(openAiMocks.embeddingsCreate.mock.calls[0]?.[1]).toBeUndefined();
    expect(openAiMocks.embeddingsCreate.mock.calls[1]?.[1]).toBeUndefined();
  });

  test("returns null query embeddings on timeout errors without retrying", async () => {
    openAiMocks.embeddingsCreate.mockRejectedValue(new Error("request timed out"));

    const llm = new OpenAIEmbedding({ apiKey: "test-key" });
    const single = await llm.embed("query text", { isQuery: true });
    const batch = await llm.embedBatch(["query one", "query two"], { isQuery: true });

    expect(single).toBeNull();
    expect(batch).toEqual([null, null]);
    expect(openAiMocks.embeddingsCreate).toHaveBeenCalledTimes(2);
    expectQueryRequestOptions(openAiMocks.embeddingsCreate.mock.calls[0]?.[1], DEFAULT_OPENAI_QUERY_TIMEOUT_MS);
    expectQueryRequestOptions(openAiMocks.embeddingsCreate.mock.calls[1]?.[1], DEFAULT_OPENAI_QUERY_TIMEOUT_MS);
  });

  test("application deadline bounds a never-settling query embedding request", async () => {
    vi.useFakeTimers();
    process.env.QMD_OPENAI_QUERY_TIMEOUT_MS = "50";
    let capturedSignal: AbortSignal | undefined;
    openAiMocks.embeddingsCreate.mockImplementation((_body, options?: CapturedRequestOptions) => {
      capturedSignal = options?.signal ?? undefined;
      return new Promise(() => undefined);
    });

    const llm = new OpenAIEmbedding({ apiKey: "test-key" });
    const pending = llm.embed("query text", { isQuery: true });
    let settled = false;
    void pending.then(() => { settled = true; }, () => { settled = true; });

    await vi.advanceTimersByTimeAsync(49);
    await Promise.resolve();
    expect(settled).toBe(false);

    await vi.advanceTimersByTimeAsync(1);
    await expect(pending).resolves.toBeNull();
    expect(capturedSignal?.aborted).toBe(true);
    expectQueryRequestOptions(openAiMocks.embeddingsCreate.mock.calls[0]?.[1], 50);
  });

  test("query batch embedding, expansion, and rerank abort at the application deadline", async () => {
    vi.useFakeTimers();
    process.env.QMD_OPENAI_QUERY_TIMEOUT_MS = "25";
    const capturedSignals: AbortSignal[] = [];
    const abortAwareRequest = vi.fn((_body, options?: CapturedRequestOptions) => {
      if (options?.signal) capturedSignals.push(options.signal);
      return new Promise((_resolve, reject) => {
        options?.signal?.addEventListener("abort", () => {
          reject(options.signal?.reason ?? new Error("aborted"));
        }, { once: true });
      });
    });
    openAiMocks.embeddingsCreate.mockImplementation(abortAwareRequest);
    openAiMocks.chatCreate.mockImplementation(abortAwareRequest);

    const llm = new OpenAIEmbedding({ apiKey: "test-key" });

    const batch = llm.embedBatch(["query one", "query two"], { isQuery: true });
    await vi.advanceTimersByTimeAsync(25);
    await expect(batch).resolves.toEqual([null, null]);

    const expanded = llm.expandQuery("bounded fallback");
    await vi.advanceTimersByTimeAsync(25);
    await expect(expanded).resolves.toEqual([
      { type: "lex", text: "bounded fallback" },
      { type: "vec", text: "bounded fallback" },
    ]);

    const reranked = llm.rerank("bounded rerank", [
      { file: "a.md", text: "alpha" },
      { file: "b.md", text: "beta" },
      { file: "c.md", text: "gamma" },
    ]);
    await vi.advanceTimersByTimeAsync(25);
    await expect(reranked).resolves.toMatchObject({
      model: "passthrough-fallback",
      results: [
        { file: "a.md" },
        { file: "b.md" },
        { file: "c.md" },
      ],
    });

    expect(capturedSignals).toHaveLength(3);
    expect(capturedSignals.every(signal => signal.aborted)).toBe(true);
    expectQueryRequestOptions(openAiMocks.embeddingsCreate.mock.calls[0]?.[1], 25);
    expectQueryRequestOptions(openAiMocks.chatCreate.mock.calls[0]?.[1], 25);
    expectQueryRequestOptions(openAiMocks.chatCreate.mock.calls[1]?.[1], 25);
  });

  test("bounds query expansion and falls back to the original query", async () => {
    openAiMocks.chatCreate.mockRejectedValue(new Error("request timed out"));

    const llm = new OpenAIEmbedding({ apiKey: "test-key" });
    const expanded = await llm.expandQuery("bounded fallback");

    expect(expanded).toEqual([
      { type: "lex", text: "bounded fallback" },
      { type: "vec", text: "bounded fallback" },
    ]);
    expectQueryRequestOptions(openAiMocks.chatCreate.mock.calls[0]?.[1], DEFAULT_OPENAI_QUERY_TIMEOUT_MS);
  });

  test("bounds OpenAI rerank and preserves original order on failure", async () => {
    openAiMocks.chatCreate.mockRejectedValue(new Error("request timed out"));
    const docs = [
      { file: "a.md", text: "alpha" },
      { file: "b.md", text: "beta" },
      { file: "c.md", text: "gamma" },
    ];

    const llm = new OpenAIEmbedding({ apiKey: "test-key" });
    const reranked = await llm.rerank("bounded rerank", docs);

    expect(reranked.model).toBe("passthrough-fallback");
    expect(reranked.results.map(result => result.file)).toEqual(["a.md", "b.md", "c.md"]);
    expectQueryRequestOptions(openAiMocks.chatCreate.mock.calls[0]?.[1], DEFAULT_OPENAI_QUERY_TIMEOUT_MS);
  });
});
