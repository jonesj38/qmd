#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { runFrozenMatrix, type LocalRunSpec } from "../src/bench/matrix.js";

function valueAfter(flag: string): string | undefined {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

const manifestPath = resolve(valueAfter("--manifest") ?? "bench/model-matrix.json");
const runsPath = valueAfter("--runs");
const localSpec = runsPath
  ? JSON.parse(readFileSync(resolve(runsPath), "utf8")) as LocalRunSpec
  : undefined;

try {
  const result = await runFrozenMatrix(manifestPath, localSpec);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
