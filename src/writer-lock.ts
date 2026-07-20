import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, statSync } from "node:fs";
import { dirname } from "node:path";
import { hostname } from "node:os";
import { openDatabase, type Database } from "./db.js";

export const DEFAULT_WRITER_LOCK_WAIT_MS = 5_000;
export const DEFAULT_WRITER_LOCK_STALE_MS = 10 * 60_000;
export const DEFAULT_WRITER_LOCK_INIT_GRACE_MS = 1_000;

const LOCK_NAME = "writer";

type WriterLockRow = {
  name: string;
  token: string;
  pid: number;
  host: string;
  purpose: string;
  created_at: number;
  heartbeat_at: number;
  expires_at: number;
};

export type WriterLockOptions = {
  waitMs?: number;
  staleMs?: number;
  pollMs?: number;
  initGraceMs?: number;
  testHooks?: {
    beforeStaleTakeover?: (observed: {
      token: string;
      heartbeatAt: number;
      expiresAt: number;
    }) => void | Promise<void>;
  };
};

export type WriterLockHandle = {
  path: string;
  token: string;
  release: () => void;
};

export class WriterLockTimeoutError extends Error {
  constructor(lockPath: string, waitMs: number) {
    super(`Timed out waiting ${waitMs}ms for QMD writer lock: ${lockPath}`);
    this.name = "WriterLockTimeoutError";
  }
}

export class WriterLockCorruptError extends Error {
  constructor(lockPath: string) {
    super(
      `QMD writer lock sidecar is malformed and was left untouched: ${lockPath}. ` +
      "After confirming no qmd maintenance process is using this index, move or remove that sidecar and retry.",
    );
    this.name = "WriterLockCorruptError";
  }
}

function parseNonNegativeIntegerEnv(name: string, fallback: number): number {
  const raw = process.env[name]?.trim();
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

export function resolveWriterLockWaitMs(value?: number): number {
  return value ?? parseNonNegativeIntegerEnv("QMD_WRITER_LOCK_WAIT_MS", DEFAULT_WRITER_LOCK_WAIT_MS);
}

export function resolveWriterLockStaleMs(value?: number): number {
  return value ?? parseNonNegativeIntegerEnv("QMD_WRITER_LOCK_STALE_MS", DEFAULT_WRITER_LOCK_STALE_MS);
}

export function getWriterLockPath(resourcePath: string): string {
  return `${resourcePath}.qmd-writer-lock.sqlite`;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isPidAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function isRecoverableLease(row: WriterLockRow, staleMs: number, now: number): boolean {
  if (row.host === hostname()) {
    // A paused but live local process must not be split-brained solely because
    // its heartbeat is old. Dead-PID recovery remains immediate and bounded.
    return !isPidAlive(row.pid);
  }
  return now - row.heartbeat_at > staleMs || now >= row.expires_at;
}

function sidecarAgeMs(path: string, now: number): number | null {
  try {
    return Math.max(0, now - statSync(path).mtimeMs);
  } catch {
    return null;
  }
}

function sqliteBusyTimeoutFor(remainingWaitMs: number): number {
  return Math.max(0, Math.min(50, remainingWaitMs));
}

function initializeLockDatabase(db: Database): void {
  db.exec(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS writer_lock (
      name TEXT PRIMARY KEY,
      token TEXT NOT NULL,
      pid INTEGER NOT NULL,
      host TEXT NOT NULL,
      purpose TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      heartbeat_at INTEGER NOT NULL,
      expires_at INTEGER NOT NULL
    );
  `);
}

function openLockDatabase(lockPath: string, remainingWaitMs: number): Database {
  const db = openDatabase(lockPath, {
    busyTimeoutMs: sqliteBusyTimeoutFor(remainingWaitMs),
  });
  try {
    initializeLockDatabase(db);
  } catch (error) {
    try {
      db.close();
    } catch {}
    throw error;
  }
  return db;
}

function isRecoverableSidecarOpenError(error: unknown, lockPath: string): boolean {
  const code = (error as { code?: string }).code;
  if (code === "SQLITE_NOTADB" || code === "SQLITE_CORRUPT") return true;
  if (code === "SQLITE_CANTOPEN" || code === "EISDIR" || code === "ENOTDIR") {
    try {
      return !statSync(lockPath).isFile();
    } catch {
      return false;
    }
  }
  return false;
}

function tryOpenLockDatabase(lockPath: string, remainingWaitMs: number, initGraceMs: number): Database | null {
  try {
    return openLockDatabase(lockPath, remainingWaitMs);
  } catch (error) {
    const code = (error as { code?: string }).code;
    if (code === "SQLITE_BUSY" || code === "SQLITE_LOCKED") throw error;

    if (!isRecoverableSidecarOpenError(error, lockPath)) throw error;

    const ageMs = existsSync(lockPath) ? sidecarAgeMs(lockPath, Date.now()) : null;
    if (ageMs !== null && ageMs < initGraceMs) return null;
    if (ageMs !== null && ageMs >= initGraceMs) {
      throw new WriterLockCorruptError(lockPath);
    }
    throw error;
  }
}

function readCurrentLease(db: Database): WriterLockRow | undefined {
  return db.prepare(`
    SELECT name, token, pid, host, purpose, created_at, heartbeat_at, expires_at
    FROM writer_lock
    WHERE name = ?
  `).get<WriterLockRow>(LOCK_NAME);
}

function insertLease(db: Database, token: string, purpose: string, now: number, staleMs: number): boolean {
  const result = db.prepare(`
    INSERT OR IGNORE INTO writer_lock
      (name, token, pid, host, purpose, created_at, heartbeat_at, expires_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `).run(LOCK_NAME, token, process.pid, hostname(), purpose, now, now, now + staleMs);
  return result.changes === 1;
}

function takeOverLease(db: Database, observed: WriterLockRow, token: string, purpose: string, now: number, staleMs: number): boolean {
  const result = db.prepare(`
    UPDATE writer_lock
    SET token = ?,
        pid = ?,
        host = ?,
        purpose = ?,
        created_at = ?,
        heartbeat_at = ?,
        expires_at = ?
    WHERE name = ?
      AND token = ?
      AND heartbeat_at = ?
      AND expires_at = ?
  `).run(
    token,
    process.pid,
    hostname(),
    purpose,
    now,
    now,
    now + staleMs,
    LOCK_NAME,
    observed.token,
    observed.heartbeat_at,
    observed.expires_at,
  );
  return result.changes === 1;
}

function heartbeatLease(db: Database, token: string, staleMs: number): boolean {
  const now = Date.now();
  const result = db.prepare(`
    UPDATE writer_lock
    SET heartbeat_at = ?, expires_at = ?
    WHERE name = ? AND token = ?
  `).run(now, now + staleMs, LOCK_NAME, token);
  return result.changes === 1;
}

function releaseLease(db: Database, token: string): void {
  db.prepare(`
    DELETE FROM writer_lock
    WHERE name = ? AND token = ?
  `).run(LOCK_NAME, token);
}

export async function acquireWriterLock(resourcePath: string, purpose: string, options: WriterLockOptions = {}): Promise<WriterLockHandle> {
  const lockPath = getWriterLockPath(resourcePath);
  const waitMs = resolveWriterLockWaitMs(options.waitMs);
  const staleMs = resolveWriterLockStaleMs(options.staleMs);
  const pollMs = options.pollMs ?? 100;
  const initGraceMs = options.initGraceMs ?? DEFAULT_WRITER_LOCK_INIT_GRACE_MS;
  const startedAt = Date.now();
  const token = randomUUID();

  mkdirSync(dirname(lockPath), { recursive: true });

  while (true) {
    const elapsedMs = Date.now() - startedAt;
    const remainingWaitMs = Math.max(0, waitMs - elapsedMs);
    let db: Database | null = null;
    let keepDbOpen = false;

    try {
      db = tryOpenLockDatabase(lockPath, remainingWaitMs, initGraceMs);
      if (db) {
        const now = Date.now();
        if (insertLease(db, token, purpose, now, staleMs)) {
          const heartbeatMs = Math.max(250, Math.min(5_000, Math.floor(staleMs / 4) || 250));
          keepDbOpen = true;
          const heartbeat = setInterval(() => {
            try {
              if (!heartbeatLease(db!, token, staleMs)) clearInterval(heartbeat);
            } catch {
              clearInterval(heartbeat);
            }
          }, heartbeatMs);
          heartbeat.unref();

          return {
            path: lockPath,
            token,
            release: () => {
              clearInterval(heartbeat);
              try {
                releaseLease(db!, token);
              } catch {
                // Best effort. The lease expires or is recoverable by dead PID.
              } finally {
                try {
                  db!.close();
                } catch {}
              }
            },
          };
        }

        const current = readCurrentLease(db);
        if (current && isRecoverableLease(current, staleMs, Date.now())) {
          await options.testHooks?.beforeStaleTakeover?.({
            token: current.token,
            heartbeatAt: current.heartbeat_at,
            expiresAt: current.expires_at,
          });
          if (takeOverLease(db, current, token, purpose, Date.now(), staleMs)) {
            const heartbeatMs = Math.max(250, Math.min(5_000, Math.floor(staleMs / 4) || 250));
            keepDbOpen = true;
            const heartbeat = setInterval(() => {
              try {
                if (!heartbeatLease(db!, token, staleMs)) clearInterval(heartbeat);
              } catch {
                clearInterval(heartbeat);
              }
            }, heartbeatMs);
            heartbeat.unref();

            return {
              path: lockPath,
              token,
              release: () => {
                clearInterval(heartbeat);
                try {
                  releaseLease(db!, token);
                } catch {
                  // Best effort. The lease expires or is recoverable by dead PID.
                } finally {
                  try {
                    db!.close();
                  } catch {}
                }
              },
            };
          }
        }
      }
    } catch (error) {
      const code = (error as { code?: string }).code;
      if (code !== "SQLITE_BUSY" && code !== "SQLITE_LOCKED") throw error;
    } finally {
      if (db && !keepDbOpen) {
        try {
          db.close();
        } catch {}
      }
    }

    if (Date.now() - startedAt >= waitMs) {
      throw new WriterLockTimeoutError(lockPath, waitMs);
    }
    await sleep(Math.min(pollMs, Math.max(0, waitMs - (Date.now() - startedAt))));
  }
}

export async function withWriterLock<T>(
  resourcePath: string,
  purpose: string,
  fn: () => T | Promise<T>,
  options: WriterLockOptions = {},
): Promise<T> {
  const lock = await acquireWriterLock(resourcePath, purpose, options);
  try {
    return await fn();
  } finally {
    lock.release();
  }
}
