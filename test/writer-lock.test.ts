import { describe, test, expect, afterEach, vi } from "vitest";
import { mkdir, mkdtemp, rm, stat, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { hostname } from "node:os";
import { openDatabase, type Database } from "../src/db.js";
import {
  acquireWriterLock,
  getWriterLockPath,
  withWriterLock,
  WriterLockCorruptError,
  WriterLockTimeoutError,
} from "../src/writer-lock.js";

const tempDirs: string[] = [];

async function tempResource(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "qmd-lock-test-"));
  tempDirs.push(dir);
  return join(dir, "index.sqlite");
}

type LeaseRow = {
  token: string;
  pid: number;
  host: string;
  heartbeat_at: number;
  expires_at: number;
};

function openLockSidecar(resource: string): Database {
  const db = openDatabase(getWriterLockPath(resource), { busyTimeoutMs: 25 });
  db.exec(`
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
  return db;
}

function readLease(db: Database): LeaseRow | undefined {
  return db.prepare(`
    SELECT token, pid, host, heartbeat_at, expires_at
    FROM writer_lock
    WHERE name = 'writer'
  `).get<LeaseRow>();
}

function replaceLease(db: Database, token: string, pid: number, now: number): void {
  db.prepare(`
    UPDATE writer_lock
    SET token = ?,
        pid = ?,
        host = ?,
        purpose = ?,
        created_at = ?,
        heartbeat_at = ?,
        expires_at = ?
    WHERE name = 'writer'
  `).run(token, pid, hostname(), "replacement", now, now, now + 60_000);
}

function insertLease(db: Database, token: string, pid: number, now: number, host = hostname()): void {
  db.prepare(`
    INSERT INTO writer_lock
      (name, token, pid, host, purpose, created_at, heartbeat_at, expires_at)
    VALUES ('writer', ?, ?, ?, ?, ?, ?, ?)
  `).run(token, pid, host, "inserted", now, now, now);
}

afterEach(async () => {
  vi.useRealTimers();
  for (const dir of tempDirs.splice(0)) {
    await rm(dir, { recursive: true, force: true });
  }
});

describe("writer lock", () => {
  test("times out when another owner holds the lock", async () => {
    const resource = await tempResource();
    const lock = await acquireWriterLock(resource, "test-holder", { waitMs: 100 });
    try {
      await expect(withWriterLock(resource, "test-waiter", () => undefined, {
        waitMs: 25,
        pollMs: 5,
      })).rejects.toBeInstanceOf(WriterLockTimeoutError);
    } finally {
      lock.release();
    }
  });

  test("recovers dead-PID stale leases", async () => {
    const resource = await tempResource();
    const db = openLockSidecar(resource);
    insertLease(db, "stale-owner", 99999999, Date.now() - 10_000);
    db.close();

    const lock = await acquireWriterLock(resource, "new-owner", {
      waitMs: 100,
      staleMs: 1,
      pollMs: 5,
    });
    try {
      expect(lock.token).not.toBe("stale-owner");
    } finally {
      lock.release();
    }
  });

  test("does not replace malformed sidecar during initialization grace period", async () => {
    const resource = await tempResource();
    const lockPath = getWriterLockPath(resource);
    await mkdir(lockPath);

    await expect(acquireWriterLock(resource, "contender", {
      waitMs: 25,
      staleMs: 1,
      pollMs: 5,
      initGraceMs: 1_000,
    })).rejects.toBeInstanceOf(WriterLockTimeoutError);
    expect((await stat(lockPath)).isDirectory()).toBe(true);
  });

  test("fails without replacing malformed sidecar after initialization grace period", async () => {
    const resource = await tempResource();
    const lockPath = getWriterLockPath(resource);
    await mkdir(lockPath);
    const old = new Date(Date.now() - 10_000);
    await utimes(lockPath, old, old);

    await expect(acquireWriterLock(resource, "after-grace", {
      waitMs: 100,
      staleMs: 1,
      pollMs: 5,
      initGraceMs: 10,
    })).rejects.toBeInstanceOf(WriterLockCorruptError);
    expect((await stat(lockPath)).isDirectory()).toBe(true);
  });

  test("does not quarantine valid sidecar schema errors", async () => {
    const resource = await tempResource();
    const lockPath = getWriterLockPath(resource);
    const db = openDatabase(lockPath, { busyTimeoutMs: 25 });
    db.exec("CREATE VIEW writer_lock AS SELECT 'not-a-lock-table' AS token;");
    db.close();

    await expect(acquireWriterLock(resource, "schema-error", {
      waitMs: 100,
      pollMs: 5,
      initGraceMs: 0,
    })).rejects.not.toBeInstanceOf(WriterLockCorruptError);

    const check = openDatabase(lockPath, { busyTimeoutMs: 25 });
    try {
      const row = check.prepare("SELECT token FROM writer_lock").get<{ token: string }>();
      expect(row?.token).toBe("not-a-lock-table");
    } finally {
      check.close();
    }
  });

  test("does not let old owner heartbeat or release affect a replacement token", async () => {
    vi.useFakeTimers({ now: Date.now() });
    const resource = await tempResource();
    const oldOwner = await acquireWriterLock(resource, "old-owner", {
      waitMs: 100,
      staleMs: 1_000,
    });
    const db = openLockSidecar(resource);
    const replacementToken = "replacement-token";

    try {
      replaceLease(db, replacementToken, process.pid, Date.now());
      await vi.advanceTimersByTimeAsync(300);
      expect(readLease(db)?.token).toBe(replacementToken);

      oldOwner.release();
      expect(readLease(db)?.token).toBe(replacementToken);
    } finally {
      db.close();
    }
  });

  test("does not steal an expired same-host lease while the PID is alive", async () => {
    const resource = await tempResource();
    const owner = await acquireWriterLock(resource, "live-owner", {
      waitMs: 100,
      staleMs: 1,
    });
    const db = openLockSidecar(resource);
    try {
      replaceLease(db, owner.token, process.pid, Date.now() - 10_000);
      await expect(acquireWriterLock(resource, "contender", {
        waitMs: 25,
        staleMs: 1,
        pollMs: 5,
      })).rejects.toBeInstanceOf(WriterLockTimeoutError);
      expect(readLease(db)?.token).toBe(owner.token);
    } finally {
      db.close();
      owner.release();
    }
  });

  test("does not take over when an expired remote lease refreshes before CAS update", async () => {
    const resource = await tempResource();
    const db = openLockSidecar(resource);
    const oldHeartbeat = Date.now() - 120_000;
    insertLease(db, "remote-token", 12345, oldHeartbeat, "remote-host");
    let refreshed = false;

    try {
      await expect(acquireWriterLock(resource, "contender", {
        waitMs: 35,
        staleMs: 60_000,
        pollMs: 5,
        testHooks: {
          beforeStaleTakeover: ({ token }) => {
            if (refreshed) return;
            refreshed = true;
            const now = Date.now();
            db.prepare(`
              UPDATE writer_lock
              SET heartbeat_at = ?, expires_at = ?
              WHERE name = 'writer' AND token = ?
            `).run(now, now + 60_000, token);
          },
        },
      })).rejects.toBeInstanceOf(WriterLockTimeoutError);

      const lease = readLease(db);
      expect(refreshed).toBe(true);
      expect(lease?.token).toBe("remote-token");
      expect(lease?.heartbeat_at).toBeGreaterThan(oldHeartbeat);
    } finally {
      db.close();
    }
  });

  test("legacy JSON lock artifacts do not block SQLite lease acquisition", async () => {
    const resource = await tempResource();
    await writeFile(`${resource}.qmd-writer.lock`, "{", "utf-8");

    const lock = await acquireWriterLock(resource, "sqlite-owner", {
      waitMs: 100,
      pollMs: 5,
    });
    try {
      expect(lock.path).toBe(getWriterLockPath(resource));
    } finally {
      lock.release();
    }
  });

  test("serializes contenders and releases ownership", async () => {
    const resource = await tempResource();
    const first = await acquireWriterLock(resource, "first", { waitMs: 100 });
    let acquired = false;
    const second = withWriterLock(resource, "second", () => {
      acquired = true;
      return "ok";
    }, { waitMs: 250, pollMs: 5 });

    await new Promise(resolve => setTimeout(resolve, 30));
    expect(acquired).toBe(false);
    first.release();

    await expect(second).resolves.toBe("ok");
    expect(acquired).toBe(true);
  });
});
