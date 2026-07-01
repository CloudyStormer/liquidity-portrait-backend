import fs from 'node:fs/promises';
import path from 'node:path';
import { DATA_DIR } from './config.js';
import type { StoreShape } from './types.js';

const STORE_PATH = path.join(DATA_DIR, 'store.json');

const emptyStore = (): StoreShape => ({
  users: [],
  history: [],
  photoUsageRecords: [],
  openaiRequests: []
});

export async function ensureDataDirs() {
  await fs.mkdir(DATA_DIR, { recursive: true });
}

export async function readStore(): Promise<StoreShape> {
  await ensureDataDirs();
  try {
    const raw = await fs.readFile(STORE_PATH, 'utf8');
    const parsed = JSON.parse(raw) as StoreShape;
    return {
      users: Array.isArray(parsed.users) ? parsed.users : [],
      history: Array.isArray(parsed.history) ? parsed.history : [],
      photoUsageRecords: Array.isArray(parsed.photoUsageRecords) ? parsed.photoUsageRecords : [],
      openaiRequests: Array.isArray(parsed.openaiRequests) ? parsed.openaiRequests : []
    };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return emptyStore();
    }
    throw error;
  }
}

export async function writeStore(store: StoreShape) {
  await ensureDataDirs();
  const tempPath = `${STORE_PATH}.tmp`;
  await fs.writeFile(tempPath, JSON.stringify(store, null, 2), 'utf8');
  await fs.rename(tempPath, STORE_PATH);
}

export async function updateStore(mutator: (store: StoreShape) => void | Promise<void>) {
  const store = await readStore();
  await mutator(store);
  await writeStore(store);
  return store;
}
