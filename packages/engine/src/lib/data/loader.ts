import { WhiteboxError } from '../base/index.ts'
import type { Dataset } from './schema/common.ts'

/**
 * Where a dataset comes from: embedded in the bundle (a generated `*.data.ts` module, loaded on demand), a remote
 * asset pinned by its SHA-256 and parsed from its bytes, or several such files parsed together.
 *
 * @concept datasets
 */
export type DatasetRef<D extends Dataset = Dataset> =
  | {
      readonly kind: 'embedded'
      readonly name: string
      readonly load: () => D | Promise<D>
    }
  | {
      readonly kind: 'remote'
      readonly name: string
      readonly url: string
      /** Lowercase hex SHA-256 of the asset's bytes. */
      readonly sha256: string
      readonly size?: number
      readonly licence?: string
      readonly parse: (bytes: Uint8Array) => D | Promise<D>
    }
  | {
      readonly kind: 'remote-files'
      readonly name: string
      readonly files: readonly RemoteFile[]
      readonly licence?: string
      /** Parse the files' bytes, in the order of `files`. */
      readonly parse: (files: readonly Uint8Array[]) => D | Promise<D>
    }

/** One pinned file of a multi-file remote dataset. */
export interface RemoteFile {
  readonly url: string
  /** Lowercase hex SHA-256 of the file's bytes. */
  readonly sha256: string
  readonly size?: number
}

/** A byte store for downloaded assets, keyed by checksum. */
export interface AssetCache {
  get(key: string): Promise<Uint8Array | undefined>
  set(key: string, bytes: Uint8Array): Promise<void>
  delete(key: string): Promise<void>
}

/**
 * An in-memory asset cache (lost when the page or process ends).
 *
 * @returns The cache.
 * @example
 * ```ts
 * import { memoryCache } from 'whitebox-engine/data'
 * const cache = memoryCache()
 * await cache.set('k', new Uint8Array([1]))
 * (await cache.get('k'))?.length // => 1
 * ```
 */
export function memoryCache(): AssetCache {
  const store = new Map<string, Uint8Array>()
  return {
    get: async (key) => store.get(key),
    set: async (key, bytes) => {
      store.set(key, bytes)
    },
    delete: async (key) => {
      store.delete(key)
    },
  }
}

function request<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result)
    r.onerror = () => reject(r.error ?? new WhiteboxError('value', 'IndexedDB request failed'))
  })
}

/** Options of {@link indexedDbCache}. */
export interface IndexedDbCacheOptions {
  /** The database name (default `whitebox-assets`). */
  readonly name?: string
  /** The IndexedDB factory (default `globalThis.indexedDB`). */
  readonly factory?: IDBFactory
}

/**
 * An asset cache in the browser's IndexedDB, so downloaded datasets survive reloads.
 *
 * @param options - Database name and factory.
 * @returns The cache.
 * @throws WhiteboxError with code `not-implemented` where IndexedDB is unavailable (such as Node).
 * @example
 * ```ts
 * import { indexedDbCache } from 'whitebox-engine/data'
 * typeof indexedDbCache // => 'function'
 * ```
 */
export function indexedDbCache(options: IndexedDbCacheOptions = {}): AssetCache {
  const factory = options.factory ?? (globalThis as { indexedDB?: IDBFactory }).indexedDB
  if (factory === undefined)
    throw new WhiteboxError('not-implemented', 'indexedDbCache: IndexedDB is not available here')
  const storeName = 'assets'
  let opened: Promise<IDBDatabase> | undefined
  const db = (): Promise<IDBDatabase> => {
    if (opened === undefined) {
      const r = factory.open(options.name ?? 'whitebox-assets', 1)
      r.onupgradeneeded = () => {
        if (!r.result.objectStoreNames.contains(storeName)) r.result.createObjectStore(storeName)
      }
      opened = request(r)
    }
    return opened
  }
  const run = async <T>(mode: IDBTransactionMode, f: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> =>
    request(f((await db()).transaction(storeName, mode).objectStore(storeName)))
  return {
    get: async (key) => {
      const v = await run('readonly', (s) => s.get(key))
      return v === undefined ? undefined : new Uint8Array(v as ArrayBuffer)
    },
    set: async (key, bytes) => {
      await run('readwrite', (s) => s.put(bytes.slice().buffer, key))
    },
    delete: async (key) => {
      await run('readwrite', (s) => s.delete(key))
    },
  }
}

/**
 * The lowercase hex SHA-256 of bytes, with Web Crypto (browsers and Node 20+).
 *
 * @param bytes - The data.
 * @returns The digest.
 * @example
 * ```ts
 * import { sha256Hex } from 'whitebox-engine/data'
 * await sha256Hex(new TextEncoder().encode('abc')) // => 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
 * ```
 */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle
  if (subtle === undefined) throw new WhiteboxError('not-implemented', 'sha256Hex: Web Crypto is not available here')
  const digest = await subtle.digest('SHA-256', bytes.slice().buffer)
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
}

/** A minimal `fetch`: what {@link load} needs from a response. */
export type Fetcher = (
  url: string,
) => Promise<{ readonly ok: boolean; readonly status: number; arrayBuffer(): Promise<ArrayBuffer> }>

/** Options of {@link load}. */
export interface LoadOptions {
  /** A cache for remote assets (default none). */
  readonly cache?: AssetCache
  /** The fetch function (default `globalThis.fetch`). */
  readonly fetch?: Fetcher
}

/**
 * Load a dataset from its reference: embedded data directly, remote assets from the cache or the network, verifying
 * the SHA-256 of downloaded bytes before parsing (and before caching them).
 *
 * @param ref - The dataset reference.
 * @param options - Cache and fetch function.
 * @returns The dataset.
 * @throws WhiteboxError with code `value` when a download fails or its checksum differs.
 * @example
 * ```ts
 * import { load, xy } from 'whitebox-engine/data'
 * (await load({ kind: 'embedded', name: 'tiny', load: () => xy([[1]], [2]) })).kind // => 'regression'
 * ```
 */
export async function load<D extends Dataset>(ref: DatasetRef<D>, options: LoadOptions = {}): Promise<D> {
  if (ref.kind === 'embedded') return ref.load()
  if (ref.kind === 'remote') return ref.parse(await fetchPinned(ref.name, ref, options))
  const files: Uint8Array[] = []
  for (const f of ref.files) files.push(await fetchPinned(ref.name, f, options))
  return ref.parse(files)
}

/** The bytes of a pinned file: from the cache when they still match, else downloaded, verified and cached. */
async function fetchPinned(name: string, file: RemoteFile, options: LoadOptions): Promise<Uint8Array> {
  const cached = await options.cache?.get(file.sha256)
  if (cached !== undefined && (await sha256Hex(cached)) === file.sha256) return cached
  const fetcher = options.fetch ?? (globalThis.fetch as Fetcher | undefined)
  if (fetcher === undefined) throw new WhiteboxError('not-implemented', `load: no fetch to download ${name}`)
  const response = await fetcher(file.url)
  if (!response.ok) throw new WhiteboxError('value', `load: downloading ${name} failed with status ${response.status}`)
  const bytes = new Uint8Array(await response.arrayBuffer())
  const digest = await sha256Hex(bytes)
  if (digest !== file.sha256)
    throw new WhiteboxError('value', `load: ${name} has checksum ${digest}, expected ${file.sha256}`)
  await options.cache?.set(file.sha256, bytes)
  return bytes
}
