import { expect, describe as group, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type AssetCache, type Fetcher, load, memoryCache, sha256Hex, xy } from './index.ts'

const enc = new TextEncoder()
/** A fetcher over a table of urls that counts its calls (a missing url is a 404). */
function fetcherOf(files: Readonly<Record<string, string>>): { fetch: Fetcher; calls: string[] } {
  const calls: string[] = []
  const fetch: Fetcher = async (url) => {
    calls.push(url)
    const body = files[url]
    return {
      ok: body !== undefined,
      status: body === undefined ? 404 : 200,
      arrayBuffer: async () => enc.encode(body ?? '').buffer,
    }
  }
  return { fetch, calls }
}
const code = async (f: () => Promise<unknown>): Promise<string> => {
  try {
    await f()
  } catch (e) {
    return isOmaiError(e) ? e.code : `not an OmaiError: ${String(e)}`
  }
  return 'returned'
}

group('loading datasets by reference', () => {
  it('loads several pinned files in order, verifies the cache it reads, and says what it could not do', async () => {
    const files = { 'https://x/a.csv': '1,2\n', 'https://x/b.csv': '3,4\n' }
    const pin = async (url: keyof typeof files): Promise<{ url: string; sha256: string }> => ({
      url,
      sha256: await sha256Hex(enc.encode(files[url])),
    })
    const [a, b] = [await pin('https://x/a.csv'), await pin('https://x/b.csv')]
    const parse = (parts: readonly Uint8Array[]) => {
      const rows = parts.map((p) => new TextDecoder().decode(p).trim().split(',').map(Number))
      return xy(
        rows.map((r) => [r[0] ?? 0]),
        rows.map((r) => r[1] ?? 0),
      )
    }
    const ref = { kind: 'remote-files' as const, name: 'two files', files: [a, b], parse }
    const { fetch, calls } = fetcherOf(files)
    const cache = memoryCache()
    expect(toFlat((await load(ref, { fetch, cache })).y)).toEqual([2, 4])
    expect(calls).toEqual([a.url, b.url])
    // A second load comes from the cache; an entry whose bytes no longer match its checksum is fetched again and replaced.
    await load(ref, { fetch, cache })
    expect(calls.length).toBe(2)
    await cache.set(a.sha256, enc.encode('tampered'))
    expect(toFlat((await load(ref, { fetch, cache })).y)).toEqual([2, 4])
    expect(calls).toEqual([a.url, b.url, a.url])
    expect(await sha256Hex((await cache.get(a.sha256)) ?? new Uint8Array())).toBe(a.sha256)
    // Failures: the second file missing (nothing is cached for it), a wrong pin; and the digest of nothing.
    const gone = fetcherOf({ 'https://x/a.csv': '1,2\n' })
    const fresh: AssetCache = memoryCache()
    expect(await code(() => load(ref, { fetch: gone.fetch, cache: fresh }))).toBe('value')
    expect(await fresh.get(b.sha256)).toBeUndefined()
    expect(await code(() => load({ ...ref, files: [{ ...a, sha256: '0'.repeat(64) }] }, { fetch }))).toBe('value')
    expect(await sha256Hex(new Uint8Array())).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
  })
})
