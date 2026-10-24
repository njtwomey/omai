import { describe, expect, it } from 'vitest'
import { gunzip, indexedDbCache, parseCsv, untar, unzip } from './index.ts'

const enc = new TextEncoder()

async function compress(bytes: Uint8Array, format: 'gzip' | 'deflate-raw'): Promise<Uint8Array> {
  const stream = new Blob([bytes.slice().buffer]).stream().pipeThrough(new CompressionStream(format))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

/** A zip archive with the given entries, each stored (method 0) or deflated (method 8). */
async function zip(entries: readonly { name: string; data: Uint8Array; deflate: boolean }[]): Promise<Uint8Array> {
  const locals: Uint8Array[] = []
  const centrals: Uint8Array[] = []
  let offset = 0
  for (const e of entries) {
    const body = e.deflate ? await compress(e.data, 'deflate-raw') : e.data
    const name = enc.encode(e.name)
    const local = new Uint8Array(30 + name.length + body.length)
    const lv = new DataView(local.buffer)
    lv.setUint32(0, 0x04034b50, true)
    lv.setUint16(8, e.deflate ? 8 : 0, true)
    lv.setUint32(18, body.length, true)
    lv.setUint32(22, e.data.length, true)
    lv.setUint16(26, name.length, true)
    local.set(name, 30)
    local.set(body, 30 + name.length)
    const central = new Uint8Array(46 + name.length)
    const cv = new DataView(central.buffer)
    cv.setUint32(0, 0x02014b50, true)
    cv.setUint16(10, e.deflate ? 8 : 0, true)
    cv.setUint32(20, body.length, true)
    cv.setUint32(24, e.data.length, true)
    cv.setUint16(28, name.length, true)
    cv.setUint32(42, offset, true)
    central.set(name, 46)
    locals.push(local)
    centrals.push(central)
    offset += local.length
  }
  const centralSize = centrals.reduce((s, c) => s + c.length, 0)
  const end = new Uint8Array(22)
  const ev = new DataView(end.buffer)
  ev.setUint32(0, 0x06054b50, true)
  ev.setUint16(8, entries.length, true)
  ev.setUint16(10, entries.length, true)
  ev.setUint32(12, centralSize, true)
  ev.setUint32(16, offset, true)
  const out = new Uint8Array(offset + centralSize + 22)
  let p = 0
  for (const part of [...locals, ...centrals, end]) {
    out.set(part, p)
    p += part.length
  }
  return out
}

describe('archives', () => {
  it('gunzip and untar', async () => {
    const body = enc.encode('hello tar')
    const header = new Uint8Array(512)
    header.set(enc.encode('dir/file.txt'), 0)
    header.set(enc.encode(`${body.length.toString(8).padStart(11, '0')}\0`), 124)
    header[156] = 48
    const tar = new Uint8Array(512 + 512 + 1024)
    tar.set(header, 0)
    tar.set(body, 512)
    const files = untar(await gunzip(await compress(tar, 'gzip')))
    expect([...files.keys()]).toEqual(['dir/file.txt'])
    expect(new TextDecoder().decode(files.get('dir/file.txt'))).toBe('hello tar')
  })
  it('unzip stored and deflated entries', async () => {
    const bytes = await zip([
      { name: 'a.txt', data: enc.encode('stored'), deflate: false },
      { name: 'b/c.csv', data: enc.encode('x,y\n1,2\n'.repeat(50)), deflate: true },
    ])
    const files = await unzip(bytes)
    expect(new TextDecoder().decode(files.get('a.txt'))).toBe('stored')
    expect(new TextDecoder().decode(files.get('b/c.csv')).length).toBe(400)
    await expect(unzip(new Uint8Array(30))).rejects.toThrow(/central directory/)
  })
  it('parse CSV with quotes, separators and line endings', () => {
    const t = parseCsv('name;note\r\n"Smith, J";"said ""hi""\nthen left"\r\nDoe;\n', { delimiter: ';' })
    expect(t.columns).toEqual(['name', 'note'])
    expect(t.rows).toEqual([
      ['Smith, J', 'said "hi"\nthen left'],
      ['Doe', ''],
    ])
    expect(parseCsv('1,2\n3,4', { header: false })).toEqual({
      columns: ['c0', 'c1'],
      rows: [
        ['1', '2'],
        ['3', '4'],
      ],
    })
  })
})

/** A minimal in-memory IndexedDB: one database, object stores of key-value pairs, requests that succeed asynchronously. */
function fakeIndexedDb(): IDBFactory {
  const stores = new Map<string, Map<string, unknown>>()
  const request = <T>(f: () => T): IDBRequest<T> => {
    const r = {
      result: undefined as T | undefined,
      error: null,
      onsuccess: null as null | (() => void),
      onerror: null,
    } as unknown as {
      result: T
      onsuccess: null | (() => void)
    }
    queueMicrotask(() => {
      r.result = f()
      r.onsuccess?.()
    })
    return r as unknown as IDBRequest<T>
  }
  const db = {
    objectStoreNames: { contains: (n: string) => stores.has(n) },
    createObjectStore: (n: string) => stores.set(n, new Map()),
    transaction: (n: string) => ({
      objectStore: () => {
        const store = stores.get(n) ?? new Map<string, unknown>()
        return {
          get: (k: string) => request(() => store.get(k)),
          put: (v: unknown, k: string) => request(() => store.set(k, v)),
          delete: (k: string) => request(() => store.delete(k)),
        }
      },
    }),
  }
  return {
    open: () => {
      const r = { result: db, onsuccess: null as null | (() => void), onupgradeneeded: null as null | (() => void) }
      queueMicrotask(() => {
        r.onupgradeneeded?.()
        r.onsuccess?.()
      })
      return r
    },
  } as unknown as IDBFactory
}

describe('IndexedDB cache', () => {
  it('stores, reads and deletes bytes', async () => {
    const cache = indexedDbCache({ factory: fakeIndexedDb() })
    await cache.set('k', new Uint8Array([1, 2, 3]))
    expect(Array.from((await cache.get('k')) ?? [])).toEqual([1, 2, 3])
    await cache.delete('k')
    expect(await cache.get('k')).toBeUndefined()
  })
})
