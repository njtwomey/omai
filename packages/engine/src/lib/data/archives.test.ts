import { describe, expect, it } from 'vitest'
import { gunzip, indexedDbCache, parseCsv, parseNpy, untar, unzip } from './index.ts'

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

const longName = `${'a'.repeat(60)}/${'b'.repeat(60)}/long-name.txt`
const fromBase64 = (text: string): Uint8Array => Uint8Array.from(atob(text), (c) => c.charCodeAt(0))

describe('archive readers against the files numpy and Python write', () => {
  // [what, .npy bytes written by numpy.save, shape, dtype descriptor, values in C order]
  const npyFiles: [string, string, number[], string, number[]][] = [
    [
      'float64 vector with NaN and infinities',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDUsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAD4PwAAAAAAAPh/AAAAAAAA8H8AAAAAAADw/5qZmZmZmbk/',
      [5],
      '<f8',
      [1.5, NaN, Infinity, -Infinity, 0.1],
    ],
    [
      'float32 vector',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIArNzMw9AAAgwHaE31A=',
      [3],
      '<f4',
      [0.10000000149011612, -2.5, 30000001024],
    ],
    [
      'int64 with negatives',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAP7/////////AAAAAAABAAA=',
      [3],
      '<i8',
      [1, -2, 1099511627776],
    ],
    [
      'int32 matrix',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAr9/////v////////8AAAAAAQAAAAIAAAA=',
      [2, 3],
      '<i4',
      [-3, -2, -1, 0, 1, 2],
    ],
    [
      'int16',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGkyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIArU/iwB',
      [2],
      '<i2',
      [-300, 300],
    ],
    [
      'int8',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnfGkxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAqAfw==',
      [2],
      '|i1',
      [-128, 127],
    ],
    [
      'uint64',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPHU4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAcAAAAAAQAA',
      [2],
      '<u8',
      [0, 1099511627783],
    ],
    [
      'uint32',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPHU0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAA/////w==',
      [2],
      '<u4',
      [0, 4294967295],
    ],
    [
      'uint16',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPHUyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoHAP//',
      [2],
      '<u2',
      [7, 65535],
    ],
    [
      'uint8',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoA/w==',
      [2],
      '|u1',
      [0, 255],
    ],
    [
      'bool',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnfGIxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAE=',
      [3],
      '|b1',
      [1, 0, 1],
    ],
    [
      'big-endian float64',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPmY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo/9AAAAAAAAMAeAAAAAAAA',
      [2],
      '>f8',
      [1.25, -7.5],
    ],
    [
      'big-endian int32',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPmk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAEC/////g==',
      [2],
      '>i4',
      [258, -2],
    ],
    [
      'a scalar (shape ())',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAASQA==',
      [],
      '<f8',
      [4.5],
    ],
    [
      'an empty vector',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDAsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=',
      [0],
      '<f8',
      [],
    ],
    [
      'an empty matrix',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDApLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=',
      [2, 0],
      '<f8',
      [],
    ],
    [
      'a 3-D C-order array',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDMsIDQpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAPA/AAAAAAAAAEAAAAAAAAAIQAAAAAAAABBAAAAAAAAAFEAAAAAAAAAYQAAAAAAAABxAAAAAAAAAIEAAAAAAAAAiQAAAAAAAACRAAAAAAAAAJkAAAAAAAAAoQAAAAAAAACpAAAAAAAAALEAAAAAAAAAuQAAAAAAAADBAAAAAAAAAMUAAAAAAAAAyQAAAAAAAADNAAAAAAAAANEAAAAAAAAA1QAAAAAAAADZAAAAAAAAAN0A=',
      [2, 3, 4],
      '<f8',
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    ],
    [
      'Fortran order 2-D',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBUcnVlLCAnc2hhcGUnOiAoMiwgMyksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAhAAAAAAAAA8D8AAAAAAAAQQAAAAAAAAABAAAAAAAAAFEA=',
      [2, 3],
      '<f8',
      [0, 1, 2, 3, 4, 5],
    ],
    [
      'Fortran order 3-D',
      'k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBUcnVlLCAnc2hhcGUnOiAoMiwgMywgNCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAADAAAAAQAAAAQAAAACAAAABQAAAABAAAADQAAAAUAAAARAAAACQAAABUAAAACAAAADgAAAAYAAAASAAAACgAAABYAAAADAAAADwAAAAcAAAATAAAACwAAABcAAAA=',
      [2, 3, 4],
      '<i4',
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    ],
    [
      'format version 2',
      'k05VTVBZAgB0AAAAeydkZXNjcic6ICc8ZjgnLCAnZm9ydHJhbl9vcmRlcic6IEZhbHNlLCAnc2hhcGUnOiAoMiwpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAADwPwAAAAAAAABA',
      [2],
      '<f8',
      [1, 2],
    ],
    [
      'format version 3',
      'k05VTVBZAwB0AAAAeydkZXNjcic6ICc8ZjgnLCAnZm9ydHJhbl9vcmRlcic6IEZhbHNlLCAnc2hhcGUnOiAoMiwpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAADwPwAAAAAAAABA',
      [2],
      '<f8',
      [1, 2],
    ],
  ]
  it('parseNpy reads every numeric dtype, byte order, memory order and format version as numpy wrote it, and parseCsv splits as Python csv does', () => {
    for (const [name, base64, shape, dtype, values] of npyFiles) {
      const got = parseNpy(fromBase64(base64))
      expect([got.shape, got.dtype, Array.from(got.values)], name).toEqual([shape, dtype, values])
    }
    expect(() => parseNpy(new Uint8Array([1, 2, 3, 4, 5, 6, 7]))).toThrow(expect.objectContaining({ code: 'value' }))
    // a complex dtype is not supported (the header of numpy.save(np.zeros(1, dtype=complex)), truncated to its data)
    const header = `${"{'descr': '<c16', 'fortran_order': False, 'shape': (1,), }".padEnd(117)}\n`
    const bytes = new Uint8Array(10 + header.length + 16)
    bytes.set([0x93, 78, 85, 77, 80, 89, 1, 0, header.length, 0])
    bytes.set(new TextEncoder().encode(header), 10)
    expect(() => parseNpy(bytes)).toThrow(/not supported/)

    for (const [text, delimiter, rows] of csvCases) {
      const got = parseCsv(text, { delimiter, header: false })
      expect(got.rows, JSON.stringify(text)).toEqual(rows)
    }
  })

  // [text, delimiter, rows]: Python's csv.reader on the same text, with blank lines dropped as the reader does
  const csvCases: [string, string, string[][]][] = [
    [
      'a,b,c\n1,2,3\n',
      ',',
      [
        ['a', 'b', 'c'],
        ['1', '2', '3'],
      ],
    ],
    [
      'x,y\n"1,5","he said ""hi"""\n',
      ',',
      [
        ['x', 'y'],
        ['1,5', 'he said "hi"'],
      ],
    ],
    [
      'a,b\r\n1,2\r\n3,4',
      ',',
      [
        ['a', 'b'],
        ['1', '2'],
        ['3', '4'],
      ],
    ],
    [
      'a,b\n"multi\nline",2\n',
      ',',
      [
        ['a', 'b'],
        ['multi\nline', '2'],
      ],
    ],
    [
      'a,b\n,\n1,\n',
      ',',
      [
        ['a', 'b'],
        ['', ''],
        ['1', ''],
      ],
    ],
    [
      'a;b\n1;"2;3"\n',
      ';',
      [
        ['a', 'b'],
        ['1', '2;3'],
      ],
    ],
    [
      'a,b\n1,2\n\n3,4\n\n',
      ',',
      [
        ['a', 'b'],
        ['1', '2'],
        ['3', '4'],
      ],
    ],
    [
      '"q""x",b\n1,2',
      ',',
      [
        ['q"x', 'b'],
        ['1', '2'],
      ],
    ],
    [
      'a,b\n "1",2\n',
      ',',
      [
        ['a', 'b'],
        [' "1"', '2'],
      ],
    ],
    ['a,b\n1,2,3\n4\n', ',', [['a', 'b'], ['1', '2', '3'], ['4']]],
  ]

  it('untar reads the regular files of a tar.gz Python wrote: prefixes, directories, empty and block-aligned files', async () => {
    const files = untar(
      await gunzip(
        fromBase64(
          'H4sIAAAAAAAC/yvJL9ArqShhoCUwAAIzExMwDQTotKGROVwOKm5ubGjCoGDAQAdQWlySWAS0kmFkgopRMAqGKGAYBRSDlMwifVrbQaD8x8I2MzUyYFAwHS3/6RL/mXl5qUV6SZl5AxT/Jujxb2hgbGQ0Wv/TBTAyMbOwsrFzcHJx8/Dy8QsICgmLiIqJS0hKScvIyskrKCopq6iqqWtoamnr6OrpA1trxiamZuYWllbWNrZ29g6OTs4urm7uHp5e3j6+fv4BgUHBIaFh4RGRUdExsXHxCYlJySmpaekZmVnZObl5+QWFRcUlpWXlFZVV1TW1dfUNjU3NLa1t7R2dXd09vX39EyZOmjxl6rTpM2bOmj1n7rz5CxYuWrxk6bLlK1auWr1m7br1GzZu2rxl67btO3bu2r1n7779Bw4eOnzk6LHjJ06eOn3m7LnzFy5eunzl6rXrN27eun3n7r37Dx4+evzk6bPnL16+ev3m7bv3Hz5++vzl67fvP37++v3n77//I738T6IA6Ofk56Xr5iXmpuLvQhAs/w3R2v/G5sZGhkM8/ydSAOgY/wVFqWmZFQpF+aUlqaOtoZEHUnMLSippPAJAevvP3Mx4tP4fBaNgFIyCUTAKRsEoGAWjYBSMglEwCkbBKBgFo2AUjIJRMArIBgBrYGCMACgAAA==',
        ),
      ),
    )
    expect([...files.keys()].sort()).toEqual([longName, 'dir/inner.bin', 'empty.txt', 'top.txt'])
    expect(Array.from(files.get('dir/inner.bin') ?? [])).toEqual(Array.from({ length: 256 }, (_, i) => i))
    expect([files.get('top.txt')?.length, files.get('empty.txt')?.length]).toEqual([700, 0])
    expect(new TextDecoder().decode(files.get(longName))).toBe('prefix route')
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
