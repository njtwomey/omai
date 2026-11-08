/**
 * Readers for the file formats datasets arrive in: gzip, tar and zip archives, delimited text (RFC 4180 CSV with quoted
 * fields) and NumPy `.npy` arrays. They work on raw bytes with platform APIs only, so remote datasets can be unpacked
 * and parsed in the browser or in Node.
 */

import { OmaiError } from '../base/index.ts'

async function streamThrough(bytes: Uint8Array, format: 'gzip' | 'deflate-raw'): Promise<Uint8Array> {
  const Decompression = (globalThis as { DecompressionStream?: typeof DecompressionStream }).DecompressionStream
  if (Decompression === undefined)
    throw new OmaiError('not-implemented', 'archives: DecompressionStream is not available here')
  const stream = new Blob([bytes.slice().buffer]).stream().pipeThrough(new Decompression(format))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

/**
 * Decompress gzip bytes with the platform's `DecompressionStream` (browsers and Node 18+).
 *
 * @param bytes - Gzip data.
 * @returns The decompressed bytes.
 * @throws OmaiError with code `not-implemented` without `DecompressionStream`.
 * @example
 * ```ts
 * import { gunzip } from 'omai-engine/data'
 * const packed = new Uint8Array(await new Response(new Blob(['hi']).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer())
 * new TextDecoder().decode(await gunzip(packed)) // => 'hi'
 * ```
 */
export function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  return streamThrough(bytes, 'gzip')
}

function text(bytes: Uint8Array, start: number, length: number): string {
  let end = start
  while (end < start + length && bytes[end] !== 0) end++
  return new TextDecoder().decode(bytes.subarray(start, end))
}

/**
 * The regular files of a POSIX (ustar) tar archive by path.
 *
 * @param bytes - The archive.
 * @returns A map from file path to contents.
 * @throws OmaiError with code `value` for a header whose size is not an octal number, or a file the archive is too short
 *   to hold.
 * @example
 * ```ts
 * import { untar } from 'omai-engine/data'
 * untar(new Uint8Array(1024)).size // => 0
 * ```
 */
export function untar(bytes: Uint8Array): Map<string, Uint8Array> {
  const out = new Map<string, Uint8Array>()
  let offset = 0
  while (offset + 512 <= bytes.length) {
    const name = text(bytes, offset, 100)
    if (name === '') break
    const size = Number.parseInt(text(bytes, offset + 124, 12).trim() || '0', 8)
    if (!(size >= 0)) throw new OmaiError('value', `untar: the size of ${name} is not an octal number`)
    if (offset + 512 + size > bytes.length) throw new OmaiError('value', `untar: the archive ends inside ${name}`)
    const type = bytes[offset + 156]
    const prefix = text(bytes, offset + 345, 155)
    const path = prefix === '' ? name : `${prefix}/${name}`
    // '0' or NUL is a regular file.
    if (type === 48 || type === 0) out.set(path, bytes.slice(offset + 512, offset + 512 + size))
    offset += 512 + Math.ceil(size / 512) * 512
  }
  return out
}

/**
 * The files of a zip archive by path (stored or deflated entries), read from the central directory.
 *
 * @param bytes - The archive.
 * @returns A map from file path to contents.
 * @throws OmaiError with code `value` for a malformed or cut-short archive or an unsupported compression method.
 * @example
 * ```ts
 * import { unzip } from 'omai-engine/data'
 * typeof unzip // => 'function'
 * ```
 */
export async function unzip(bytes: Uint8Array): Promise<Map<string, Uint8Array>> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let end = -1
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65_557); i--) {
    if (view.getUint32(i, true) === 0x06054b50) {
      end = i
      break
    }
  }
  if (end < 0) throw new OmaiError('value', 'unzip: no end of central directory record')
  const count = view.getUint16(end + 10, true)
  let p = view.getUint32(end + 16, true)
  const out = new Map<string, Uint8Array>()
  for (let e = 0; e < count; e++) {
    if (view.getUint32(p, true) !== 0x02014b50) throw new OmaiError('value', 'unzip: bad central directory entry')
    const method = view.getUint16(p + 10, true)
    const compressed = view.getUint32(p + 20, true)
    const nameLength = view.getUint16(p + 28, true)
    const extraLength = view.getUint16(p + 30, true)
    const commentLength = view.getUint16(p + 32, true)
    const local = view.getUint32(p + 42, true)
    const name = new TextDecoder().decode(bytes.subarray(p + 46, p + 46 + nameLength))
    p += 46 + nameLength + extraLength + commentLength
    if (name.endsWith('/')) continue
    if (local + 30 > bytes.length)
      throw new OmaiError('value', `unzip: the local header of ${name} is outside the archive`)
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true)
    if (start + compressed > bytes.length) throw new OmaiError('value', `unzip: the archive ends inside ${name}`)
    const data = bytes.subarray(start, start + compressed)
    if (method === 0) out.set(name, data.slice())
    else if (method === 8) out.set(name, await streamThrough(data, 'deflate-raw'))
    else throw new OmaiError('value', `unzip: compression method ${method} of ${name} is not supported`)
  }
  return out
}

/** Options of {@link parseCsv}. */
export interface CsvOptions {
  /** The field separator (default `,`). */
  readonly delimiter?: string
  /** Whether the first row names the columns (default true). */
  readonly header?: boolean
}

/**
 * Parse delimited text (RFC 4180: quoted fields may contain separators, newlines and doubled quotes).
 *
 * @param source - The text.
 * @param options - Delimiter and header.
 * @returns The column names (generated `c0, c1, ...` without a header) and the rows of fields.
 * @example
 * ```ts
 * import { parseCsv } from 'omai-engine/data'
 * parseCsv('a,b\n1,"x, y"\n').rows // => [['1', 'x, y']]
 * ```
 */
export function parseCsv(
  source: string,
  options: CsvOptions = {},
): { readonly columns: readonly string[]; readonly rows: readonly (readonly string[])[] } {
  const sep = options.delimiter ?? ','
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  for (let i = 0; i < source.length; i++) {
    const ch = source[i] as string
    if (quoted) {
      if (ch === '"' && source[i + 1] === '"') {
        field += '"'
        i++
      } else if (ch === '"') quoted = false
      else field += ch
    } else if (ch === '"' && field === '') quoted = true
    else if (ch === sep) {
      row.push(field)
      field = ''
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && source[i + 1] === '\n') i++
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else field += ch
  }
  if (field !== '' || row.length > 0) {
    row.push(field)
    rows.push(row)
  }
  const body = rows.filter((r) => !(r.length === 1 && r[0] === ''))
  if (options.header === false) {
    const width = body[0]?.length ?? 0
    return { columns: Array.from({ length: width }, (_, j) => `c${j}`), rows: body }
  }
  return { columns: body[0] ?? [], rows: body.slice(1) }
}

const NPY_READERS: Readonly<
  Record<string, { size: number; read: (v: DataView, o: number, little: boolean) => number }>
> = {
  f8: { size: 8, read: (v, o, l) => v.getFloat64(o, l) },
  f4: { size: 4, read: (v, o, l) => v.getFloat32(o, l) },
  i8: { size: 8, read: (v, o, l) => Number(v.getBigInt64(o, l)) },
  i4: { size: 4, read: (v, o, l) => v.getInt32(o, l) },
  i2: { size: 2, read: (v, o, l) => v.getInt16(o, l) },
  i1: { size: 1, read: (v, o) => v.getInt8(o) },
  u8: { size: 8, read: (v, o, l) => Number(v.getBigUint64(o, l)) },
  u4: { size: 4, read: (v, o, l) => v.getUint32(o, l) },
  u2: { size: 2, read: (v, o, l) => v.getUint16(o, l) },
  u1: { size: 1, read: (v, o) => v.getUint8(o) },
  b1: { size: 1, read: (v, o) => v.getUint8(o) },
}

/**
 * Read a NumPy `.npy` array (format versions 1 to 3, numeric dtypes, C or Fortran order) as float64 values in C order.
 *
 * @param bytes - The file.
 * @returns The shape, the dtype descriptor and the values.
 * @throws OmaiError with code `value` for a file that is not `.npy`, an unsupported dtype, a shape that is not a list of
 *   whole numbers, or a file shorter than its header says.
 * @example
 * ```ts
 * import { parseNpy } from 'omai-engine/data'
 * const header = "{'descr': '<u2', 'fortran_order': False, 'shape': (2,), }"
 * const pad = ' '.repeat(64 - ((10 + header.length + 1) % 64)) + '\n'
 * const text = new TextEncoder().encode(header + pad)
 * const bytes = new Uint8Array(10 + text.length + 4)
 * bytes.set([0x93, 78, 85, 77, 80, 89, 1, 0, text.length & 255, text.length >> 8])
 * bytes.set(text, 10)
 * bytes.set([7, 0, 9, 1], 10 + text.length)
 * Array.from(parseNpy(bytes).values) // => [7, 265]
 * ```
 */
export function parseNpy(bytes: Uint8Array): {
  readonly shape: readonly number[]
  readonly dtype: string
  readonly values: Float64Array
} {
  if (bytes[0] !== 0x93 || new TextDecoder().decode(bytes.subarray(1, 6)) !== 'NUMPY')
    throw new OmaiError('value', 'parseNpy: not a .npy file')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const major = bytes[6] ?? 1
  const start = major === 1 ? 10 : 12
  if (bytes.length < start) throw new OmaiError('value', 'parseNpy: the file ends inside the preamble')
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
  if (start + headerLength > bytes.length) throw new OmaiError('value', 'parseNpy: the file ends inside the header')
  const header = new TextDecoder().decode(bytes.subarray(start, start + headerLength))
  const descr = /'descr':\s*'([<>|=])(\w)(\d+)'/.exec(header)
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? ''
  const fortran = /'fortran_order':\s*True/.test(header)
  if (descr === null) throw new OmaiError('value', 'parseNpy: no dtype in the header')
  const [, order, kind, width] = descr as unknown as [string, string, string, string]
  const reader = NPY_READERS[`${kind}${width}`]
  if (reader === undefined) throw new OmaiError('value', `parseNpy: dtype ${kind}${width} is not supported`)
  const shape = shapeText
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s !== '')
    .map(Number)
  if (!shape.every((s) => Number.isInteger(s) && s >= 0))
    throw new OmaiError('value', `parseNpy: the shape (${shapeText}) is not a list of whole numbers`)
  const count = shape.reduce((a, b) => a * b, 1)
  const little = order !== '>'
  const offset = start + headerLength
  if (offset + count * reader.size > bytes.length)
    throw new OmaiError(
      'value',
      `parseNpy: the file holds ${bytes.length - offset} bytes of data, ${count * reader.size} needed`,
    )
  const raw = Float64Array.from({ length: count }, (_, i) => reader.read(view, offset + i * reader.size, little))
  if (!fortran || shape.length < 2) return { shape, dtype: `${order}${kind}${width}`, values: raw }
  // Fortran order: the first index varies fastest.
  const values = new Float64Array(count)
  const strides = shape.map((_, a) => shape.slice(0, a).reduce((p, q) => p * q, 1))
  const cStrides = shape.map((_, a) => shape.slice(a + 1).reduce((p, q) => p * q, 1))
  for (let i = 0; i < count; i++) {
    let rest = i
    let target = 0
    for (let a = 0; a < shape.length; a++) {
      const idx = Math.floor(rest / (cStrides[a] ?? 1))
      rest -= idx * (cStrides[a] ?? 1)
      target += idx * (strides[a] ?? 1)
    }
    values[i] = raw[target] ?? 0
  }
  return { shape, dtype: `${order}${kind}${width}`, values }
}
