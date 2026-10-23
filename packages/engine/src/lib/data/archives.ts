import { WhiteboxError } from '../base/index.ts'

async function streamThrough(bytes: Uint8Array, format: 'gzip' | 'deflate-raw'): Promise<Uint8Array> {
  const Decompression = (globalThis as { DecompressionStream?: typeof DecompressionStream }).DecompressionStream
  if (Decompression === undefined)
    throw new WhiteboxError('not-implemented', 'archives: DecompressionStream is not available here')
  const stream = new Blob([bytes.slice().buffer]).stream().pipeThrough(new Decompression(format))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

/**
 * Decompress gzip bytes with the platform's `DecompressionStream` (browsers and Node 18+).
 *
 * @param bytes - Gzip data.
 * @returns The decompressed bytes.
 * @throws WhiteboxError with code `not-implemented` without `DecompressionStream`.
 * @example
 * ```ts
 * import { gunzip } from 'whitebox-engine/data'
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
 * @example
 * ```ts
 * import { untar } from 'whitebox-engine/data'
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
 * @throws WhiteboxError with code `value` for a malformed archive or an unsupported compression method.
 * @example
 * ```ts
 * import { unzip } from 'whitebox-engine/data'
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
  if (end < 0) throw new WhiteboxError('value', 'unzip: no end of central directory record')
  const count = view.getUint16(end + 10, true)
  let p = view.getUint32(end + 16, true)
  const out = new Map<string, Uint8Array>()
  for (let e = 0; e < count; e++) {
    if (view.getUint32(p, true) !== 0x02014b50) throw new WhiteboxError('value', 'unzip: bad central directory entry')
    const method = view.getUint16(p + 10, true)
    const compressed = view.getUint32(p + 20, true)
    const nameLength = view.getUint16(p + 28, true)
    const extraLength = view.getUint16(p + 30, true)
    const commentLength = view.getUint16(p + 32, true)
    const local = view.getUint32(p + 42, true)
    const name = new TextDecoder().decode(bytes.subarray(p + 46, p + 46 + nameLength))
    p += 46 + nameLength + extraLength + commentLength
    if (name.endsWith('/')) continue
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true)
    const data = bytes.subarray(start, start + compressed)
    if (method === 0) out.set(name, data.slice())
    else if (method === 8) out.set(name, await streamThrough(data, 'deflate-raw'))
    else throw new WhiteboxError('value', `unzip: compression method ${method} of ${name} is not supported`)
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
 * import { parseCsv } from 'whitebox-engine/data'
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
