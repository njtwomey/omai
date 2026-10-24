// The Sobol' engine behind sequences.ts: direction numbers (Joe and Kuo, new-joe-kuo-6.21201), the linear matrix
// scramble and Gray-code generation, all in unsigned 32-bit integer arithmetic. Private to random/.
import { WhiteboxError } from '../base/index.ts'
import { SOBOL_DIMENSIONS, SOBOL_M, SOBOL_POLY } from './sobol-directions.data.ts'

const BASE64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

function decodeBase64(text: string): Uint8Array {
  const clean = text.replace(/=+$/, '')
  const out = new Uint8Array(Math.floor((clean.length * 6) / 8))
  let acc = 0
  let nbits = 0
  let o = 0
  for (let i = 0; i < clean.length; i++) {
    acc = (acc << 6) | BASE64.indexOf(clean.charAt(i))
    nbits += 6
    if (nbits >= 8) {
      nbits -= 8
      out[o++] = (acc >>> nbits) & 0xff
    }
  }
  return out
}

/** Initial direction numbers m_1..m_s of every dimension (dimension 0 has none), decoded from the packed table. */
function decodeInitial(): Int32Array[] {
  const bytes = decodeBase64(SOBOL_M)
  let pos = 0
  const read = (width: number): number => {
    let w = 0
    for (let b = 0; b < width; b++, pos++) w |= (((bytes[pos >> 3] ?? 0) >> (pos & 7)) & 1) << b
    return w
  }
  const out: Int32Array[] = [new Int32Array(0)]
  for (let d = 1; d < SOBOL_DIMENSIONS; d++) {
    const degree = Math.floor(Math.log2(SOBOL_POLY[d] ?? 1))
    const m = new Int32Array(degree)
    for (let i = 0; i < degree; i++) m[i] = 2 * read(i) + 1
    out.push(m)
  }
  return out
}

const INITIAL: readonly Int32Array[] = decodeInitial()

/**
 * Direction numbers `v[j]` (j = 0..bits-1) of dimension `dim`, as integers scaled by 2^bits (the `_initialize_v`
 * recurrence of Bratley and Fox, as in scipy).
 */
export function directions(dim: number, bits: number): Uint32Array {
  const v = new Uint32Array(bits)
  if (dim === 0) {
    for (let j = 0; j < bits; j++) v[j] = 2 ** (bits - 1 - j)
    return v
  }
  const poly = SOBOL_POLY[dim] ?? 1
  const m = INITIAL[dim] ?? new Int32Array(0)
  const s = m.length
  // Work with m_j (odd, below 2^(j+1)) and scale at the end; m_j < 2^bits <= 2^32, so XORs stay in uint32.
  const w = new Uint32Array(bits)
  for (let j = 0; j < Math.min(s, bits); j++) w[j] = m[j] ?? 1
  for (let j = s; j < bits; j++) {
    let next = w[j - s] ?? 0
    for (let k = 0; k < s; k++) {
      if ((poly >> (s - 1 - k)) & 1) next = (next ^ ((w[j - k - 1] ?? 0) * 2 ** (k + 1))) >>> 0
    }
    w[j] = next
  }
  for (let j = 0; j < bits; j++) v[j] = ((w[j] ?? 0) * 2 ** (bits - 1 - j)) >>> 0
  return v
}

function parity(x: number): number {
  let y = x ^ (x >>> 16)
  y ^= y >>> 8
  y ^= y >>> 4
  y ^= y >>> 2
  y ^= y >>> 1
  return y & 1
}

/**
 * Matousek's linear matrix scramble of one dimension's direction numbers: each output digit is its own input digit
 * plus a random combination of the more significant ones (a random lower-triangular matrix with unit diagonal over
 * GF(2)), which preserves the net structure. `random` supplies one word per digit.
 */
export function scrambleDirections(v: Uint32Array, bits: number, random: Uint32Array): Uint32Array {
  const rows = new Uint32Array(bits)
  const mask = bits >= 32 ? 0xffffffff : 2 ** bits - 1
  for (let b = 0; b < bits; b++) {
    // Row for the digit at bit position b: bit b itself and random bits above it (the more significant digits).
    const above = b >= 31 ? 0 : 0xffffffff << (b + 1)
    rows[b] = ((random[b] ?? 0) & above & mask) | (1 << b)
  }
  const out = new Uint32Array(v.length)
  for (let j = 0; j < v.length; j++) {
    let t = 0
    const vj = v[j] ?? 0
    for (let b = 0; b < bits; b++) if (parity(vj & (rows[b] ?? 0))) t |= 1 << b
    out[j] = t
  }
  return out
}

/** Validate the size, dimension, offset and bit count of a Sobol' request. */
export function checkSobol(name: string, n: number, d: number, skip: number, bits: number): void {
  if (!Number.isInteger(bits) || bits < 1 || bits > 32) {
    throw new WhiteboxError('value', `${name}: bits must be an integer in [1, 32], got ${bits}`)
  }
  if (!Number.isInteger(d) || d < 1 || d > SOBOL_DIMENSIONS) {
    throw new WhiteboxError('value', `${name}: dimension must be an integer in [1, ${SOBOL_DIMENSIONS}], got ${d}`)
  }
  if (!Number.isInteger(n) || n < 0 || !Number.isInteger(skip) || skip < 0) {
    throw new WhiteboxError('value', `${name}: n and skip must be non-negative integers`)
  }
  if (skip + n > 2 ** bits) {
    throw new WhiteboxError('value', `${name}: at most 2^${bits} points, asked for indices up to ${skip + n - 1}`)
  }
}

/** Index of the lowest zero bit of a non-negative integer below 2^53. */
function lowestZeroBit(i: number): number {
  let c = 0
  let t = i
  while (t % 2 === 1) {
    t = (t - 1) / 2
    c++
  }
  return c
}

/**
 * Points `skip..skip+n-1` of the digital sequence with per-dimension direction numbers `v[dim]` and digital shift
 * `shift[dim]`, in Gray-code order: point i is the shift XOR the direction numbers at the set bits of gray(i).
 */
export function drawSobol(
  v: readonly Uint32Array[],
  shift: Uint32Array,
  n: number,
  skip: number,
  bits: number,
): Float64Array {
  const d = v.length
  const out = new Float64Array(n * d)
  const quasi = Uint32Array.from(shift)
  // gray(skip) = skip XOR floor(skip / 2), bit by bit so indices up to 2^32 work.
  for (let j = 0, t = skip; t > 0; j++, t = Math.floor(t / 2)) {
    const bit = (t % 2) ^ (Math.floor(t / 2) % 2)
    if (bit) for (let k = 0; k < d; k++) quasi[k] = ((quasi[k] ?? 0) ^ (v[k]?.[j] ?? 0)) >>> 0
  }
  const scale = 2 ** -bits
  for (let i = 0; i < n; i++) {
    if (i > 0) {
      const l = lowestZeroBit(skip + i - 1)
      for (let k = 0; k < d; k++) quasi[k] = ((quasi[k] ?? 0) ^ (v[k]?.[l] ?? 0)) >>> 0
    }
    for (let k = 0; k < d; k++) out[i * d + k] = (quasi[k] ?? 0) * scale
  }
  return out
}
