/**
 * Reed-Solomon codes, the burst-error-correcting codes of CDs, QR codes and storage, built on arithmetic in the finite
 * field GF(2^m): the field itself, systematic encoding, and decoding of errors and erasures by syndromes, the
 * Berlekamp-Massey algorithm, a Chien search and Forney's formula.
 */

import { OmaiError } from '../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'

/** Arithmetic in the finite field GF(2^m), by exponent and logarithm tables of a primitive element. */
export interface GaloisField {
  readonly m: number
  /** 2^m. */
  readonly size: number
  /** The primitive polynomial, as an integer (bit i is the coefficient of x^i). */
  readonly primitive: number
  /** exp[i] = alpha^i, for i in [0, 2 (size - 1)). */
  readonly exp: Int32Array
  /** log[x] for x != 0. */
  readonly log: Int32Array
  add(a: number, b: number): number
  mul(a: number, b: number): number
  div(a: number, b: number): number
  pow(a: number, e: number): number
  inverse(a: number): number
}

/** Conway-style primitive polynomials (the defaults of most Reed-Solomon libraries) for m = 2 ... 16. */
const PRIMITIVE: Readonly<Record<number, number>> = {
  2: 0x7,
  3: 0xb,
  4: 0x13,
  5: 0x25,
  6: 0x43,
  7: 0x89,
  8: 0x11d,
  9: 0x211,
  10: 0x409,
  11: 0x805,
  12: 0x1053,
  13: 0x201b,
  14: 0x4443,
  15: 0x8003,
  16: 0x1100b,
}

/**
 * The finite field GF(2^m): elements are m-bit integers (polynomials over GF(2)), addition is XOR and multiplication is
 * polynomial multiplication modulo a primitive polynomial, done through the powers of the primitive element alpha = x.
 *
 * @param m - Bits per element (2 to 16; default 8, as for bytes).
 * @param primitive - The primitive polynomial (default 0x11d for m = 8, the usual choice for Reed-Solomon).
 * @returns The field.
 * @throws OmaiError with code `value` when the polynomial is not primitive.
 * @concept finite-fields
 * @example
 * ```ts
 * import { galoisField } from 'omai-engine/information'
 * const f = galoisField(8)
 * f.mul(2, 0x8e) // => 1
 * f.mul(f.inverse(7), 7) // => 1
 * ```
 */
export function galoisField(m = 8, primitive?: number): GaloisField {
  if (!Number.isInteger(m) || m < 2 || m > 16) throw new OmaiError('value', 'galoisField: m must be in [2, 16]')
  const prim = primitive ?? PRIMITIVE[m] ?? 0
  const size = 1 << m
  const exp = new Int32Array(2 * size)
  const log = new Int32Array(size).fill(-1)
  let x = 1
  for (let i = 0; i < size - 1; i++) {
    if (log[x] !== -1) throw new OmaiError('value', `galoisField: 0x${prim.toString(16)} is not primitive`)
    exp[i] = x
    log[x] = i
    x <<= 1
    if (x & size) x ^= prim
  }
  if (x !== 1) throw new OmaiError('value', `galoisField: 0x${prim.toString(16)} is not primitive`)
  for (let i = size - 1; i < 2 * size; i++) exp[i] = exp[i - (size - 1)] ?? 0
  const order = size - 1
  const mul = (a: number, b: number): number => (a === 0 || b === 0 ? 0 : (exp[(log[a] ?? 0) + (log[b] ?? 0)] ?? 0))
  const div = (a: number, b: number): number => {
    if (b === 0) throw new OmaiError('value', 'galoisField: division by zero')
    return a === 0 ? 0 : (exp[((log[a] ?? 0) - (log[b] ?? 0) + order) % order] ?? 0)
  }
  return {
    m,
    size,
    primitive: prim,
    exp,
    log,
    add: (a, b) => a ^ b,
    mul,
    div,
    pow: (a, e) => {
      if (a === 0) return e === 0 ? 1 : 0
      return exp[((((log[a] ?? 0) * e) % order) + order) % order] ?? 0
    },
    inverse: (a) => div(1, a),
  }
}

/** A Reed-Solomon code over GF(2^m) with nsym parity symbols. */
export interface ReedSolomonCode {
  readonly field: GaloisField
  readonly nsym: number
  /** First consecutive root: the generator is the product of (x - alpha^(fcr + i)) for i < nsym. */
  readonly fcr: number
  /** Generator polynomial coefficients, highest degree first (monic). */
  readonly generator: readonly number[]
}

/**
 * A Reed-Solomon code (Reed and Solomon 1960) with nsym parity symbols over GF(2^m): codewords are the multiples of
 * g(x) = (x - alpha^fcr) ... (x - alpha^(fcr + nsym - 1)), up to 2^m - 1 symbols long; it corrects any e errors and s
 * erasures with 2e + s <= nsym. The conventions (systematic, parity last, highest degree first) are those of the
 * reedsolo library.
 *
 * @param nsym - Parity symbols (1 to 2^m - 2).
 * @param options - Field size m (default 8), primitive polynomial and first consecutive root (default 0).
 * @returns The code.
 * @concept reed-solomon
 * @example
 * ```ts
 * import { reedSolomon } from 'omai-engine/information'
 * reedSolomon(2).generator // => [1, 3, 2]
 * ```
 */
export function reedSolomon(
  nsym: number,
  options: { readonly m?: number; readonly primitive?: number; readonly fcr?: number } = {},
): ReedSolomonCode {
  const field = galoisField(options.m ?? 8, options.primitive)
  if (!Number.isInteger(nsym) || nsym < 1 || nsym >= field.size - 1)
    throw new OmaiError('value', `reedSolomon: nsym must be an integer in [1, ${field.size - 2}]`)
  const fcr = options.fcr ?? 0
  let g = [1]
  for (let i = 0; i < nsym; i++) {
    const root = field.pow(2, i + fcr)
    const next = new Array<number>(g.length + 1).fill(0)
    g.forEach((c, j) => {
      next[j] = (next[j] ?? 0) ^ c
      next[j + 1] = (next[j + 1] ?? 0) ^ field.mul(c, root)
    })
    g = next
  }
  return { field, nsym, fcr, generator: g }
}

function symbols(code: ReedSolomonCode, xs: ArrayLike<number>, name: string): number[] {
  const out = Array.from(xs)
  if (out.some((v) => !Number.isInteger(v) || v < 0 || v >= code.field.size))
    throw new OmaiError('value', `${name}: symbols must be integers in [0, ${code.field.size - 1}]`)
  return out
}

/**
 * Systematic Reed-Solomon encoding: the message followed by the remainder of message(x) x^nsym divided by g(x), so
 * that the codeword is a multiple of g. Messages shorter than 2^m - 1 - nsym give shortened codes.
 *
 * @param code - The code.
 * @param message - Up to 2^m - 1 - nsym symbols.
 * @returns The codeword (message then parity).
 * @concept reed-solomon
 * @example
 * ```ts
 * import { reedSolomon, rsEncode } from 'omai-engine/information'
 * rsEncode(reedSolomon(4), [1, 2, 3, 4]).slice(4) // => [117, 163, 178, 96]
 * ```
 */
export function rsEncode(code: ReedSolomonCode, message: ArrayLike<number>): number[] {
  const msg = symbols(code, message, 'rsEncode')
  if (msg.length + code.nsym > code.field.size - 1)
    throw new OmaiError('value', `rsEncode: at most ${code.field.size - 1 - code.nsym} message symbols`)
  const g = code.generator
  const out = [...msg, ...new Array<number>(code.nsym).fill(0)]
  for (let i = 0; i < msg.length; i++) {
    const coef = out[i] ?? 0
    if (coef === 0) continue
    for (let j = 1; j < g.length; j++) out[i + j] = (out[i + j] ?? 0) ^ code.field.mul(g[j] ?? 0, coef)
  }
  return [...msg, ...out.slice(msg.length)]
}

/** Evaluate a polynomial with coefficients lowest degree first. */
function evalLow(f: GaloisField, p: readonly number[], x: number): number {
  let y = 0
  for (let i = p.length - 1; i >= 0; i--) y = f.mul(y, x) ^ (p[i] ?? 0)
  return y
}

/** Multiply polynomials with coefficients lowest degree first. */
function mulLow(f: GaloisField, a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(a.length + b.length - 1).fill(0)
  a.forEach((x, i) => {
    b.forEach((y, j) => {
      out[i + j] = (out[i + j] ?? 0) ^ f.mul(x, y)
    })
  })
  return out
}

/** The outcome of Reed-Solomon decoding. */
export interface RsDecodeResult {
  readonly message: number[]
  readonly codeword: number[]
  /** Positions (indices into the received word) that were corrected, errors and erasures, sorted. */
  readonly errata: number[]
}

/** The paths {@link rsDecodeAlgorithm} logs. */
export type RsPath = 'rs/syndrome' | 'rs/locator' | 'rs/discrepancy' | 'rs/errors'

/**
 * What the Reed-Solomon decoder logs: the syndromes, the errata locator polynomial after each Berlekamp-Massey step, the
 * discrepancy that step corrected, and the number of errors the locator implies (at the last step, the number found).
 */
export const rsContract: Contract<unknown, RsPath> = defineContract({
  'rs/syndrome': tensor({ role: 'state', shape: ['nsym'], description: 'the syndromes of the received word' }),
  'rs/locator': tensor({ role: 'state', shape: ['degree'], description: 'the errata locator, lowest degree first' }),
  'rs/discrepancy': scalar({
    description: 'the discrepancy of the Berlekamp-Massey step (0: the locator already fits)',
  }),
  'rs/errors': scalar({
    description: 'the errors (beyond the erasures) the locator implies; at the end, the errata found',
  }),
})

/**
 * The state of {@link rsDecodeAlgorithm}: the syndromes, the errata locator being built by Berlekamp-Massey and the
 * correction register, then the corrected word. Steps replace the arrays; they serialise as plain numbers.
 */
export interface RsDecodeState {
  readonly received: readonly number[]
  readonly erasures: readonly number[]
  readonly syndromes: readonly number[]
  /** The next Berlekamp-Massey step (from `erasures + 1` to `nsym`). */
  readonly step: number
  /** The errata locator, lowest degree first. */
  readonly lambda: readonly number[]
  /** The correction register of the algorithm. */
  readonly b: readonly number[]
  /** The current length of the locator. */
  readonly l: number
  /** The last discrepancy. */
  readonly discrepancy: number
  /** `bm` while stepping, `correct` once the locator is final and the errata remain to be found and fixed, `done` after. */
  readonly phase: 'bm' | 'correct' | 'done'
  /** The corrected word (the received word until `done`). */
  readonly corrected: readonly number[]
  /** The corrected positions, sorted. */
  readonly errata: readonly number[]
  /** Why decoding failed, if it did. */
  readonly failure: string | null
}

const TOO_MANY = 'rsDecode: too many errors to correct'

/**
 * Reed-Solomon decoding with errors and erasures as a stepped algorithm: the syndromes and the erasure locator are the
 * start, each step is one iteration of the Berlekamp-Massey algorithm (Blahut's errata form) computing the discrepancy and
 * updating the errata locator, and a last step does the Chien search for the error positions, Forney's formula for the
 * values and the check that the corrected word has zero syndromes. It logs the syndromes, the locator, the discrepancy
 * and the errors implied, and is done (`converged`) when the word is corrected (or was clean) and `failed` when there
 * are too many errors. The state serialises; read the result with {@link rsDecodeResult}.
 *
 * @param code - The code.
 * @param received - The received codeword (message symbols then nsym parity symbols).
 * @param options - Known erasure positions (indices into `received`).
 * @returns The algorithm.
 * @throws OmaiError with code `value` for symbols out of the field, a word of the wrong length, erasures out of range
 * or more than nsym erasures.
 * @concept reed-solomon
 * @example
 * ```ts
 * import { reedSolomon, rsDecodeAlgorithm, rsDecodeResult, rsEncode } from 'omai-engine/information'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const code = reedSolomon(4)
 * const word = rsEncode(code, [1, 2, 3, 4])
 * word[1] = 99
 * const run = runAlgorithm(rsDecodeAlgorithm(code, word), { maxSteps: 20 })
 * [run.status, rsDecodeResult(run.state, code).message] // => ['converged', [1, 2, 3, 4]]
 * ```
 */
export function rsDecodeAlgorithm(
  code: ReedSolomonCode,
  received: ArrayLike<number>,
  options: { readonly erasures?: readonly number[] } = {},
): Algorithm<RsDecodeState, RsPath> {
  const f = code.field
  const r = symbols(code, received, 'rsDecode')
  const n = r.length
  const nsym = code.nsym
  if (n <= nsym || n > f.size - 1)
    throw new OmaiError('value', `rsDecode: expected ${nsym + 1} to ${f.size - 1} symbols`)
  const erasures = [...new Set(options.erasures ?? [])]
  if (erasures.some((e) => !Number.isInteger(e) || e < 0 || e >= n))
    throw new OmaiError('value', 'rsDecode: erasure positions out of range')
  if (erasures.length > nsym) throw new OmaiError('value', 'rsDecode: too many erasures')
  const rho = erasures.length
  const locatorOf = (i: number): number => f.pow(2, n - 1 - i)
  const syndromesOf = (word: readonly number[]): number[] =>
    Array.from({ length: nsym }, (_, j) => {
      const x = f.pow(2, j + code.fcr)
      let y = 0
      for (const c of word) y = f.mul(y, x) ^ c
      return y
    })
  return {
    name: 'information/reed-solomon-decode',
    config: { nsym, n, m: f.m, erasures },
    init: () => {
      const s = syndromesOf(r)
      // Erasure locator Gamma(x) = product of (1 - X_e x), lowest degree first.
      let gamma = [1]
      for (const e of erasures) gamma = mulLow(f, gamma, [1, locatorOf(e)])
      const clean = s.every((v) => v === 0) && rho === 0
      return {
        received: r,
        erasures,
        syndromes: s,
        step: rho + 1,
        lambda: gamma,
        b: gamma,
        l: rho,
        discrepancy: 0,
        phase: clean ? 'done' : rho + 1 > nsym ? 'correct' : 'bm',
        corrected: r,
        errata: [],
        failure: null,
      }
    },
    step: (st, ctx) => {
      const s = st.syndromes
      if (st.phase === 'bm') {
        let lambda = [...st.lambda]
        let b = [...st.b]
        let l = st.l
        const step = st.step
        let delta = 0
        for (let i = 0; i <= l && i < lambda.length; i++) delta ^= f.mul(lambda[i] ?? 0, s[step - 1 - i] ?? 0)
        const shifted = [0, ...b]
        if (delta === 0) b = shifted
        else {
          const t = new Array<number>(Math.max(lambda.length, shifted.length)).fill(0)
          for (let i = 0; i < t.length; i++) t[i] = (lambda[i] ?? 0) ^ f.mul(delta, shifted[i] ?? 0)
          if (2 * l <= step + rho - 1) {
            b = lambda.map((c) => f.div(c, delta))
            l = step + rho - l
          } else b = shifted
          lambda = t
        }
        ctx.log('rs/syndrome', s)
        ctx.log('rs/locator', lambda)
        ctx.log('rs/discrepancy', delta)
        ctx.log('rs/errors', Math.max(0, l - rho))
        return { ...st, step: step + 1, lambda, b, l, discrepancy: delta, phase: step + 1 > nsym ? 'correct' : 'bm' }
      }
      // The locator is final: find the errata and fix them.
      const lambda = [...st.lambda]
      while (lambda.length > 1 && lambda[lambda.length - 1] === 0) lambda.pop()
      const fail = (failure: string): RsDecodeState => ({ ...st, lambda, phase: 'done', failure })
      const degree = lambda.length - 1
      if (2 * (degree - rho) + rho > nsym) return fail(TOO_MANY)
      // Chien search over the positions of this (possibly shortened) word.
      const positions: number[] = []
      for (let i = 0; i < n; i++) if (evalLow(f, lambda, f.inverse(locatorOf(i))) === 0) positions.push(i)
      if (positions.length !== degree) return fail(TOO_MANY)
      // Forney: e = X^(1 - fcr) Omega(X^-1) / Lambda'(X^-1), with Omega = S Lambda mod x^nsym.
      const omega = mulLow(f, [...s], lambda).slice(0, nsym)
      const derivative = lambda.slice(1).map((c, i) => (i % 2 === 0 ? c : 0))
      const corrected = [...r]
      for (const i of positions) {
        const x = locatorOf(i)
        const xInv = f.inverse(x)
        const denominator = evalLow(f, derivative, xInv)
        if (denominator === 0) return fail(TOO_MANY)
        const magnitude = f.mul(f.pow(x, 1 - code.fcr), f.div(evalLow(f, omega, xInv), denominator))
        corrected[i] = (corrected[i] ?? 0) ^ magnitude
      }
      if (syndromesOf(corrected).some((v) => v !== 0)) return fail(TOO_MANY)
      const errata = positions.filter((i) => corrected[i] !== r[i] || erasures.includes(i)).sort((a, c) => a - c)
      ctx.log('rs/locator', lambda)
      ctx.log('rs/errors', errata.length)
      return { ...st, lambda, phase: 'done', corrected, errata }
    },
    done: (st) => (st.phase !== 'done' ? undefined : st.failure === null ? 'converged' : 'failed'),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as RsDecodeState,
  }
}

/**
 * The decoded word of a finished Reed-Solomon decoding state.
 *
 * @param state - A finished state of {@link rsDecodeAlgorithm}.
 * @param code - The code (for the number of parity symbols).
 * @returns The message, the corrected codeword and the corrected positions.
 * @throws OmaiError with code `value` when the decoding failed (too many errors).
 * @concept reed-solomon
 * @example
 * ```ts
 * import { reedSolomon, rsDecodeAlgorithm, rsDecodeResult, rsEncode } from 'omai-engine/information'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const code = reedSolomon(4)
 * const run = runAlgorithm(rsDecodeAlgorithm(code, rsEncode(code, [1, 2, 3])), { maxSteps: 20 })
 * rsDecodeResult(run.state, code).errata // => []
 * ```
 */
export function rsDecodeResult(state: RsDecodeState, code: ReedSolomonCode): RsDecodeResult {
  if (state.failure !== null) throw new OmaiError('value', state.failure)
  const n = state.received.length
  return { message: state.corrected.slice(0, n - code.nsym), codeword: [...state.corrected], errata: [...state.errata] }
}

/**
 * Decode a Reed-Solomon codeword with errors and erasures: syndromes, the Berlekamp-Massey algorithm started from the
 * erasure locator (Blahut's errata form), a Chien search for the error positions and Forney's formula for the values.
 * {@link rsDecodeAlgorithm} run to the end.
 *
 * @param code - The code.
 * @param received - The received codeword (message symbols then nsym parity symbols).
 * @param options - Known erasure positions (indices into `received`).
 * @returns The corrected message and codeword and the corrected positions.
 * @throws OmaiError with code `value` when there are too many errors to correct.
 * @concept reed-solomon
 * @example
 * ```ts
 * import { reedSolomon, rsDecode, rsEncode } from 'omai-engine/information'
 * const code = reedSolomon(4)
 * const word = rsEncode(code, [1, 2, 3, 4])
 * word[1] = 99
 * word[5] = 0
 * rsDecode(code, word).message // => [1, 2, 3, 4]
 * ```
 */
export function rsDecode(
  code: ReedSolomonCode,
  received: ArrayLike<number>,
  options: { readonly erasures?: readonly number[] } = {},
): RsDecodeResult {
  const run = runAlgorithm(rsDecodeAlgorithm(code, received, options), { maxSteps: Number.MAX_SAFE_INTEGER })
  return rsDecodeResult(run.state, code)
}
