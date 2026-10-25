/**
 * Low-density parity-check (LDPC) codes, linear codes with sparse parity checks that come close to channel capacity:
 * Gallager's random regular construction, the Tanner graph and encoder, log-likelihood ratios from AWGN and binary
 * symmetric channels, and decoding by belief propagation on the Tanner graph (also as a traced algorithm).
 */

import { type Json, WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { type BitMatrix, bitMatrix, matVec, nullSpace, rref, vecMat } from './_gf2.ts'
import { float } from './_shared.ts'
import type { LinearCode } from './error-correction.ts'

/** A low-density parity-check code: a linear code with a sparse parity-check matrix, and its Tanner graph. */
export interface LdpcCode extends LinearCode {
  /** The variables in each check (rows of the parity-check matrix). */
  readonly checks: readonly (readonly number[])[]
  /** Positions of the message bits in a codeword (the generator is the identity there). */
  readonly information: readonly number[]
}

/**
 * Gallager's random regular parity-check matrix (1962): `columnWeight` bands of n / rowWeight checks, the first band
 * covering consecutive blocks of rowWeight variables and the others random column permutations of it, so every
 * variable is in columnWeight checks and every check has rowWeight variables.
 *
 * @param n - Block length (a multiple of rowWeight).
 * @param columnWeight - Checks per variable (at least 2).
 * @param rowWeight - Variables per check (larger than columnWeight).
 * @param key - Randomness for the permutations.
 * @returns The `(n columnWeight / rowWeight) x n` parity-check matrix.
 * @concept ldpc-codes
 * @example
 * ```ts
 * import { gallagerParityCheck } from 'whitebox-engine/information'
 * import { key } from 'whitebox-engine/random'
 * const h = gallagerParityCheck(12, 3, 4, key(0))
 * h.length // => 9
 * h.every((row) => row.reduce((a, b) => a + b, 0) === 4) // => true
 * ```
 */
export function gallagerParityCheck(n: number, columnWeight: number, rowWeight: number, key: Key): number[][] {
  if (!Number.isInteger(n) || !Number.isInteger(columnWeight) || !Number.isInteger(rowWeight) || n % rowWeight !== 0)
    throw new WhiteboxError('value', 'gallagerParityCheck: n must be a multiple of rowWeight')
  if (columnWeight < 2 || rowWeight <= columnWeight)
    throw new WhiteboxError('value', 'gallagerParityCheck: need 2 <= columnWeight < rowWeight')
  const band = n / rowWeight
  const rows: number[][] = []
  for (let b = 0; b < columnWeight; b++) {
    const perm = b === 0 ? Array.from({ length: n }, (_, i) => i) : toFlat(permutation(foldIn(key, b), n))
    for (let i = 0; i < band; i++) {
      const row = new Array<number>(n).fill(0)
      for (let j = 0; j < rowWeight; j++) row[perm[i * rowWeight + j] ?? 0] = 1
      rows.push(row)
    }
  }
  return rows
}

/**
 * The LDPC code of a parity-check matrix: its Tanner graph and a generator in reduced echelon form (so the message
 * bits sit at the pivot positions of the codeword). Redundant checks are allowed; k = n - rank(H).
 *
 * @param parityCheck - The parity-check matrix.
 * @returns The code.
 * @throws WhiteboxError with code `value` when the matrix has full column rank (no codewords but zero).
 * @concept ldpc-codes
 * @example
 * ```ts
 * import { ldpcCode } from 'whitebox-engine/information'
 * ldpcCode([[1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1]]).k // => 3
 * ```
 */
export function ldpcCode(parityCheck: BitMatrix): LdpcCode {
  const h = bitMatrix(parityCheck, 'ldpcCode')
  const n = h[0]?.length ?? 0
  const basis = nullSpace(h)
  if (basis.length === 0) throw new WhiteboxError('value', 'ldpcCode: the only codeword is zero')
  const { rows, pivots } = rref(basis)
  return {
    n,
    k: rows.length,
    generator: rows,
    parityCheck: h,
    checks: h.map((row) => row.flatMap((v, j) => (v === 1 ? [j] : []))),
    information: pivots,
  }
}

/**
 * Encode a message with an LDPC code (message x G).
 *
 * @param code - The code.
 * @param message - k bits.
 * @returns The codeword.
 * @concept ldpc-codes
 * @example
 * ```ts
 * import { ldpcCode, ldpcEncode } from 'whitebox-engine/information'
 * const code = ldpcCode([[1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1]])
 * ldpcEncode(code, [1, 0, 1]).length // => 6
 * ```
 */
export function ldpcEncode(code: LdpcCode, message: readonly number[]): number[] {
  if (message.length !== code.k) throw new WhiteboxError('shape', `ldpcEncode: expected ${code.k} message bits`)
  return vecMat(message, code.generator)
}

/**
 * Log-likelihood ratios log P(0 | y) / P(1 | y) for BPSK (0 sent as +1, 1 as -1) over an AWGN channel: 2 y / sigma^2.
 *
 * @param received - Channel outputs.
 * @param noiseVariance - The noise variance sigma^2.
 * @returns The LLRs.
 * @concept ldpc-codes
 * @example
 * ```ts
 * import { bpskLlr } from 'whitebox-engine/information'
 * bpskLlr([1, -0.5], 0.5) // => [4, -2]
 * ```
 */
export function bpskLlr(received: TensorLike, noiseVariance: number): number[] {
  if (!(noiseVariance > 0)) throw new WhiteboxError('value', 'bpskLlr: noiseVariance must be positive')
  return toFlat(float(received)).map((y) => (2 * y) / noiseVariance)
}

/**
 * Log-likelihood ratios for bits received over a binary symmetric channel with crossover p: +/- log((1 - p) / p).
 *
 * @param bits - Received bits.
 * @param p - The crossover probability in (0, 1/2).
 * @returns The LLRs.
 * @concept ldpc-codes
 * @example
 * ```ts
 * import { bscLlr } from 'whitebox-engine/information'
 * bscLlr([0, 1], 0.2).map((l) => Math.round(l * 1000) / 1000) // => [1.386, -1.386]
 * ```
 */
export function bscLlr(bits: readonly number[], p: number): number[] {
  if (!(p > 0 && p <= 0.5)) throw new WhiteboxError('value', 'bscLlr: p must lie in (0, 1/2]')
  const l = Math.log((1 - p) / p)
  return bits.map((b) => (b === 1 ? -l : l))
}

/** The state of {@link beliefPropagationAlgorithm}. */
export interface BeliefPropagationState {
  readonly iteration: number
  /** Check-to-variable messages, one per edge of the Tanner graph (edges ordered check by check). */
  readonly messages: Float64Array
  /** Posterior LLRs of the variables. */
  readonly posterior: Float64Array
  /** Unsatisfied checks of the hard decision. */
  readonly unsatisfied: number
}

/** Options of belief-propagation decoding. */
export interface BeliefPropagationOptions {
  /** `'sumProduct'` (default, the tanh rule) or `'minSum'` (signs and minimum magnitudes). */
  readonly method?: 'sumProduct' | 'minSum'
  /** Scaling of the min-sum messages (normalised min-sum, default 1). */
  readonly scale?: number
}

const CLIP = 1 - 1e-15

/**
 * Belief propagation on the Tanner graph of an LDPC code (Gallager 1962, MacKay and Neal 1996) as a traced algorithm:
 * each step is one flooding iteration, variables sending their posterior minus the message from each check and
 * checks answering by the tanh rule (or min-sum). It converges when the hard decision satisfies every check.
 *
 * @param parityCheck - The parity-check matrix (or a code's).
 * @param llr - Channel LLRs, log P(0) / P(1) per variable.
 * @param options - Method and min-sum scale.
 * @returns The algorithm.
 * @concept belief-propagation
 * @example
 * ```ts
 * import { beliefPropagationAlgorithm } from 'whitebox-engine/information'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const h = [[1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1]]
 * runAlgorithm(beliefPropagationAlgorithm(h, [2, 2, -1, 2, 2, 2]), { maxSteps: 20 }).status // => 'converged'
 * ```
 */
export function beliefPropagationAlgorithm(
  parityCheck: BitMatrix,
  llr: readonly number[],
  options: BeliefPropagationOptions = {},
): Algorithm<BeliefPropagationState> {
  const h = bitMatrix(parityCheck, 'beliefPropagation')
  const n = h[0]?.length ?? 0
  if (llr.length !== n) throw new WhiteboxError('shape', `beliefPropagation: expected ${n} LLRs`)
  const minSum = options.method === 'minSum'
  const scale = options.scale ?? 1
  const edgeCheck: number[] = []
  const edgeVar: number[] = []
  const checkStart: number[] = []
  h.forEach((row, c) => {
    checkStart.push(edgeCheck.length)
    row.forEach((v, j) => {
      if (v === 1) {
        edgeCheck.push(c)
        edgeVar.push(j)
      }
    })
  })
  checkStart.push(edgeCheck.length)
  const edges = edgeVar.length
  const unsatisfiedOf = (posterior: Float64Array): number =>
    matVec(
      h,
      Array.from(posterior, (l) => (l < 0 ? 1 : 0)),
    ).reduce((s, v) => s + v, 0)
  return {
    name: 'beliefPropagation',
    config: { variables: n, checks: h.length, edges, method: minSum ? 'minSum' : 'sumProduct', scale } as Json,
    schema: channels({
      metrics: { unsatisfied: { description: 'checks failed by the hard decision', direction: 'lower' } },
      tensors: { posterior: { role: 'state', description: 'posterior LLR of each variable' } },
    }),
    init: () => {
      const posterior = Float64Array.from(llr)
      return { iteration: 0, messages: new Float64Array(edges), posterior, unsatisfied: unsatisfiedOf(posterior) }
    },
    step: (s) => {
      const q = Float64Array.from(
        { length: edges },
        (_, e) => (s.posterior[edgeVar[e] ?? 0] ?? 0) - (s.messages[e] ?? 0),
      )
      const r = new Float64Array(edges)
      for (let c = 0; c < h.length; c++) {
        const a = checkStart[c] ?? 0
        const b = checkStart[c + 1] ?? 0
        for (let e = a; e < b; e++) {
          if (minSum) {
            let sign = 1
            let smallest = Number.POSITIVE_INFINITY
            for (let f = a; f < b; f++) {
              if (f === e) continue
              const v = q[f] ?? 0
              if (v < 0) sign = -sign
              smallest = Math.min(smallest, Math.abs(v))
            }
            r[e] = scale * sign * smallest
          } else {
            let product = 1
            for (let f = a; f < b; f++) if (f !== e) product *= Math.tanh((q[f] ?? 0) / 2)
            r[e] = 2 * Math.atanh(Math.max(-CLIP, Math.min(CLIP, product)))
          }
        }
      }
      const posterior = Float64Array.from(llr)
      for (let e = 0; e < edges; e++) posterior[edgeVar[e] ?? 0] = (posterior[edgeVar[e] ?? 0] ?? 0) + (r[e] ?? 0)
      return { iteration: s.iteration + 1, messages: r, posterior, unsatisfied: unsatisfiedOf(posterior) }
    },
    done: (s) => (s.unsatisfied === 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { unsatisfied: s.unsatisfied },
      tensors: { posterior: { role: 'state', value: s.posterior } },
    }),
  }
}

/** The outcome of LDPC decoding. */
export interface LdpcDecodeResult {
  /** The decoded codeword (the hard decision on the posterior). */
  readonly codeword: number[]
  /** The message bits read from the information positions. */
  readonly message: number[]
  readonly posterior: Float64Array
  readonly iterations: number
  /** Whether every parity check is satisfied. */
  readonly converged: boolean
}

/**
 * Decode an LDPC code by belief propagation (see {@link beliefPropagationAlgorithm}), stopping as soon as every check
 * is satisfied.
 *
 * @param code - The code.
 * @param llr - Channel LLRs (for example from {@link bpskLlr} or {@link bscLlr}).
 * @param options - Method, scale, iteration cap (default 50), and a tracer and policy.
 * @returns The codeword, message, posterior, iterations and whether it converged.
 * @concept belief-propagation
 * @example
 * ```ts
 * import { bscLlr, ldpcCode, ldpcDecode, ldpcEncode } from 'whitebox-engine/information'
 * const code = ldpcCode([[1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1]])
 * const word = ldpcEncode(code, [1, 0, 1])
 * const noisy = word.map((b, i) => (i === 2 ? 1 - b : b))
 * ldpcDecode(code, bscLlr(noisy, 0.1)).message // => [1, 0, 1]
 * ```
 */
export function ldpcDecode(
  code: LdpcCode,
  llr: readonly number[],
  options: BeliefPropagationOptions & { readonly maxIter?: number } & Partial<
      Omit<RunOptions, 'maxSteps' | 'start'>
    > = {},
): LdpcDecodeResult {
  const { method, scale, maxIter, ...run } = options
  const algorithm = beliefPropagationAlgorithm(code.parityCheck, llr, {
    ...(method === undefined ? {} : { method }),
    ...(scale === undefined ? {} : { scale }),
  })
  const result = runAlgorithm(algorithm, { ...run, maxSteps: maxIter ?? 50 })
  const codeword = Array.from(result.state.posterior, (l) => (l < 0 ? 1 : 0))
  return {
    codeword,
    message: code.information.map((i) => codeword[i] ?? 0),
    posterior: result.state.posterior,
    iterations: result.state.iteration,
    converged: result.state.unsatisfied === 0,
  }
}
