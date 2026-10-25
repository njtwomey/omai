/**
 * Convolutional codes, error-correcting codes in which each output bit is a parity of the last K input bits held in a
 * shift register: building a code from its generator polynomials, encoding with optional termination, and
 * maximum-likelihood decoding by the Viterbi algorithm over the code's trellis (also as a traced algorithm), with hard
 * or soft decisions.
 */

import { argminOf, type Json, MdmlError, maxOf, minOf } from '../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'

/**
 * A rate 1/n feed-forward convolutional code: generator polynomials over the last K inputs (bit K - 1 of each generator
 * taps the current input, bit 0 the oldest), as in the usual octal notation (7, 5).
 */
export interface ConvolutionalCode {
  readonly generators: readonly number[]
  readonly constraintLength: number
  /** 2^(K - 1) encoder states (the last K - 1 inputs, most recent in the top bit). */
  readonly states: number
  /** next[2 s + u]: the state after input u in state s. */
  readonly next: Int32Array
  /** outputs[2 s + u]: the n output bits. */
  readonly outputs: readonly (readonly number[])[]
}

const parity = (x: number): number => {
  let v = x
  let p = 0
  while (v !== 0) {
    p ^= v & 1
    v >>>= 1
  }
  return p
}

/**
 * A convolutional code from its generator polynomials (Elias 1955): each input bit enters a K-bit shift register and
 * each generator emits the parity of the taps it selects.
 *
 * @param generators - Generator polynomials, for example `[0o7, 0o5]` (the K = 3 code) or `[0o171, 0o133]` (K = 7).
 * @param constraintLength - K (default: the bit length of the largest generator).
 * @returns The code with its trellis tables.
 * @concept convolutional-codes
 * @example
 * ```ts
 * import { convolutionalCode } from 'mdml-engine/information'
 * convolutionalCode([0o7, 0o5]).states // => 4
 * ```
 */
export function convolutionalCode(generators: readonly number[], constraintLength?: number): ConvolutionalCode {
  if (generators.length === 0 || generators.some((g) => !Number.isInteger(g) || g < 1))
    throw new MdmlError('value', 'convolutionalCode: generators must be positive integers')
  const k = constraintLength ?? maxOf(generators.map((g) => g.toString(2).length))
  if (!Number.isInteger(k) || k < 2 || k > 16 || generators.some((g) => g >= 2 ** k))
    throw new MdmlError('value', 'convolutionalCode: the constraint length must be in [2, 16] and fit the generators')
  const states = 2 ** (k - 1)
  const next = new Int32Array(2 * states)
  const outputs: number[][] = []
  for (let s = 0; s < states; s++)
    for (let u = 0; u < 2; u++) {
      const register = (u << (k - 1)) | s
      next[2 * s + u] = register >>> 1
      outputs[2 * s + u] = generators.map((g) => parity(g & register))
    }
  return { generators, constraintLength: k, states, next, outputs }
}

/**
 * Encode bits with a convolutional code, n output bits per input bit, optionally followed by K - 1 zeros that return
 * the encoder to the zero state (termination).
 *
 * @param code - The code.
 * @param bits - The input bits.
 * @param options - `terminate` (default true).
 * @returns The coded bits.
 * @concept convolutional-codes
 * @example
 * ```ts
 * import { convolutionalCode, convolutionalEncode } from 'mdml-engine/information'
 * convolutionalEncode(convolutionalCode([0o7, 0o5]), [1, 0, 1, 1]) // => [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1]
 * ```
 */
export function convolutionalEncode(
  code: ConvolutionalCode,
  bits: readonly number[],
  options: { readonly terminate?: boolean } = {},
): number[] {
  const input = [...bits, ...new Array<number>(options.terminate === false ? 0 : code.constraintLength - 1).fill(0)]
  const out: number[] = []
  let s = 0
  for (const b of input) {
    if (b !== 0 && b !== 1) throw new MdmlError('value', 'convolutionalEncode: bits must be 0 or 1')
    out.push(...(code.outputs[2 * s + b] ?? []))
    s = code.next[2 * s + b] ?? 0
  }
  return out
}

/** Survivor history as a linked list of per-stage predecessor tables (so each step is O(states)). */
interface History {
  readonly predecessors: Int32Array
  readonly previous?: History
}

/** The state of {@link viterbiAlgorithm}: path metrics after `t` trellis stages and the survivors. */
export interface ViterbiState {
  readonly t: number
  readonly metrics: Float64Array
  readonly history?: History
}

/** Options of the Viterbi decoder. */
export interface ViterbiOptions {
  /**
   * `'hard'` (default): received bits, Hamming branch metrics. `'soft'`: received real values with 0 sent as +1 and 1
   * as -1 (BPSK), squared Euclidean branch metrics.
   */
  readonly decision?: 'hard' | 'soft'
  /** The encoder was terminated in the zero state (default true). */
  readonly terminated?: boolean
}

/**
 * The Viterbi algorithm (1967) as a traced algorithm: one step per trellis stage keeps, for every encoder state, the
 * best path into it (add, compare, select). Path metrics are recorded as a tensor and the best metric as a metric, so a
 * figure can show the trellis being explored.
 *
 * @param code - The code.
 * @param received - n values per stage (bits, or reals for soft decisions).
 * @param options - Decision type.
 * @returns The algorithm; it completes after len(received) / n stages.
 * @concept viterbi
 * @example
 * ```ts
 * import { convolutionalCode, viterbiAlgorithm } from 'mdml-engine/information'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const code = convolutionalCode([0o7, 0o5])
 * runAlgorithm(viterbiAlgorithm(code, [1, 1, 1, 0, 0, 0]), { maxSteps: 10 }).steps // => 3
 * ```
 */
export function viterbiAlgorithm(
  code: ConvolutionalCode,
  received: readonly number[],
  options: Pick<ViterbiOptions, 'decision'> = {},
): Algorithm<ViterbiState> {
  const n = code.generators.length
  if (received.length % n !== 0) throw new MdmlError('shape', `viterbi: received length must be a multiple of ${n}`)
  const stages = received.length / n
  const soft = options.decision === 'soft'
  const branch = (t: number, out: readonly number[]): number => {
    let m = 0
    for (let i = 0; i < n; i++) {
      const r = received[t * n + i] ?? 0
      const c = out[i] ?? 0
      m += soft ? (r - (1 - 2 * c)) ** 2 : r === c ? 0 : 1
    }
    return m
  }
  return {
    name: 'viterbi',
    config: { generators: [...code.generators], constraintLength: code.constraintLength, stages, soft } as Json,
    schema: channels({
      metrics: { best: { description: 'the smallest path metric', direction: 'lower' } },
      tensors: { metrics: { role: 'state', description: 'path metric of each encoder state' } },
    }),
    init: () => {
      const metrics = new Float64Array(code.states).fill(Number.POSITIVE_INFINITY)
      metrics[0] = 0
      return { t: 0, metrics }
    },
    step: (s) => {
      const metrics = new Float64Array(code.states).fill(Number.POSITIVE_INFINITY)
      const predecessors = new Int32Array(code.states).fill(-1)
      for (let state = 0; state < code.states; state++) {
        const m = s.metrics[state] ?? Number.POSITIVE_INFINITY
        if (m === Number.POSITIVE_INFINITY) continue
        for (let u = 0; u < 2; u++) {
          const ns = code.next[2 * state + u] ?? 0
          const candidate = m + branch(s.t, code.outputs[2 * state + u] ?? [])
          if (candidate < (metrics[ns] ?? Number.POSITIVE_INFINITY)) {
            metrics[ns] = candidate
            predecessors[ns] = state
          }
        }
      }
      return {
        t: s.t + 1,
        metrics,
        history: { predecessors, ...(s.history === undefined ? {} : { previous: s.history }) },
      }
    },
    done: (s) => (s.t >= stages ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { best: minOf(s.metrics) },
      tensors: { metrics: { role: 'state', value: Array.from(s.metrics, (v) => (Number.isFinite(v) ? v : -1)) } },
    }),
  }
}

/** The outcome of Viterbi decoding. */
export interface ViterbiResult {
  /** The decoded input bits (without the termination tail). */
  readonly bits: number[]
  /** The metric of the chosen path (Hamming distance for hard decisions). */
  readonly metric: number
}

/**
 * Maximum-likelihood decoding of a convolutional code by the traced {@link viterbiAlgorithm}: the surviving path into
 * the zero state (or the best state for an unterminated code) is traced back to the input bits.
 *
 * @param code - The code.
 * @param received - The received values, n per stage.
 * @param options - Decision type, termination, and a tracer and policy.
 * @returns The decoded bits and the path metric.
 * @concept viterbi
 * @example
 * ```ts
 * import { convolutionalCode, viterbiDecode } from 'mdml-engine/information'
 * const code = convolutionalCode([0o7, 0o5])
 * // The codeword of [1, 0, 1, 1] with two bits flipped.
 * viterbiDecode(code, [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1]).bits // => [1, 0, 1, 1]
 * ```
 */
export function viterbiDecode(
  code: ConvolutionalCode,
  received: readonly number[],
  options: ViterbiOptions & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): ViterbiResult {
  const { decision, terminated = true, ...run } = options
  const algorithm = viterbiAlgorithm(code, received, decision === undefined ? {} : { decision })
  const stages = received.length / code.generators.length
  const { state } = runAlgorithm(algorithm, { ...run, maxSteps: stages })
  let s = terminated ? 0 : argminOf(state.metrics)
  const metric = state.metrics[s] ?? Number.POSITIVE_INFINITY
  if (!Number.isFinite(metric)) throw new MdmlError('value', 'viterbiDecode: no path ends in the zero state')
  const bits: number[] = []
  for (let h = state.history; h !== undefined; h = h.previous) {
    bits.push((s >> (code.constraintLength - 2)) & 1)
    s = h.predecessors[s] ?? 0
  }
  bits.reverse()
  return { bits: terminated ? bits.slice(0, Math.max(0, bits.length - (code.constraintLength - 1))) : bits, metric }
}
