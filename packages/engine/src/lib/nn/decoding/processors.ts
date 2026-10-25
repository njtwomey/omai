/**
 * Logit processors, the functions that reshape next-token logits before choosing a token: temperature, top-k, nucleus
 * (top-p) and min-p filtering, CTRL's repetition penalty and OpenAI-style frequency and presence penalties, with
 * `composeProcessors` to chain them and helpers for log-probabilities and entropy of logits.
 */

import { maxOf, WhiteboxError } from '../../base/index.ts'
import { logSoftmax, tensor, toFlat } from '../../tensor/index.ts'

/** A logit processor: new logits for the next token from the current ones and the whole prefix (prompt included). */
export type LogitProcessor = (logits: readonly number[], prefix: readonly number[]) => number[]

const NEG = Number.NEGATIVE_INFINITY

/**
 * Natural log-probabilities of logits (a log-softmax; positions at -infinity stay there).
 *
 * @param logits - Unnormalised scores.
 * @returns log p, or all -infinity when every logit is -infinity.
 * @example
 * ```ts
 * import { logProbs } from 'whitebox-engine/nn'
 * logProbs([0, 0]).map((v) => Math.round(v * 1e6) / 1e6) // => [-0.693147, -0.693147]
 * ```
 */
export function logProbs(logits: readonly number[]): number[] {
  if (logits.length === 0 || maxOf(logits) === NEG) return logits.map(() => NEG)
  return toFlat(logSoftmax(tensor([...logits]), -1))
}

/**
 * The entropy (nats) of the distribution of some logits.
 *
 * @param logits - Unnormalised scores.
 * @returns -sum p log p.
 * @example
 * ```ts
 * import { entropyOfLogits } from 'whitebox-engine/nn'
 * entropyOfLogits([0, 0]) === Math.log(2) // => true
 * ```
 */
export function entropyOfLogits(logits: readonly number[]): number {
  let h = 0
  for (const lp of logProbs(logits)) if (lp > NEG) h -= Math.exp(lp) * lp
  return h
}

/**
 * Divide the logits by a temperature (below 1 sharpens, above 1 flattens).
 *
 * @param temperature - Positive temperature.
 * @returns The processor.
 * @example
 * ```ts
 * import { temperatureProcessor } from 'whitebox-engine/nn'
 * temperatureProcessor(2)([2, 4], []) // => [1, 2]
 * ```
 */
export function temperatureProcessor(temperature: number): LogitProcessor {
  if (!(temperature > 0)) throw new WhiteboxError('value', 'temperatureProcessor: the temperature must be positive')
  return (logits) => logits.map((z) => z / temperature)
}

/**
 * Keep the k largest logits (ties at the k-th value kept), the rest to -infinity (top-k sampling, Fan et al. 2018).
 *
 * @param k - How many to keep.
 * @returns The processor.
 * @example
 * ```ts
 * import { topKProcessor } from 'whitebox-engine/nn'
 * topKProcessor(2)([1, 3, 2], []) // => [-Infinity, 3, 2]
 * ```
 */
export function topKProcessor(k: number): LogitProcessor {
  return (logits) => {
    if (k >= logits.length) return [...logits]
    const kth = [...logits].sort((a, b) => b - a)[Math.max(k, 1) - 1] ?? NEG
    return logits.map((z) => (z >= kth ? z : NEG))
  }
}

/**
 * Nucleus filtering (top-p, Holtzman et al. 2020): keep the smallest set of most likely tokens whose probability
 * reaches p (always at least the most likely one), the rest to -infinity.
 *
 * @param p - The cumulative probability in (0, 1].
 * @returns The processor.
 * @reference Holtzman, Buys, Du, Forbes and Choi (2020), The curious case of neural text degeneration.
 * @example
 * ```ts
 * import { topPProcessor } from 'whitebox-engine/nn'
 * topPProcessor(0.6)([Math.log(0.5), Math.log(0.3), Math.log(0.2)], []) // => [Math.log(0.5), Math.log(0.3), -Infinity]
 * ```
 */
export function topPProcessor(p: number): LogitProcessor {
  return (logits) => {
    const lp = logProbs(logits)
    const order = lp.map((_, i) => i).sort((a, b) => (lp[b] ?? NEG) - (lp[a] ?? NEG))
    const keep = new Set<number>()
    let mass = 0
    for (const i of order) {
      keep.add(i)
      mass += Math.exp(lp[i] ?? NEG)
      if (mass >= p - 1e-12) break
    }
    return logits.map((z, i) => (keep.has(i) ? z : NEG))
  }
}

/**
 * Min-p filtering (Nguyen et al. 2024): keep tokens whose probability is at least p times the largest probability.
 *
 * @param p - The relative threshold in [0, 1].
 * @returns The processor.
 * @example
 * ```ts
 * import { minPProcessor } from 'whitebox-engine/nn'
 * minPProcessor(0.4)([Math.log(0.6), Math.log(0.3), Math.log(0.1)], []) // => [Math.log(0.6), Math.log(0.3), -Infinity]
 * ```
 */
export function minPProcessor(p: number): LogitProcessor {
  return (logits) => {
    const top = maxOf(logits)
    const cut = top + Math.log(p)
    return logits.map((z) => (z >= cut ? z : NEG))
  }
}

/**
 * The repetition penalty of CTRL (Keskar et al. 2019): the logit of every token already in the prefix is divided by
 * the penalty when positive and multiplied by it when negative, so repeats become less likely.
 *
 * @param penalty - Above 1 discourages repetition.
 * @returns The processor.
 * @example
 * ```ts
 * import { repetitionPenaltyProcessor } from 'whitebox-engine/nn'
 * repetitionPenaltyProcessor(2)([4, -1, 3], [0, 1]) // => [2, -2, 3]
 * ```
 */
export function repetitionPenaltyProcessor(penalty: number): LogitProcessor {
  return (logits, prefix) => {
    const seen = new Set(prefix)
    return logits.map((z, i) => (seen.has(i) ? (z > 0 ? z / penalty : z * penalty) : z))
  }
}

/**
 * Frequency and presence penalties (OpenAI's): subtract frequency times the token's count in the prefix and presence
 * once if it appears at all.
 *
 * @param options - `frequency` and `presence` (default 0).
 * @returns The processor.
 * @example
 * ```ts
 * import { frequencyPresenceProcessor } from 'whitebox-engine/nn'
 * frequencyPresenceProcessor({ frequency: 0.5, presence: 1 })([0, 0, 0], [1, 1, 2]) // => [0, -2, -1.5]
 * ```
 */
export function frequencyPresenceProcessor(
  options: { readonly frequency?: number; readonly presence?: number } = {},
): LogitProcessor {
  const f = options.frequency ?? 0
  const p = options.presence ?? 0
  return (logits, prefix) => {
    const counts = new Map<number, number>()
    for (const t of prefix) counts.set(t, (counts.get(t) ?? 0) + 1)
    return logits.map((z, i) => {
      const c = counts.get(i) ?? 0
      return c === 0 ? z : z - c * f - p
    })
  }
}

/**
 * Apply processors in order.
 *
 * @param processors - The processors.
 * @returns One processor.
 * @example
 * ```ts
 * import { composeProcessors, temperatureProcessor, topKProcessor } from 'whitebox-engine/nn'
 * composeProcessors(temperatureProcessor(2), topKProcessor(1))([2, 4], []) // => [-Infinity, 2]
 * ```
 */
export function composeProcessors(...processors: readonly LogitProcessor[]): LogitProcessor {
  return (logits, prefix) => processors.reduce<number[]>((z, f) => f(z, prefix), [...logits])
}
