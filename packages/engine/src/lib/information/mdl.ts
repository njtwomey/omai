/**
 * Minimum description length: choose the model that compresses the data best, counting the bits that describe the model
 * as well as the bits that describe the data given it. The universal codes for the multinomial family (normalised
 * maximum likelihood with its exact parametric complexity, the Krichevsky-Trofimov and Laplace sequential codes, and
 * Rissanen's asymptotic expansion), two-part codes whose model cost is (k / 2) log2 n (so the total is BIC in bits), and
 * the two classic selections, a polynomial's degree and a histogram's number of bins, as tables and as algorithms
 * that score one candidate per step.
 */

import { MdmlError, maxOf, minOf } from '../base/index.ts'
import { lstsq } from '../numerics/linalg/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { fromStorage, item, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'

const LN2 = Math.LN2
const lg = (x: number): number => item(lgamma(x))

function logFactorials(n: number): Float64Array {
  const out = new Float64Array(n + 1)
  for (let i = 2; i <= n; i++) out[i] = (out[i - 1] ?? 0) + Math.log(i)
  return out
}

function checkCounts(counts: ArrayLike<number>, name: string): number[] {
  const c = Array.from(counts)
  if (c.length === 0 || c.some((x) => !(Number.isInteger(x) && x >= 0)))
    throw new MdmlError('value', `${name}: counts must be a non-empty list of non-negative integers`)
  return c
}

/**
 * The parametric complexity C(K, n) of the multinomial family with K categories on sequences of length n: the sum over
 * all sequences of their maximum-likelihood probability (Shtarkov's sum). It is the normaliser of the NML distribution,
 * computed exactly: C(1, n) = 1, C(2, n) is a sum of n + 1 terms, and Kontkanen and Myllymaki's recurrence
 * C(K, n) = C(K - 1, n) + n / (K - 2) C(K - 2, n) gives the rest. Its logarithm is the regret of the best universal
 * code over the best code knowing the parameter.
 *
 * @param k - The number of categories.
 * @param n - The sequence length.
 * @returns C(K, n).
 * @throws MdmlError with code `value` unless k >= 1 and n >= 0 are integers.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { multinomialParametricComplexity } from 'mdml-engine/information'
 * [multinomialParametricComplexity(2, 2), multinomialParametricComplexity(3, 2)] // => [2.5, 4.5]
 * ```
 */
export function multinomialParametricComplexity(k: number, n: number): number {
  if (!(Number.isInteger(k) && k >= 1 && Number.isInteger(n) && n >= 0))
    throw new MdmlError('value', 'multinomialParametricComplexity: need integers k >= 1 and n >= 0')
  if (k === 1 || n === 0) return 1
  const f = logFactorials(n)
  let c2 = 0
  for (let h = 0; h <= n; h++) {
    const term =
      (f[n] ?? 0) -
      (f[h] ?? 0) -
      (f[n - h] ?? 0) +
      (h > 0 ? h * Math.log(h / n) : 0) +
      (n - h > 0 ? (n - h) * Math.log((n - h) / n) : 0)
    c2 += Math.exp(term)
  }
  if (k === 2) return c2
  let previous = 1
  let current = c2
  for (let m = 3; m <= k; m++) {
    const next = current + (n / (m - 2)) * previous
    previous = current
    current = next
  }
  return current
}

/**
 * The regret of the normalised maximum likelihood code, log2 C(K, n) in bits: what knowing nothing of the parameter
 * costs over the best code that knew it, for every sequence alike. It is the model-complexity term of the stochastic
 * complexity of a multinomial.
 *
 * @param k - The number of categories.
 * @param n - The sequence length.
 * @returns The regret in bits.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { nmlRegret } from 'mdml-engine/information'
 * Math.round(nmlRegret(2, 2) * 1e6) / 1e6 // => 1.321928
 * ```
 */
export function nmlRegret(k: number, n: number): number {
  return Math.log2(multinomialParametricComplexity(k, n))
}

/**
 * Rissanen's asymptotic expansion of the regret, (K - 1) / 2 log2(n / 2) + log2(sqrt(pi) / Gamma(K / 2)): the cost per
 * parameter grows as half a bit per doubling of the data. {@link nmlRegret} approaches it as n grows, with an error of
 * order 1 / sqrt(n).
 *
 * @param k - The number of categories.
 * @param n - The sequence length.
 * @returns The expansion, in bits.
 * @throws MdmlError with code `value` unless k >= 1 and n >= 1.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { nmlRegretAsymptotic } from 'mdml-engine/information'
 * Math.round(nmlRegretAsymptotic(2, 1000) * 1e3) / 1e3 // => 5.309
 * ```
 */
export function nmlRegretAsymptotic(k: number, n: number): number {
  if (!(Number.isInteger(k) && k >= 1 && n >= 1))
    throw new MdmlError('value', 'nmlRegretAsymptotic: need k >= 1 and n >= 1')
  return ((k - 1) / 2) * Math.log2(n / 2) + (0.5 * Math.log(Math.PI) - lg(k / 2)) / LN2
}

/**
 * The code length in bits of a sequence under the normalised maximum likelihood code for the multinomial family: the
 * maximum-likelihood code length plus the regret, -sum h_k log2(h_k / n) + log2 C(K, n), where h are the counts of each
 * category. It depends on the sequence only through its counts, is the minimax-optimal universal code, and the
 * probabilities 2^-length of all sequences of length n sum to one.
 *
 * @param counts - How often each category occurs.
 * @returns The code length in bits.
 * @throws MdmlError with code `value` for counts that are not non-negative integers.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { nmlCodeLength } from 'mdml-engine/information'
 * Math.round(nmlCodeLength([2, 0]) * 1e6) / 1e6 // => 1.321928
 * ```
 */
export function nmlCodeLength(counts: ArrayLike<number>): number {
  const h = checkCounts(counts, 'nmlCodeLength')
  const n = h.reduce((a, b) => a + b, 0)
  let ml = 0
  for (const x of h) if (x > 0) ml -= x * Math.log2(x / n)
  return ml + nmlRegret(h.length, n)
}

/**
 * The code length in bits of a sequence under the sequential Dirichlet(alpha) code, which predicts each symbol by
 * (count + alpha) / (seen + K alpha) and learns the source as it codes: alpha = 1/2 is the Krichevsky-Trofimov code
 * (regret within a constant of the NML code's), alpha = 1 the Laplace rule that an adaptive arithmetic coder with unit
 * counts implements. The length depends only on the counts, not their order.
 *
 * @param counts - How often each category occurs.
 * @param alpha - The Dirichlet parameter (default 1/2).
 * @returns The code length in bits.
 * @throws MdmlError with code `value` for bad counts or a non-positive alpha.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { dirichletCodeLength } from 'mdml-engine/information'
 * Math.round(dirichletCodeLength([1, 0], 1) * 1e9) / 1e9 // => 1
 * Math.round(dirichletCodeLength([2, 0], 0.5) * 1e6) / 1e6 // => 1.415037
 * ```
 */
export function dirichletCodeLength(counts: ArrayLike<number>, alpha = 0.5): number {
  const h = checkCounts(counts, 'dirichletCodeLength')
  if (!(alpha > 0)) throw new MdmlError('value', 'dirichletCodeLength: alpha must be positive')
  const k = h.length
  const n = h.reduce((a, b) => a + b, 0)
  let logP = lg(k * alpha) - lg(n + k * alpha)
  for (const x of h) logP += lg(x + alpha) - lg(alpha)
  return -logP / LN2
}

/**
 * The cost in bits of stating k parameters to the precision the data can justify, (k / 2) log2 n (Rissanen): each
 * parameter of a regular model is worth half a bit per doubling of the sample, since the data pin it down to about
 * 1 / sqrt(n).
 *
 * @param k - The number of parameters.
 * @param n - The sample size.
 * @returns The bits.
 * @throws MdmlError with code `value` for n < 1 or a negative k.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { parameterBits } from 'mdml-engine/information'
 * parameterBits(4, 256) // => 16
 * ```
 */
export function parameterBits(k: number, n: number): number {
  if (!(k >= 0 && n >= 1)) throw new MdmlError('value', 'parameterBits: need k >= 0 and n >= 1')
  return (k / 2) * Math.log2(n)
}

/**
 * The bits to code n real observations with their Gaussian model at its maximum-likelihood variance, relative to a fixed
 * precision (the constant cost of writing each value to that precision is the same for every candidate model and is
 * left out): (n / 2) log2(2 pi e RSS / n), the negative log likelihood in bits.
 *
 * @param rss - The residual sum of squares.
 * @param n - The sample size.
 * @returns The bits.
 * @throws MdmlError with code `value` unless rss > 0 and n >= 1.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { gaussianDataBits } from 'mdml-engine/information'
 * Math.round(gaussianDataBits(1, 2) * 1e6) / 1e6 // => 3.094
 * ```
 */
export function gaussianDataBits(rss: number, n: number): number {
  if (!(rss > 0 && n >= 1)) throw new MdmlError('value', 'gaussianDataBits: need rss > 0 and n >= 1')
  return (n / 2) * Math.log2((2 * Math.PI * Math.E * rss) / n)
}

/** One candidate model's description length. */
export interface MdlRow {
  /** The candidate's complexity: the polynomial's degree or the histogram's number of bins. */
  readonly complexity: number
  /** The bits that describe the model. */
  readonly modelBits: number
  /** The bits that describe the data given the model. */
  readonly dataBits: number
  /** modelBits + dataBits. */
  readonly total: number
  /** For the polynomial: the BIC of the fit, which equals `2 ln 2 x total`. */
  readonly bic?: number
}

function row(complexity: number, modelBits: number, dataBits: number, bic?: number): MdlRow {
  return { complexity, modelBits, dataBits, total: modelBits + dataBits, ...(bic === undefined ? {} : { bic }) }
}

/**
 * The description length of a polynomial regression of each degree 0 .. maxDegree: the data bits are the Gaussian
 * negative log likelihood at the maximum-likelihood variance ({@link gaussianDataBits}) and the model bits are
 * {@link parameterBits} for the degree + 1 coefficients and the variance. With these two the total is exactly the BIC
 * divided by 2 ln 2, so minimum description length and BIC pick the same degree; the model cost is what stops the
 * data cost, which falls with every extra coefficient, from always winning. The fit is by least squares on x scaled to
 * [-1, 1].
 *
 * @param x - The inputs.
 * @param y - The responses.
 * @param maxDegree - The largest degree (at most n - 2).
 * @returns One row per degree.
 * @throws MdmlError with code `shape` for inputs of different lengths, `value` for a degree beyond n - 2 or constant x.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { polynomialMdl } from 'mdml-engine/information'
 * const x = [0, 1, 2, 3, 4, 5, 6, 7]
 * const rows = polynomialMdl(x, x.map((v) => 2 * v + 1 + (v % 2 ? 0.01 : -0.01)), 3)
 * rows.map((r) => r.complexity) // => [0, 1, 2, 3]
 * ```
 */
export function polynomialMdl(x: ArrayLike<number>, y: ArrayLike<number>, maxDegree: number): MdlRow[] {
  const n = x.length
  if (y.length !== n) throw new MdmlError('shape', 'polynomialMdl: x and y differ in length')
  if (!(Number.isInteger(maxDegree) && maxDegree >= 0 && maxDegree <= n - 2))
    throw new MdmlError('value', `polynomialMdl: maxDegree must be an integer in [0, ${n - 2}]`)
  const lo = minOf(Array.from(x))
  const hi = maxOf(Array.from(x))
  if (!(hi > lo)) throw new MdmlError('value', 'polynomialMdl: x is constant')
  const u = Array.from(x, (v) => (2 * (v - lo)) / (hi - lo) - 1)
  const my = Array.from(y).reduce((a, b) => a + b, 0) / n
  const floor =
    1e-12 *
    Math.max(
      Array.from(y).reduce((a, b) => a + (b - my) ** 2, 0),
      1e-300,
    )
  const out: MdlRow[] = []
  for (let d = 0; d <= maxDegree; d++) {
    const design = Array.from({ length: n }, (_, i) => Array.from({ length: d + 1 }, (_, j) => (u[i] as number) ** j))
    const beta = toFlat(
      lstsq(
        fromStorage(Float64Array.from(design.flat()), [n, d + 1], 'float64'),
        fromStorage(Float64Array.from(y), [n], 'float64'),
      ),
    )
    let rss = 0
    for (let i = 0; i < n; i++) {
      let fit = 0
      for (let j = 0; j <= d; j++) fit += (beta[j] ?? 0) * (u[i] as number) ** j
      rss += ((y[i] ?? 0) - fit) ** 2
    }
    const k = d + 2
    const data = gaussianDataBits(Math.max(rss, floor), n)
    out.push(row(d, parameterBits(k, n), data, 2 * LN2 * (data + parameterBits(k, n))))
  }
  return out
}

/**
 * The description length of equal-width histograms with 1 .. maxBins bins: the data bits are the maximum-likelihood
 * code length of the bin counts plus n log2(w / precision) for writing each value to the given precision inside a bin of
 * width w, and the model bits are the NML regret of the multinomial over the bins ({@link nmlRegret}). The minimum
 * picks a number of bins that grows with the data and the structure in it: one bin for uniform data, many for rugged
 * data, and never so many that the regret outweighs what the finer bins capture.
 *
 * @param data - The observations.
 * @param maxBins - The largest number of bins.
 * @param precision - The resolution the values are coded to (default the range over 2^20; it shifts every row alike).
 * @returns One row per number of bins.
 * @throws MdmlError with code `value` for fewer than two observations, constant data or a non-positive precision.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { histogramMdl } from 'mdml-engine/information'
 * const rows = histogramMdl([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 3)
 * rows.map((r) => r.complexity) // => [1, 2, 3]
 * ```
 */
export function histogramMdl(data: ArrayLike<number>, maxBins: number, precision?: number): MdlRow[] {
  const n = data.length
  if (n < 2) throw new MdmlError('value', 'histogramMdl: need at least two observations')
  const lo = minOf(Array.from(data))
  const hi = maxOf(Array.from(data))
  if (!(hi > lo)) throw new MdmlError('value', 'histogramMdl: the data are constant')
  if (!(Number.isInteger(maxBins) && maxBins >= 1))
    throw new MdmlError('value', 'histogramMdl: maxBins must be a positive integer')
  const delta = precision ?? (hi - lo) / 2 ** 20
  if (!(delta > 0)) throw new MdmlError('value', 'histogramMdl: the precision must be positive')
  const out: MdlRow[] = []
  for (let k = 1; k <= maxBins; k++) {
    const counts = new Array<number>(k).fill(0)
    for (const v of Array.from(data)) {
      const b = Math.min(k - 1, Math.floor(((v - lo) / (hi - lo)) * k))
      counts[b] = (counts[b] ?? 0) + 1
    }
    let ml = 0
    for (const c of counts) if (c > 0) ml -= c * Math.log2(c / n)
    out.push(row(k, nmlRegret(k, n), ml + n * Math.log2((hi - lo) / k / delta)))
  }
  return out
}

/**
 * The candidate with the shortest total description.
 *
 * @param rows - The candidates ({@link polynomialMdl}, {@link histogramMdl}).
 * @returns The row with the least total (the first of equals).
 * @throws MdmlError with code `value` for no rows.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { bestMdl } from 'mdml-engine/information'
 * bestMdl([{ complexity: 1, modelBits: 1, dataBits: 9, total: 10 }, { complexity: 2, modelBits: 2, dataBits: 4, total: 6 }]).complexity // => 2
 * ```
 */
export function bestMdl(rows: readonly MdlRow[]): MdlRow {
  const first = rows[0]
  if (first === undefined) throw new MdmlError('value', 'bestMdl: no candidates')
  return rows.reduce((best, r) => (r.total < best.total ? r : best), first)
}

/** State of {@link mdlSelectionAlgorithm}. */
export interface MdlSelectionState {
  /** Candidates scored so far. */
  readonly index: number
  /** Their description lengths: complexity, model bits, data bits, total, per candidate, flattened. */
  readonly table: readonly number[]
  /** The complexity of the best candidate so far. */
  readonly best: number
}

/** The paths of {@link mdlSelectionAlgorithm}. */
export type MdlPath =
  | 'model/complexity'
  | 'code/modelBits'
  | 'code/dataBits'
  | 'code/total'
  | 'code/best'
  | 'study/curve'

/** The log contract of {@link mdlSelectionAlgorithm}. */
export const mdlContract: Contract<MdlSelectionState, MdlPath> = defineContract({
  'model/complexity': scalar({
    description: 'the complexity of the candidate scored: the degree, or the number of bins',
  }),
  'code/modelBits': scalar({ description: 'the bits that describe the model', unit: 'bits' }),
  'code/dataBits': scalar({ description: 'the bits that describe the data given the model', unit: 'bits' }),
  'code/total': scalar({ description: 'model bits plus data bits', unit: 'bits', direction: 'lower' }),
  'code/best': scalar({ description: 'the complexity of the shortest description so far' }),
  'study/curve': tensor({
    role: 'output',
    shape: ['candidates', 4],
    description: 'complexity, model bits, data bits and total of every candidate scored',
  }),
})

/**
 * Model selection by description length as an algorithm: one candidate per step, in the order of the rows given (from the
 * simplest up), logging the bits that describe the model, the bits that describe the data given it, their sum, and the
 * best complexity so far. The model bits climb with the candidate's complexity and the data bits fall; the total has a
 * minimum. The state serialises.
 *
 * @param rows - The candidates' description lengths ({@link polynomialMdl}, {@link histogramMdl}).
 * @returns The algorithm; it completes after one step per candidate.
 * @throws MdmlError with code `value` for no rows.
 * @concept minimum-description-length
 * @example
 * ```ts
 * import { mdlSelectionAlgorithm } from 'mdml-engine/information'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const rows = [
 *   { complexity: 1, modelBits: 1, dataBits: 9, total: 10 },
 *   { complexity: 2, modelBits: 2, dataBits: 4, total: 6 },
 *   { complexity: 3, modelBits: 3, dataBits: 3.5, total: 6.5 },
 * ]
 * runAlgorithm(mdlSelectionAlgorithm(rows), { seed: 0, maxSteps: 10 }).state.best // => 2
 * ```
 */
export function mdlSelectionAlgorithm(rows: readonly MdlRow[]): Algorithm<MdlSelectionState, MdlPath> {
  if (rows.length === 0) throw new MdmlError('value', 'mdlSelectionAlgorithm: no candidates')
  return {
    name: 'information/mdl-selection',
    config: { candidates: rows.length },
    schema: contractSchema(mdlContract),
    init: () => ({ index: 0, table: [], best: (rows[0] as MdlRow).complexity }),
    step: (s) => {
      const r = rows[s.index] as MdlRow
      const table = [...s.table, r.complexity, r.modelBits, r.dataBits, r.total]
      let best = s.best
      let least = Number.POSITIVE_INFINITY
      for (let i = 0; i < table.length; i += 4) {
        if ((table[i + 3] ?? 0) < least) {
          least = table[i + 3] ?? 0
          best = table[i] ?? 0
        }
      }
      return { index: s.index + 1, table, best }
    },
    done: (s) => (s.index >= rows.length ? 'completed' : undefined),
    observe: (s) => {
      const i = Math.max(s.table.length - 4, 0)
      return {
        metrics: {
          'model/complexity': s.table[i] ?? 0,
          'code/modelBits': s.table[i + 1] ?? 0,
          'code/dataBits': s.table[i + 2] ?? 0,
          'code/total': s.table[i + 3] ?? 0,
          'code/best': s.best,
        },
        tensors: { 'study/curve': { role: 'output', value: s.table, shape: [s.table.length / 4, 4] } },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (json) => stateFromJson(json) as MdlSelectionState,
  }
}
