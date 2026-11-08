/**
 * Stability selection with the Meinshausen-Buhlmann error bound (2010). A selection procedure (here the first q lasso
 * variables) is run on many random half-samples; the variables chosen in a large fraction of them are the stable ones.
 * The bound is the point: if the procedure selects q variables per subsample out of p, then for a threshold pi in
 * (1/2, 1) the expected number of false selections is at most q^2 / ((2 pi - 1) p), whatever the data, so pi can be
 * chosen to fix the error rate in advance. The generic version over any selector is `stabilitySelection` in
 * `feature-selection-wrappers.ts`; this one adds the bound, the threshold that achieves a target, and the lasso procedure.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { lassoCoordinateDescent } from '../optim/sparse.ts'
import { type Key, key as makeKey, subsample } from '../random/index.ts'
import { fromStorage, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { matrixView, targetVector } from './validation.ts'

/**
 * The Meinshausen-Buhlmann bound on the expected number of falsely selected variables, E[V] <= q^2 / ((2 pi - 1) p), for a
 * procedure that selects q of p variables per subsample, a selection threshold pi in (1/2, 1], under exchangeability of the
 * null variables' selection probabilities and a selection procedure no worse than random guessing (their theorem 1).
 *
 * @param q - The average number of variables selected per subsample.
 * @param p - The number of variables.
 * @param threshold - The selection frequency threshold pi.
 * @returns The bound on E[V].
 * @throws OmaiError with code `value` unless 1/2 < pi <= 1.
 * @concept feature-selection
 * @example
 * ```ts
 * import { stabilitySelectionBound } from 'omai-engine/learn'
 * stabilitySelectionBound(8, 40, 0.9) // => 2
 * ```
 */
export function stabilitySelectionBound(q: number, p: number, threshold: number): number {
  if (!(threshold > 0.5 && threshold <= 1))
    throw new OmaiError('value', 'stabilitySelectionBound: threshold must be in (1/2, 1]')
  if (!(q > 0 && p > 0)) throw new OmaiError('value', 'stabilitySelectionBound: q and p must be positive')
  return (q * q) / ((2 * threshold - 1) * p)
}

/**
 * The selection threshold that guarantees at most `expectedFalse` falsely selected variables in expectation: the solution
 * of q^2 / ((2 pi - 1) p) = E[V], that is pi = (1 + q^2 / (p E[V])) / 2.
 *
 * @param q - The average number selected per subsample.
 * @param p - The number of variables.
 * @param expectedFalse - The tolerated expected number of false selections.
 * @returns pi.
 * @throws OmaiError with code `value` when even pi = 1 does not reach the target (take a smaller q).
 * @concept feature-selection
 * @example
 * ```ts
 * import { stabilityThreshold } from 'omai-engine/learn'
 * stabilityThreshold(4, 40, 1) // => 0.7
 * ```
 */
export function stabilityThreshold(q: number, p: number, expectedFalse: number): number {
  if (!(expectedFalse > 0)) throw new OmaiError('value', 'stabilityThreshold: expectedFalse must be positive')
  const pi = (1 + (q * q) / (p * expectedFalse)) / 2
  if (pi > 1) throw new OmaiError('value', 'stabilityThreshold: no threshold reaches this target with this q; lower q')
  return pi
}

/** Options of {@link lassoStabilityAlgorithm}. */
export interface LassoStabilityOptions {
  /** Variables to select per subsample: the lasso path is followed down until at least q are active (default sqrt(0.8 p)). */
  readonly q?: number
  /** Number of random half-samples (default 100). */
  readonly subsamples?: number
  /** The selection frequency threshold pi (default 0.8). */
  readonly threshold?: number
  readonly key?: Key
}

/** State of {@link lassoStabilityAlgorithm}. */
export interface LassoStabilityState {
  readonly t: number
  /** For each variable, the number of subsamples that selected it. */
  readonly counts: readonly number[]
  /** The number selected in each subsample. */
  readonly sizes: readonly number[]
}

/** The lasso active set at the first penalty on a descending path with at least q active variables. */
function firstQ(design: Float64Array, y: Float64Array, n: number, p: number, q: number): number[] {
  let amax = 0
  for (let j = 0; j < p; j++) {
    let c = 0
    for (let i = 0; i < n; i++) c += (design[i * p + j] ?? 0) * (y[i] ?? 0)
    amax = Math.max(amax, Math.abs(c))
  }
  let warm: Float64Array | undefined
  let active: number[] = []
  for (let k = 1; k <= 60; k++) {
    const lambda = amax * 0.93 ** k
    const fit = lassoCoordinateDescent(fromStorage(design, [n, p], 'float64'), y, lambda, {
      tol: 1e-8,
      maxIterations: 300,
      ...(warm === undefined ? {} : { x0: warm }),
    })
    warm = Float64Array.from(toFlat(fit.coef))
    active = Array.from(warm.keys()).filter((j) => warm?.[j] !== 0)
    if (active.length >= q) break
  }
  return active
}

/**
 * Stability selection with the lasso as an algorithm, one random half-sample per step: standardise the half-sample, follow
 * the lasso path from the largest penalty down until q variables are active, and count them.
 *
 * @param x - The design `[n, p]`.
 * @param y - The response `[n]`.
 * @param options - q, the number of subsamples, the threshold and the key.
 * @returns The algorithm over {@link LassoStabilityState}.
 * @concept feature-selection
 * @example
 * ```ts
 * import { lassoStabilityAlgorithm } from 'omai-engine/learn'
 * lassoStabilityAlgorithm([[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [3, 1]], [1, 0, 1, 2, 0, 3], { q: 1 }).name // => 'lasso-stability-selection'
 * ```
 */
export function lassoStabilityAlgorithm(
  x: TensorLike,
  y: TensorLike,
  options: LassoStabilityOptions = {},
): Algorithm<LassoStabilityState> {
  const m = matrixView(x, 'lassoStabilitySelection')
  const yv = targetVector(y, 'lassoStabilitySelection', m.n)
  const { n, d: p } = m
  const half = Math.floor(n / 2)
  if (half < 3) throw new OmaiError('value', 'lassoStabilitySelection: needs at least 6 samples')
  const q = options.q ?? Math.max(1, Math.round(Math.sqrt(0.8 * p)))
  const subsamples = options.subsamples ?? 100
  return {
    name: 'lasso-stability-selection',
    config: { n, p, q, subsamples } as Json,
    schema: channels({
      metrics: { size: { description: 'variables selected in the subsample' } },
      tensors: { frequencies: { role: 'state', description: 'selection frequency of each variable so far' } },
    }),
    init: () => ({ t: 0, counts: new Array<number>(p).fill(0), sizes: [] }),
    step: (s, ctx) => {
      const rows = toFlat(subsample(ctx.key, n, half))
      const design = new Float64Array(half * p)
      const yh = new Float64Array(half)
      let ym = 0
      rows.forEach((r, i) => {
        yh[i] = yv[r] ?? 0
        ym += (yv[r] ?? 0) / half
      })
      for (let j = 0; j < p; j++) {
        let mean = 0
        for (const r of rows) mean += (m.data[r * p + j] ?? 0) / half
        let ss = 0
        for (const r of rows) ss += ((m.data[r * p + j] ?? 0) - mean) ** 2
        const sd = Math.sqrt(ss / half) || 1
        rows.forEach((r, i) => {
          design[i * p + j] = ((m.data[r * p + j] ?? 0) - mean) / sd
        })
      }
      for (let i = 0; i < half; i++) yh[i] = (yh[i] ?? 0) - ym
      const chosen = firstQ(design, yh, half, p, q)
      const counts = s.counts.map((c, j) => c + (chosen.includes(j) ? 1 : 0))
      return { t: s.t + 1, counts, sizes: [...s.sizes, chosen.length] }
    },
    done: (s) => (s.t >= subsamples ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { size: s.sizes[s.sizes.length - 1] ?? 0 },
      tensors: { frequencies: { role: 'state', value: s.counts.map((c) => c / Math.max(1, s.t)) } },
    }),
    serialise: (s) => ({ t: s.t, counts: [...s.counts], sizes: [...s.sizes] }),
    deserialise: (j) => j as unknown as LassoStabilityState,
  }
}

/** The outcome of {@link lassoStabilitySelection}. */
export interface StabilitySelectionResult {
  /** Variables whose selection frequency reached the threshold. */
  readonly selected: readonly number[]
  /** The selection frequency of every variable. */
  readonly frequencies: readonly number[]
  /** The average number selected per subsample (the q of the bound). */
  readonly meanSize: number
  /** The Meinshausen-Buhlmann bound on the expected number of false selections at this threshold. */
  readonly expectedFalseBound: number
  readonly threshold: number
}

/**
 * Lasso stability selection with the error bound ({@link lassoStabilityAlgorithm} run to the end).
 *
 * @param x - The design `[n, p]`.
 * @param y - The response `[n]`.
 * @param options - q, the subsamples, the threshold and the key.
 * @returns The stable variables, all frequencies and the bound on the expected number of false selections.
 * @concept feature-selection
 * @example
 * ```ts
 * import { lassoStabilitySelection } from 'omai-engine/learn'
 * import { key, normal } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const z = toFlat(normal(key(1), [60 * 5]))
 * const x = Array.from({ length: 60 }, (_, i) => Array.from({ length: 5 }, (_, j) => z[i * 5 + j] ?? 0))
 * const y = x.map((r) => 3 * (r[1] ?? 0))
 * lassoStabilitySelection(x, y, { q: 1, subsamples: 20 }).selected // => [1]
 * ```
 */
export function lassoStabilitySelection(
  x: TensorLike,
  y: TensorLike,
  options: LassoStabilityOptions = {},
): StabilitySelectionResult {
  const alg = lassoStabilityAlgorithm(x, y, options)
  const subsamples = options.subsamples ?? 100
  const r = runAlgorithm(alg, { maxSteps: subsamples, key: options.key ?? makeKey(0) })
  const threshold = options.threshold ?? 0.8
  const p = r.state.counts.length
  const frequencies = r.state.counts.map((c) => c / subsamples)
  const meanSize = r.state.sizes.reduce((a, b) => a + b, 0) / Math.max(1, r.state.sizes.length)
  return {
    selected: frequencies.flatMap((f, j) => (f >= threshold ? [j] : [])),
    frequencies,
    meanSize,
    expectedFalseBound: stabilitySelectionBound(meanSize, p, threshold),
    threshold,
  }
}
