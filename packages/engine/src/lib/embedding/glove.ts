/**
 * GloVe (Pennington, Socher and Manning 2014): word vectors by weighted least squares on the logarithm of the
 * co-occurrence counts, `J = sum f(X_ij) (w_i . w~_j + b_i + b~_j - log X_ij)^2 / 2` over the nonzero counts, with the
 * weight `f(x) = min((x / xmax)^alpha, 1)`. The loss is a smooth problem over the two matrices and two bias vectors,
 * minimised by the solver slot, of which GloVe's own training (AdaGrad, one co-occurrence at a time in a fresh random
 * order each pass) is the named mode `gloveAdaGrad` beside the generic solvers. The word vector is `w + w~`.
 */

import { type Json, OmaiError } from '../base/index.ts'
import type { BatchObjective, SmoothProblem, VectorParams } from '../optim/index.ts'
import { foldIn, type Key, permutation, uniform } from '../random/index.ts'
import { dict, int, real, type SpaceValue } from '../space/index.ts'
import { add, fromStorage, item, mul, reshape, sliceAxis, square, sub, sum, take, type Value } from '../tensor/index.ts'
import { cooccurrence, type WordMatrix } from '../text/index.ts'
import { type Algorithm, type ModeSpec, type ModeTable, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { solversWith } from './word2vec.ts'

/** The knobs of the GloVe model and its co-occurrence counts. */
export interface GloveOptions {
  /** The embedding dimension (default 50). */
  readonly dimension?: number
  /** Context words on each side (default 5). */
  readonly window?: number
  /** The weight function's cut-off count (default 100). */
  readonly xmax?: number
  /** The weight function's exponent (default 0.75). */
  readonly alpha?: number
  /** Words seen fewer times are dropped (default 1). */
  readonly minCount?: number
}

/** A GloVe problem: the smooth loss over `[W, W~, b, b~]` and the nonzero co-occurrences it sums over. */
export interface GloveProblem extends SmoothProblem {
  readonly words: readonly string[]
  readonly dimension: number
  /** The nonzero entries: row word, column word, count. */
  readonly rows: Int32Array
  readonly cols: Int32Array
  readonly counts: Float64Array
  readonly xmax: number
  readonly alpha: number
}

/** The weight function of GloVe. */
const weight = (x: number, xmax: number, alpha: number): number => (x < xmax ? (x / xmax) ** alpha : 1)

/**
 * The GloVe loss as a {@link SmoothProblem}: the mean over the nonzero entries of the symmetric harmonic-weighted
 * co-occurrence matrix (`1 / distance` weighting inside the window, as GloVe counts) of `f(X) (w_i . w~_j + b_i +
 * b~_j - log X)^2 / 2`, over the flat vector `[W, W~, b, b~]`. The start is GloVe's: every parameter uniform in
 * `[-0.5 / d, 0.5 / d]`.
 *
 * @param sentences - Token lists.
 * @param options - Dimension, window, weight function and cut-off.
 * @param key - The key for the start.
 * @returns The problem.
 * @throws OmaiError with code `value` for a corpus without co-occurrences.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { gloveProblem } from 'omai-engine/embedding'
 * import { key } from 'omai-engine/random'
 * const p = gloveProblem([['a', 'b', 'c', 'a', 'b']], { dimension: 2, window: 2 }, key(0))
 * [p.words, p.batches?.n] // => [['a', 'b', 'c'], 6]
 * ```
 */
export function gloveProblem(sentences: readonly (readonly string[])[], options: GloveOptions, key: Key): GloveProblem {
  const d = options.dimension ?? 50
  const xmax = options.xmax ?? 100
  const alpha = options.alpha ?? 0.75
  const m = cooccurrence(sentences, {
    window: options.window ?? 5,
    weighting: 'harmonic',
    minCount: options.minCount ?? 1,
  })
  const V = m.words.length
  const rows: number[] = []
  const cols: number[] = []
  const counts: number[] = []
  for (const [i, r] of m.rows.entries())
    for (let j = 0; j < V; j++) {
      if ((r[j] ?? 0) <= 0) continue
      rows.push(i)
      cols.push(j)
      counts.push(r[j] as number)
    }
  const n = counts.length
  if (n === 0) throw new OmaiError('value', 'glove: the corpus has no co-occurrences')
  const R = Int32Array.from(rows)
  const C = Int32Array.from(cols)
  const X = Float64Array.from(counts)
  const x0 = Float64Array.from(uniform(foldIn(key, 0), [2 * V * d + 2 * V]).data, (u) => (u - 0.5) / d)
  const objective: BatchObjective = (params, batch) => {
    const x = params as Value
    const idx = batch ?? Array.from({ length: n }, (_, i) => i)
    const b = idx.length
    const ri = fromStorage(
      Int32Array.from(idx, (e) => R[e] as number),
      [b],
      'int32',
    )
    const ci = fromStorage(
      Int32Array.from(idx, (e) => C[e] as number),
      [b],
      'int32',
    )
    const logX = fromStorage(
      Float64Array.from(idx, (e) => Math.log(X[e] as number)),
      [b],
      'float64',
    )
    const f = fromStorage(
      Float64Array.from(idx, (e) => weight(X[e] as number, xmax, alpha)),
      [b],
      'float64',
    )
    const W = reshape(sliceAxis(x, 0, 0, V * d), [V, d])
    const Wt = reshape(sliceAxis(x, 0, V * d, 2 * V * d), [V, d])
    const bias = sliceAxis(x, 0, 2 * V * d, 2 * V * d + V)
    const biasT = sliceAxis(x, 0, 2 * V * d + V, 2 * V * d + 2 * V)
    const dot = sum(mul(take(W, ri, 0), take(Wt, ci, 0)), -1)
    const diff = sub(add(add(dot, take(bias, ri, 0)), take(biasT, ci, 0)), logX)
    return mul(sum(mul(f, square(diff))), 0.5 / b)
  }
  return { objective, x0, batches: { n }, words: m.words, dimension: d, rows: R, cols: C, counts: X, xmax, alpha }
}

/** The state of GloVe's AdaGrad training. */
export interface GloveAdaGradState {
  readonly x: Float64Array
  /** The AdaGrad accumulators of squared gradients, one per parameter, starting at 1. */
  readonly accumulated: Float64Array
  readonly value: number
  readonly grad: Float64Array
  readonly step: number
  readonly lr: number
  readonly t: number
  readonly order: Int32Array
}

/** Options of {@link gloveAdaGradAlgorithm}. */
export interface GloveAdaGradOptions {
  /** The AdaGrad learning rate (default 0.05). */
  readonly learningRate?: number
  /** Passes over the nonzero entries (default 25). */
  readonly epochs?: number
  /** Entries per step, each updated in turn (default 1). */
  readonly batch?: number
}

/**
 * GloVe's own training as an {@link Algorithm}: AdaGrad, one co-occurrence at a time in a fresh random order each pass,
 * each parameter stepping by `lr * g / sqrt(accumulated g^2)` with the accumulators starting at 1. Each step takes
 * `batch` entries in turn. The per-entry gradient is the exact gradient of {@link gloveProblem}'s loss for the entry.
 *
 * @param problem - The problem.
 * @param options - Learning rate, epochs and batch.
 * @returns The algorithm; its state's arrays are updated in place.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { gloveAdaGradAlgorithm, gloveProblem } from 'omai-engine/embedding'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const p = gloveProblem([['a', 'b', 'c', 'a', 'b']], { dimension: 2, window: 2 }, key(0))
 * runAlgorithm(gloveAdaGradAlgorithm(p, { epochs: 3 }), { seed: 0, maxSteps: 1000 }).steps // => 18
 * ```
 */
export function gloveAdaGradAlgorithm(
  problem: GloveProblem,
  options: GloveAdaGradOptions = {},
): Algorithm<GloveAdaGradState> {
  const lr = options.learningRate ?? 0.05
  const epochs = options.epochs ?? 25
  const batch = options.batch ?? 1
  const { rows, cols, counts, xmax, alpha, dimension: d } = problem
  const V = problem.words.length
  const n = counts.length
  const perEpoch = Math.ceil(n / batch)
  if (!(lr > 0) || !Number.isInteger(epochs) || epochs < 1 || !Number.isInteger(batch) || batch < 1)
    throw new OmaiError('value', 'gloveAdaGrad: need a positive learning rate, epochs and batch')
  const tOff = V * d
  const bOff = 2 * V * d
  const btOff = bOff + V
  return {
    name: 'gloveAdaGrad',
    config: { learningRate: lr, epochs, batch, entries: n } as Json,
    init: (key) => ({
      x: Float64Array.from(problem.x0 as Float64Array),
      accumulated: new Float64Array(2 * V * d + 2 * V).fill(1),
      value: Number.NaN,
      grad: new Float64Array(1),
      step: 0,
      lr,
      t: 0,
      order: Int32Array.from(permutation(key, n).data),
    }),
    step: (s, ctx) => {
      const order = s.t > 0 && s.t % perEpoch === 0 ? Int32Array.from(permutation(ctx.key, n).data) : s.order
      const start = (s.t % perEpoch) * batch
      const { x, accumulated: acc } = s
      let loss = 0
      let gsq = 0
      let biggest = 0
      let seen = 0
      for (let q = start; q < Math.min(n, start + batch); q++) {
        const e = order[q] as number
        const i = rows[e] as number
        const j = cols[e] as number
        const X = counts[e] as number
        const f = weight(X, xmax, alpha)
        let diff = (x[bOff + i] ?? 0) + (x[btOff + j] ?? 0) - Math.log(X)
        for (let c = 0; c < d; c++) diff += (x[i * d + c] ?? 0) * (x[tOff + j * d + c] ?? 0)
        loss += 0.5 * f * diff * diff
        const fd = f * diff
        seen++
        const update = (at: number, g: number): void => {
          const u = (lr * g) / Math.sqrt(acc[at] ?? 1)
          x[at] = (x[at] ?? 0) - u
          acc[at] = (acc[at] ?? 1) + g * g
          biggest = Math.max(biggest, Math.abs(u))
          gsq += g * g
        }
        for (let c = 0; c < d; c++) {
          const wi = x[i * d + c] ?? 0
          const wj = x[tOff + j * d + c] ?? 0
          update(i * d + c, fd * wj)
          update(tOff + j * d + c, fd * wi)
        }
        update(bOff + i, fd)
        update(btOff + j, fd)
      }
      return {
        x,
        accumulated: acc,
        value: loss / Math.max(1, seen),
        grad: Float64Array.of(Math.sqrt(gsq / Math.max(1, seen))),
        step: biggest,
        lr,
        t: s.t + 1,
        order,
      }
    },
    done: (s) => (s.t >= perEpoch * epochs ? 'converged' : undefined),
    observe: (s) => ({ metrics: { loss: s.value } }),
    serialise: (s) =>
      ({
        x: Array.from(s.x),
        accumulated: Array.from(s.accumulated),
        value: s.value,
        step: s.step,
        lr: s.lr,
        t: s.t,
        order: Array.from(s.order),
      }) as unknown as Json,
    deserialise: (j) => {
      const o = j as unknown as {
        x: number[]
        accumulated: number[]
        value: number | null
        step: number
        lr: number
        t: number
        order: number[]
      }
      return {
        x: Float64Array.from(o.x),
        accumulated: Float64Array.from(o.accumulated),
        value: o.value ?? Number.NaN,
        grad: new Float64Array(1),
        step: o.step,
        lr: o.lr,
        t: o.t,
        order: Int32Array.from(o.order),
      }
    },
  }
}

const gloveAdaGrad: ModeSpec<GloveProblem, GloveAdaGradState, VectorParams> = {
  description: "GloVe's own training: AdaGrad, one co-occurrence at a time, a fresh order each pass",
  space: dict({
    learningRate: real(1e-4, 1, { log: true, default: 0.05, label: 'learning rate' }),
    epochs: int(1, 10_000, { default: 25, label: 'passes over the entries' }),
    batch: int(1, 100_000, { default: 1, label: 'entries per step' }),
  }),
  make: (p, k) =>
    gloveAdaGradAlgorithm(p, {
      learningRate: k['learningRate'] as number,
      epochs: k['epochs'] as number,
      batch: k['batch'] as number,
    }),
  params: (s) => ({ x: Array.from(s.x) }),
}

/** The solvers of the GloVe loss: its own AdaGrad (`gloveAdaGrad`, the default) and the generic smooth solvers. */
export const gloveSolvers: ModeTable<GloveProblem, VectorParams> = solversWith({ gloveAdaGrad })

/** A trained GloVe model. */
export interface GloveModel {
  readonly words: readonly string[]
  /** The word vectors `w + w~`. */
  readonly vectors: WordMatrix
  /** The two matrices and the biases, separately. */
  readonly wordVectors: readonly Float64Array[]
  readonly contextVectors: readonly Float64Array[]
  readonly biases: readonly number[]
  readonly contextBiases: readonly number[]
  /** The objective over all entries at the end. */
  readonly loss: number
  readonly steps: number
}

/** Hyperparameters of {@link trainGlove}. */
export interface GloveHyper extends GloveOptions {
  /** The solver, a value of {@link gloveSolvers}'s space (default GloVe's AdaGrad). */
  readonly solver?: SpaceValue
  /** The step budget (default 1,000,000). */
  readonly maxSteps?: number
}

/**
 * The model that a flat parameter vector of a problem stands for.
 *
 * @param problem - The problem.
 * @param x - The flat parameters.
 * @param extra - The loss and steps to record.
 * @returns The model.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { gloveModel, gloveProblem } from 'omai-engine/embedding'
 * import { key } from 'omai-engine/random'
 * const p = gloveProblem([['a', 'b', 'c']], { dimension: 2, window: 1 }, key(0))
 * gloveModel(p, p.x0 as Float64Array).vectors.rows.length // => 3
 * ```
 */
export function gloveModel(
  problem: GloveProblem,
  x: ArrayLike<number>,
  extra: { readonly loss?: number; readonly steps?: number } = {},
): GloveModel {
  const d = problem.dimension
  const V = problem.words.length
  const part = (off: number): Float64Array[] =>
    Array.from({ length: V }, (_, r) => Float64Array.from({ length: d }, (_, c) => x[off + r * d + c] ?? 0))
  const w = part(0)
  const wt = part(V * d)
  return {
    words: problem.words,
    vectors: {
      words: problem.words,
      rows: w.map((r, i) => Float64Array.from(r, (v, c) => v + ((wt[i] as Float64Array)[c] ?? 0))),
    },
    wordVectors: w,
    contextVectors: wt,
    biases: Array.from({ length: V }, (_, i) => x[2 * V * d + i] ?? 0),
    contextBiases: Array.from({ length: V }, (_, i) => x[2 * V * d + V + i] ?? 0),
    loss: extra.loss ?? Number.NaN,
    steps: extra.steps ?? 0,
  }
}

/**
 * Train GloVe: count the harmonic-weighted co-occurrences and run the solver of the slot on the weighted least-squares
 * loss.
 *
 * @param sentences - Token lists.
 * @param hyper - The model, the solver and the budget.
 * @param key - The key for the start and the passes.
 * @param run - Run options (`tracer`, `policy`) to record the solver's steps.
 * @returns The model.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { trainGlove } from 'omai-engine/embedding'
 * import { key } from 'omai-engine/random'
 * const m = trainGlove([['a', 'b', 'c', 'a', 'b', 'c']], { dimension: 3, window: 2, solver: { kind: 'gloveAdaGrad', epochs: 5 } }, key(0))
 * m.words // => ['a', 'b', 'c']
 * ```
 */
export function trainGlove(
  sentences: readonly (readonly string[])[],
  hyper: GloveHyper,
  key: Key,
  run: Partial<Omit<RunOptions, 'maxSteps' | 'seed' | 'key'>> = {},
): GloveModel {
  const problem = gloveProblem(sentences, hyper, foldIn(key, 0))
  const solver = hyper.solver ?? { kind: 'gloveAdaGrad' }
  const result = runAlgorithm(gloveSolvers.make(problem, solver, foldIn(key, 1)), {
    ...run,
    key: foldIn(key, 2),
    maxSteps: hyper.maxSteps ?? 1_000_000,
  })
  const x = gloveSolvers.params(solver, result.state).x
  const loss = item(problem.objective(fromStorage(Float64Array.from(x), [x.length], 'float64')) as Value)
  return gloveModel(problem, x, { loss, steps: result.steps })
}
