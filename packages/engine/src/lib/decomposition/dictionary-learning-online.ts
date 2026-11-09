/**
 * Online dictionary learning (Mairal, Bach, Ponce and Sapiro 2009; scikit-learn's `MiniBatchDictionaryLearning` and
 * `dict_learning_online`): the atoms learnt from minibatches of the signals, one batch per step. Each step codes its
 * batch over the current atoms, folds the batch into the running statistics `A = sum c c^T` and `B = sum x c^T` (with
 * scikit-learn's forgetting of old batches), and moves every atom by one pass of block coordinate descent on those
 * statistics, so the cost of a step does not grow with the number of signals seen. The run stops, as scikit-learn's,
 * when the atoms barely move or a smoothed estimate of the cost stops improving. It is a traced `Algorithm` observing
 * the cost, the sparsity of the batch's codes, the coherence of the atoms and how far they moved, with the atoms as a
 * tensor, so a view can play them settling.
 */

import { OmaiError } from '../base/index.ts'
import { componentsRecord } from '../exposition/index.ts'
import { type IterativeEstimator, iterative } from '../learn/index.ts'
import { type Key, permutation } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  asRows,
  coherenceOf,
  type DictionaryVerbs,
  dictionaryContract,
  dictionaryVerbs,
  encode,
  type FitCoding,
  meanNonzeros,
  objectiveOf,
  penaltyWeight,
  type Rows,
  randomDictionary,
  sampleDictionary,
  svdDictionary,
  updateAtoms,
} from './_dictionary.ts'
import { mm, tr } from './_linear.ts'
import type { DictionaryInit } from './dictionary-learning.ts'
import { type SparseCodeAlgorithm, sparseCode } from './sparse-coding.ts'

/** The state of online dictionary learning after a minibatch. */
export interface MiniBatchState {
  /** The atoms `[k, d]`. */
  readonly dictionary: number[][]
  /** The running statistic `A = sum c c^T` `[k, k]` (with forgetting). */
  readonly a: number[][]
  /** The running statistic `B = sum x c^T` `[d, k]` (with forgetting). */
  readonly b: number[][]
  /** The order the signals are visited in (shuffled once, then cycled through in batches). */
  readonly order: number[]
  /** Minibatches done. */
  readonly step: number
  /** The codes of the last batch `[m, k]`. */
  readonly code: number[][]
  /** The cost per signal of the last batch, coded over the atoms before their update (scikit-learn's `batch_cost`). */
  readonly batchCost: number
  /** The cost per signal that is monitored: the batch's, or all signals' (`monitor: 'full'`) at the updated atoms. */
  readonly objective: number
  /** The exponentially weighted average of the batch cost (`null` during the first epoch or 100 steps). */
  readonly ewaCost: number | null
  readonly ewaMin: number | null
  /** Steps in a row without an improvement of the smoothed cost. */
  readonly noImprovement: number
  readonly sparsity: number
  readonly coherence: number
  /** The Frobenius norm of the atoms' change over the number of atoms (scikit-learn's convergence quantity). */
  readonly change: number
  readonly unused: number
  readonly stopped: boolean
}

/** Options of {@link miniBatchDictionaryLearningAlgorithm}. */
export interface MiniBatchOptions {
  /** The penalty of the lasso coders (default 1). */
  readonly alpha?: number
  readonly fitAlgorithm?: SparseCodeAlgorithm
  readonly fitNonzeroCoefs?: number | null
  /** Signals per minibatch (default 256, at most the number of signals). */
  readonly batchSize?: number
  /** Shuffle the signals once, with the run's key, before cycling through them (default true). */
  readonly shuffle?: boolean
  /** Stop when the atoms' change falls to this (default 1e-3; 0 never). */
  readonly tol?: number
  /** Stop after this many steps without an improvement of the smoothed cost (default 10; `null` never). */
  readonly maxNoImprovement?: number | null
  readonly positiveCode?: boolean
  readonly positiveDict?: boolean
  readonly fitMaxIter?: number
  /** What `objective` measures: the batch (default, free) or all the signals (costs a coding of all of them per step). */
  readonly monitor?: 'batch' | 'full'
}

const SCHEMA = channels({
  metrics: {
    objective: {
      description: 'cost per signal: half the squared residual plus the penalty on the codes',
      direction: 'lower',
    },
    batchCost: { description: 'cost per signal of the minibatch, before the atoms are updated', direction: 'lower' },
    sparsity: { description: 'mean number of nonzero codes per signal of the batch' },
    coherence: { description: 'largest absolute cosine between two atoms' },
    change: { description: 'norm of the atoms change over the number of atoms' },
    unused: { description: 'atoms replaced because the statistics show no use of them' },
  },
  tensors: {
    dictionary: { role: 'param', description: 'the atoms, one per row', shape: ['k', 'd'] },
    codes: { role: 'state', description: 'the codes of the last minibatch', shape: ['m', 'k'] },
  },
})

/**
 * Online dictionary learning as a traced {@link Algorithm}, one minibatch per step (scikit-learn's
 * `MiniBatchDictionaryLearning`, from the same convergence tests and statistics). The signals are visited in a fixed
 * order (shuffled once from the run's key) in consecutive batches of `batchSize`, the last of an epoch possibly
 * smaller, cycling.
 *
 * @param x - Signals `[n, d]`.
 * @param dictionary0 - The starting atoms `[k, d]`.
 * @param options - Penalty, coder, batch size, shuffling, stopping rules and positivity.
 * @returns The algorithm.
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { miniBatchDictionaryLearningAlgorithm } from 'omai-engine/decomposition'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[2, 0], [0, 3], [4, 0], [0, 1], [1, 0], [0, 2]]
 * const alg = miniBatchDictionaryLearningAlgorithm(x, [[1, 0.2], [0.2, 1]], { batchSize: 3, alpha: 0.1 })
 * runAlgorithm(alg, { maxSteps: 4 }).state.step // => 4
 * ```
 */
export function miniBatchDictionaryLearningAlgorithm(
  x: Rows,
  dictionary0: Rows,
  options: MiniBatchOptions = {},
): Algorithm<MiniBatchState> {
  const n = x.length
  const d = x[0]?.length ?? 0
  const k = dictionary0.length
  const tol = options.tol ?? 1e-3
  const maxNoImprovement = options.maxNoImprovement === undefined ? 10 : options.maxNoImprovement
  const shuffle = options.shuffle ?? true
  const monitor = options.monitor ?? 'batch'
  const batch = Math.max(1, Math.min(options.batchSize ?? 256, n))
  const batches = Math.ceil(n / batch)
  const coding: FitCoding = {
    fitAlgorithm: options.fitAlgorithm ?? 'lassoLars',
    alpha: options.alpha ?? 1,
    fitNonzeroCoefs: options.fitNonzeroCoefs ?? null,
    positiveCode: options.positiveCode ?? false,
    codeMaxIter: options.fitMaxIter ?? 1000,
  }
  const penalty = penaltyWeight(coding)
  const positiveDict = options.positiveDict ?? false
  const zeros = (r: number, c: number): number[][] => Array.from({ length: r }, () => new Array<number>(c).fill(0))
  return {
    name: 'mini-batch-dictionary-learning',
    config: { ...coding, batchSize: batch, shuffle, tol, maxNoImprovement, positiveDict, monitor },
    schema: SCHEMA,
    init: (key) => ({
      dictionary: dictionary0.map((r) => [...r]),
      a: zeros(k, k),
      b: zeros(d, k),
      order: shuffle ? Array.from(permutation(key, n).data, Number) : Array.from({ length: n }, (_, i) => i),
      step: 0,
      code: [],
      batchCost: 0,
      objective: 0,
      ewaCost: null,
      ewaMin: null,
      noImprovement: 0,
      sparsity: 0,
      coherence: coherenceOf(dictionary0),
      change: 0,
      unused: 0,
      stopped: false,
    }),
    step: (s, ctx) => {
      const i = s.step
      const at = (i % batches) * batch
      const idx = s.order.slice(at, Math.min(at + batch, n))
      const xb = idx.map((r) => x[r] ?? [])
      const m = xb.length
      const code = encode(xb, s.dictionary, coding)
      const before = objectiveOf(xb, s.dictionary, code, penalty)
      const batchCost = before.objective / m
      // Mairal's running statistics with scikit-learn's forgetting (theta, beta) of the old batches.
      const theta = i < m - 1 ? (i + 1) * m : m * m + i + 1 - m
      const beta = (theta + 1 - m) / (theta + 1)
      const cc = mm(tr(code), code)
      const xc = mm(tr(xb), code)
      const a = s.a.map((row, p) => row.map((v, q) => beta * v + (cc[p]?.[q] ?? 0) / m))
      const b = s.b.map((row, p) => row.map((v, q) => beta * v + (xc[p]?.[q] ?? 0) / m))
      const next = updateAtoms(s.dictionary, xb, code, a, b, ctx.key, positiveDict)
      const objective =
        monitor === 'full'
          ? objectiveOf(x, next.dictionary, encode(x, next.dictionary, coding), penalty).objective / n
          : batchCost
      // The stopping tests of scikit-learn: skipped during the first epoch or 100 steps, then the change of the atoms
      // and the lack of improvement of a smoothed cost.
      let ewaCost = s.ewaCost
      let ewaMin = s.ewaMin
      let noImprovement = s.noImprovement
      let stopped = false
      let change = 0
      for (const [p, row] of next.dictionary.entries())
        for (const [q, v] of row.entries()) change += (v - (s.dictionary[p]?.[q] ?? 0)) ** 2
      change = Math.sqrt(change) / Math.max(k, 1)
      if (i + 1 > Math.min(100, n / m)) {
        ewaCost =
          ewaCost === null ? batchCost : ewaCost * (1 - Math.min(m / (n + 1), 1)) + batchCost * Math.min(m / (n + 1), 1)
        if (tol > 0 && change <= tol) stopped = true
        else {
          if (ewaMin === null || ewaCost < ewaMin) {
            noImprovement = 0
            ewaMin = ewaCost
          } else noImprovement++
          if (maxNoImprovement !== null && noImprovement >= maxNoImprovement) stopped = true
        }
      }
      return {
        dictionary: next.dictionary,
        a,
        b,
        order: s.order,
        step: i + 1,
        code: next.code,
        batchCost,
        objective,
        ewaCost,
        ewaMin,
        noImprovement,
        sparsity: meanNonzeros(code),
        coherence: coherenceOf(next.dictionary),
        change,
        unused: next.unused,
        stopped,
      }
    },
    done: (s) => (s.stopped ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        objective: s.objective,
        batchCost: s.batchCost,
        sparsity: s.sparsity,
        coherence: s.coherence,
        change: s.change,
        unused: s.unused,
      },
      tensors: {
        dictionary: { role: 'param', value: s.dictionary.flat(), shape: [k, d] },
        codes: { role: 'state', value: s.code.flat(), shape: [s.code.length, k] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MiniBatchState,
  }
}

/** Hyperparameters of {@link miniBatchDictionaryLearning}. */
export interface MiniBatchDictionaryLearningHyper {
  /** The number of atoms (`null`: the number of features). */
  readonly nComponents: number | null
  readonly alpha: number
  /** Epochs: the run takes at most `maxIter` times the number of batches in a pass over the signals. */
  readonly maxIter: number
  readonly batchSize: number
  readonly shuffle: boolean
  readonly tol: number
  readonly maxNoImprovement: number | null
  readonly fitAlgorithm: SparseCodeAlgorithm
  readonly fitNonzeroCoefs: number | null
  readonly fitMaxIter: number
  readonly monitor: 'batch' | 'full'
  /** How the atoms start (scikit-learn's randomised SVD is the exact SVD here, with zero atoms past the rank). */
  readonly init: DictionaryInit
  readonly dictInit: number[][] | null
  readonly transformAlgorithm: SparseCodeAlgorithm
  readonly transformNonzeroCoefs: number | null
  readonly transformAlpha: number | null
  readonly transformMaxIter: number
  readonly splitSign: boolean
  readonly positiveCode: boolean
  readonly positiveDict: boolean
}

/** The fitted parameters of {@link miniBatchDictionaryLearning}. */
export type MiniBatchDictionaryLearningParams = {
  readonly components: number[][]
  /** Minibatches taken. */
  readonly nSteps: number
  /** Passes over the signals, rounded up (scikit-learn's `n_iter_`). */
  readonly nIter: number
  /** The cost per signal monitored at the last step. */
  readonly objective: number
}

/**
 * Online dictionary learning (sklearn.decomposition.MiniBatchDictionaryLearning) on the iterative base: one minibatch
 * per step, so a session shows the atoms settling batch by batch, and `fittedAt(t)` is the model after `t` batches.
 * `transform` codes new signals with `transformAlgorithm` (OMP by default), `inverseTransform` writes signals back.
 * Where scikit-learn starts from a randomised SVD, this starts from the exact one; `partial_fit` is a session advanced
 * a step at a time.
 *
 * @index estimator Online dictionary learning from minibatches with running statistics.
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { miniBatchDictionaryLearning } from 'omai-engine/decomposition'
 * const x = [[2, 0, 1], [0, 3, 0], [4, 0, 2], [0, -1, 0], [1, 1, 0.5], [2, 3, 1]]
 * const f = miniBatchDictionaryLearning.with({ nComponents: 2, alpha: 0.1, batchSize: 3, maxIter: 5 }).fit({ x })
 * f.params.components.length // => 2
 * ```
 */
export const miniBatchDictionaryLearning: IterativeEstimator<
  MiniBatchDictionaryLearningHyper,
  MiniBatchState,
  MiniBatchDictionaryLearningParams,
  DictionaryVerbs
> = iterative<MiniBatchDictionaryLearningHyper, MiniBatchState, MiniBatchDictionaryLearningParams, DictionaryVerbs>({
  name: 'decomposition/mini-batch-dictionary-learning',
  kind: 'transformer',
  defaults: {
    nComponents: null,
    alpha: 1,
    maxIter: 1000,
    batchSize: 256,
    shuffle: true,
    tol: 1e-3,
    maxNoImprovement: 10,
    fitAlgorithm: 'lassoLars',
    fitNonzeroCoefs: null,
    fitMaxIter: 1000,
    monitor: 'batch',
    init: 'svd',
    dictInit: null,
    transformAlgorithm: 'omp',
    transformNonzeroCoefs: null,
    transformAlpha: null,
    transformMaxIter: 1000,
    splitSign: false,
    positiveCode: false,
    positiveDict: false,
  },
  hyperSpace: (h) =>
    dict({
      ...(h.nComponents === null ? {} : { nComponents: int(1, 100, { default: h.nComponents, label: 'atoms' }) }),
      alpha: real(0.001, 100, { log: true, default: h.alpha, label: 'penalty' }),
      batchSize: int(1, 1000, { default: h.batchSize, label: 'batch size' }),
      fitAlgorithm: choice(['lassoLars', 'lassoCd', 'omp', 'lars', 'threshold'] as const, {
        default: h.fitAlgorithm,
        label: 'coder',
      }),
      maxIter: int(1, 1000, { default: h.maxIter, label: 'epochs' }),
    }),
  algorithm: (data, h, key) => {
    const x = asRows(data.x, 'miniBatchDictionaryLearning')
    const k = h.nComponents ?? x[0]?.length ?? 0
    const d = x[0]?.length ?? 0
    let start: number[][]
    if (h.init === 'custom') {
      if (h.dictInit === null) throw new OmaiError('value', "miniBatchDictionaryLearning: init 'custom' needs dictInit")
      if (h.dictInit.length !== k || h.dictInit.some((r) => r.length !== d))
        throw new OmaiError('shape', `miniBatchDictionaryLearning: dictInit must be [${k}, ${d}]`)
      start = h.dictInit
    } else if (h.init === 'random') start = randomDictionary(k, d, key)
    else if (h.init === 'samples') start = sampleDictionary(x, k, key)
    else start = svdDictionary(x, k)
    return miniBatchDictionaryLearningAlgorithm(x, start, h)
  },
  contract: () => dictionaryContract<MiniBatchState>(),
  maxSteps: (h, data) => {
    const n = asRows(data.x, 'miniBatchDictionaryLearning').length
    return h.maxIter * Math.ceil(n / Math.max(1, Math.min(h.batchSize, n)))
  },
  params: (s, data, h) => {
    const n = asRows(data.x, 'miniBatchDictionaryLearning').length
    return {
      components: s.dictionary,
      nSteps: s.step,
      nIter: Math.ceil(s.step / Math.ceil(n / Math.max(1, Math.min(h.batchSize, n)))),
      objective: s.objective,
    }
  },
  records: (s) => [
    componentsRecord({
      method: 'mini-batch dictionary learning',
      features: Array.from({ length: s.dictionary[0]?.length ?? 0 }, (_, q) => `x${q}`),
      components: s.dictionary,
    }),
  ],
  verbs: (p, h) => dictionaryVerbs(p.components, h),
})

/** The result of {@link dictLearningOnline}. */
export interface DictLearningOnlineResult {
  /** The lasso codes `[n, k]` of the signals over the learnt atoms. */
  readonly code: number[][]
  readonly dictionary: number[][]
  /** Minibatches taken. */
  readonly nSteps: number
}

/**
 * Online dictionary learning as a function (sklearn.decomposition.dict_learning_online): the atoms learnt from
 * minibatches and the lasso codes of the signals over them (`method` `lars` or `cd`, with the penalty `alpha`).
 *
 * @param x - Signals `[n, d]`.
 * @param nComponents - The number of atoms.
 * @param options - Everything {@link MiniBatchOptions} has, `maxIter` epochs (default 100), `method` (default `lars`), `dictInit`, and the key of the shuffle.
 * @returns The codes, the atoms and the number of minibatches.
 * @concept dictionary-learning
 * @index function Online dictionary learning from minibatches, with the lasso codes, as a function.
 * @example
 * ```ts
 * import { dictLearningOnline } from 'omai-engine/decomposition'
 * const x = [[2, 0], [0, 3], [4, 0], [0, 1], [1, 0], [0, 2]]
 * const r = dictLearningOnline(x, 2, { alpha: 0.1, batchSize: 3, maxIter: 5 })
 * [r.dictionary.length, r.code.length] // => [2, 6]
 * ```
 */
export function dictLearningOnline(
  x: Rows,
  nComponents: number,
  options: Omit<MiniBatchOptions, 'fitAlgorithm'> & {
    readonly maxIter?: number
    readonly method?: 'lars' | 'cd'
    readonly dictInit?: Rows
    readonly key?: Key
  } = {},
): DictLearningOnlineResult {
  const fitAlgorithm = options.method === 'cd' ? 'lassoCd' : 'lassoLars'
  const start = options.dictInit?.map((r) => [...r]) ?? svdDictionary(x, nComponents)
  const n = x.length
  const batch = Math.max(1, Math.min(options.batchSize ?? 256, n))
  const out = runAlgorithm(miniBatchDictionaryLearningAlgorithm(x, start, { ...options, fitAlgorithm }), {
    maxSteps: (options.maxIter ?? 100) * Math.ceil(n / batch),
    ...(options.key === undefined ? {} : { key: options.key }),
  })
  const dictionary = out.state.dictionary
  const code = sparseCode(x as number[][], dictionary, { algorithm: fitAlgorithm, alpha: options.alpha ?? 1 })
  return { code: asRows(code, 'dictLearningOnline'), dictionary, nSteps: out.state.step }
}
