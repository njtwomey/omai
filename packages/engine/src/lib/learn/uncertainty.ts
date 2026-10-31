/**
 * Predictive uncertainty quantification by deep ensembles (Lakshminarayanan, Pritzel and Blundell 2017) and Monte Carlo
 * dropout (Gal and Ghahramani 2016): decomposing total predictive uncertainty into epistemic (model disagreement) and
 * aleatoric (observation noise) components by the law of total variance and mutual information (BALD).
 */

import { MdmlError } from '../base/index.ts'
import { bootstrapIndices, foldIn, type Key } from '../random/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from './_shared.ts'
import { baldAcquisition, predictiveEntropy } from './active-learning.ts'
import {
  type Estimator,
  type FitOptions,
  type Fitted,
  type LearnData,
  requireVerb,
  revive,
  type SerialisedFitted,
  serialiseFitted,
} from './estimator.ts'
import { integerOption, matrixView, takeData, targetVector } from './validation.ts'

/** Decomposition of predictive uncertainty across ensemble members or stochastic passes. */
export interface UncertaintySummary {
  /** Predictive mean: `[n]` for regression, `[n, k]` class probabilities for classification. */
  readonly mean: Tensor<'float64'>
  /** Epistemic variance Var_m(E[y | x, theta_m]), `[n]`. */
  readonly epistemicVariance: Tensor<'float64'>
  /** Aleatoric variance E_m[Var(y | x, theta_m)], `[n]`. */
  readonly aleatoricVariance: Tensor<'float64'>
  /** Total predictive variance: epistemicVariance + aleatoricVariance, `[n]`. */
  readonly totalVariance: Tensor<'float64'>
  /** Predictive Shannon entropy H(E_m[p_m(y | x)]), or the differential entropy of the moment-matched Gaussian for regression, `[n]`. */
  readonly predictiveEntropy: Tensor<'float64'>
  /** Expected conditional entropy E_m[H(p_m(y | x))] (aleatoric entropy), `[n]`; NaN for regression members without variances. */
  readonly expectedEntropy: Tensor<'float64'>
  /** Mutual information I(y; theta | x) = predictiveEntropy - expectedEntropy (BALD), `[n]`; NaN where the expected entropy is. */
  readonly mutualInformation: Tensor<'float64'>
  /** Per-sample member disagreement score, `[n]`. */
  readonly disagreement: Tensor<'float64'>
}

/** Options of {@link deepEnsemblePredict}. */
export interface DeepEnsemblePredictOptions {
  /** Task kind: `'regression'`, `'classification'`, or `'auto'` (default `'auto'`). */
  readonly kind?: 'auto' | 'regression' | 'classification'
  /** Optional per-member aleatoric variances `[M][n]` for heteroscedastic regression members. */
  readonly memberVariances?: readonly TensorLike[]
}

function isClassificationProba(member: TensorLike, options: DeepEnsemblePredictOptions): boolean {
  if (options.kind === 'classification') return true
  if (options.kind === 'regression' || options.memberVariances !== undefined) return false
  const first = asTensor(member)
  if (first.shape.length !== 2 || (first.shape[1] ?? 0) < 2) return false
  const m = matrixView(first, 'deepEnsemblePredict')
  for (let i = 0; i < m.n; i++) {
    let s = 0
    for (let c = 0; c < m.d; c++) {
      const v = m.data[i * m.d + c] ?? 0
      if (v < -1e-6 || v > 1 + 1e-6) return false
      s += v
    }
    if (Math.abs(s - 1) > 1e-3) return false
  }
  return true
}

/**
 * Aggregate predictions or class probabilities from a deep ensemble (Lakshminarayanan et al. 2017) into predictive
 * mean, epistemic variance, aleatoric variance, total variance, predictive entropy, expected entropy and BALD mutual
 * information.
 *
 * For regression, total variance decomposes by the law of total variance as
 * `Var(y | x) = Var_m(mu_m(x)) + E_m[sigma_m^2(x)]`. For classification, member probability vectors `p_m(x)` give the
 * mixture predictive `p_bar(x) = mean_m p_m(x)`, epistemic variance `mean_m sum_c (p_{m,c} - p_bar_c)^2`, aleatoric
 * variance `mean_m sum_c p_{m,c}(1 - p_{m,c})`, predictive entropy `H(p_bar)` and mutual information
 * `H(p_bar) - mean_m H(p_m)`.
 *
 * @param members - Member predictions (`[n]` for regression, `[n, k]` for classification).
 * @param options - Task `kind` and optional per-member regression `memberVariances`.
 * @returns The decomposed {@link UncertaintySummary}.
 * @concept uncertainty-quantification
 * @example
 * ```ts
 * import { deepEnsemblePredict } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const u = deepEnsemblePredict([[1, 2], [3, 2]], { memberVariances: [[0.5, 0.25], [0.5, 0.25]] })
 * toFlat(u.mean) // => [2, 2]
 * toFlat(u.epistemicVariance) // => [1, 0]
 * toFlat(u.aleatoricVariance) // => [0.5, 0.25]
 * toFlat(u.totalVariance) // => [1.5, 0.25]
 * ```
 */
export function deepEnsemblePredict(
  members: readonly TensorLike[],
  options: DeepEnsemblePredictOptions = {},
): UncertaintySummary {
  const M = members.length
  if (M === 0) throw new MdmlError('value', 'deepEnsemblePredict: need at least one ensemble member')
  if (isClassificationProba(members[0] as TensorLike, options)) {
    const views = members.map((m) => matrixView(m, 'deepEnsemblePredict'))
    const n = views[0]?.n ?? 0
    const k = views[0]?.d ?? 0
    for (const v of views) {
      if (v.n !== n || v.d !== k) {
        throw new MdmlError(
          'shape',
          'deepEnsemblePredict: all classification member probabilities must have shape [n, k]',
        )
      }
    }
    const meanData = new Float64Array(n * k)
    const aleatoric = new Float64Array(n)
    const expectedEnt = new Float64Array(n)
    for (const v of views) {
      const ent = predictiveEntropy(fromStorage(v.data, [n, k], 'float64'))
      for (let i = 0; i < n; i++) {
        expectedEnt[i] = (expectedEnt[i] ?? 0) + (ent[i] ?? 0) / M
        let a = 0
        for (let c = 0; c < k; c++) {
          const p = v.data[i * k + c] ?? 0
          meanData[i * k + c] = (meanData[i * k + c] ?? 0) + p / M
          a += p * (1 - p)
        }
        aleatoric[i] = (aleatoric[i] ?? 0) + a / M
      }
    }
    const epistemic = new Float64Array(n)
    for (const v of views) {
      for (let i = 0; i < n; i++) {
        let sq = 0
        for (let c = 0; c < k; c++) {
          const diff = (v.data[i * k + c] ?? 0) - (meanData[i * k + c] ?? 0)
          sq += diff * diff
        }
        epistemic[i] = (epistemic[i] ?? 0) + sq / M
      }
    }
    const total = Float64Array.from({ length: n }, (_, i) => (epistemic[i] ?? 0) + (aleatoric[i] ?? 0))
    const predEnt = Float64Array.from(predictiveEntropy(fromStorage(meanData, [n, k], 'float64')))
    const mi = Float64Array.from(baldAcquisition(members))
    return {
      mean: fromStorage(meanData, [n, k], 'float64'),
      epistemicVariance: fromStorage(epistemic, [n], 'float64'),
      aleatoricVariance: fromStorage(aleatoric, [n], 'float64'),
      totalVariance: fromStorage(total, [n], 'float64'),
      predictiveEntropy: fromStorage(predEnt, [n], 'float64'),
      expectedEntropy: fromStorage(expectedEnt, [n], 'float64'),
      mutualInformation: fromStorage(mi, [n], 'float64'),
      disagreement: fromStorage(Float64Array.from(mi), [n], 'float64'),
    }
  }

  const vecs = members.map((m) => targetVector(m, 'deepEnsemblePredict'))
  const n = vecs[0]?.length ?? 0
  for (const v of vecs) {
    if (v.length !== n) throw new MdmlError('shape', 'deepEnsemblePredict: all member predictions must have length n')
  }
  const vars =
    options.memberVariances === undefined
      ? undefined
      : options.memberVariances.map((v) => {
          const tv = targetVector(v, 'deepEnsemblePredict', n)
          for (const s of tv) if (s < 0) throw new MdmlError('value', 'deepEnsemblePredict: variances must be >= 0')
          return tv
        })
  if (vars !== undefined && vars.length !== M) {
    throw new MdmlError('shape', 'deepEnsemblePredict: memberVariances must match members length')
  }

  const meanData = new Float64Array(n)
  for (const v of vecs) {
    for (let i = 0; i < n; i++) meanData[i] = (meanData[i] ?? 0) + (v[i] ?? 0) / M
  }
  const epistemic = new Float64Array(n)
  for (const v of vecs) {
    for (let i = 0; i < n; i++) {
      const d = (v[i] ?? 0) - (meanData[i] ?? 0)
      epistemic[i] = (epistemic[i] ?? 0) + (d * d) / M
    }
  }
  const aleatoric = new Float64Array(n)
  // The Gaussian entropies need the members' own variances; without them the expected entropy and the information are
  // undefined (NaN), not zero.
  const expectedEnt = new Float64Array(n).fill(vars === undefined ? Number.NaN : 0)
  const halfLog2PiE = 0.5 * Math.log(2 * Math.PI * Math.E)
  if (vars !== undefined) {
    for (const v of vars) {
      for (let i = 0; i < n; i++) {
        const s2 = v[i] ?? 0
        aleatoric[i] = (aleatoric[i] ?? 0) + s2 / M
        expectedEnt[i] = (expectedEnt[i] ?? 0) + (halfLog2PiE + 0.5 * Math.log(Math.max(s2, 1e-300))) / M
      }
    }
  }
  const total = Float64Array.from({ length: n }, (_, i) => (epistemic[i] ?? 0) + (aleatoric[i] ?? 0))
  // Entropy of the Gaussian moment-matched to the mixture; the information is its excess over the members' mean entropy.
  const predEnt = Float64Array.from(
    { length: n },
    (_, i) => halfLog2PiE + 0.5 * Math.log(Math.max(total[i] ?? 0, 1e-300)),
  )
  const mi = Float64Array.from({ length: n }, (_, i) => Math.max(0, (predEnt[i] ?? 0) - (expectedEnt[i] ?? 0)))
  return {
    mean: fromStorage(meanData, [n], 'float64'),
    epistemicVariance: fromStorage(epistemic, [n], 'float64'),
    aleatoricVariance: fromStorage(aleatoric, [n], 'float64'),
    totalVariance: fromStorage(total, [n], 'float64'),
    predictiveEntropy: fromStorage(predEnt, [n], 'float64'),
    expectedEntropy: fromStorage(expectedEnt, [n], 'float64'),
    mutualInformation: fromStorage(mi, [n], 'float64'),
    disagreement: fromStorage(Float64Array.from(epistemic), [n], 'float64'),
  }
}

/**
 * The uncertainty decomposition of fitted estimators on new inputs: each member's `predictProba` (classifiers) or
 * `predict` (regressors) feeds {@link deepEnsemblePredict}.
 *
 * @param members - Fitted members, all classifiers or all regressors.
 * @param x - Input features `[n, d]`.
 * @param options - Options forwarded to {@link deepEnsemblePredict}.
 * @returns The decomposed {@link UncertaintySummary}.
 * @concept uncertainty-quantification
 * @example
 * ```ts
 * import { ensembleUncertainty } from 'mdml-engine/learn'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * const member = (slope: number) => ({ estimator: 'line', kind: 'regressor', params: {}, predict: (x) => tensor(x.map(([v]) => slope * v)) })
 * toFlat(ensembleUncertainty([member(1), member(3)], [[1]]).epistemicVariance) // => [1]
 * ```
 */
export function ensembleUncertainty(
  members: readonly Fitted[],
  x: TensorLike,
  options: DeepEnsemblePredictOptions = {},
): UncertaintySummary {
  if (members.length === 0) throw new MdmlError('value', 'ensembleUncertainty: need at least one member')
  const classifier = members[0]?.kind === 'classifier'
  const predictions = members.map((f) =>
    (classifier
      ? requireVerb(f, 'predictProba', 'ensembleUncertainty')
      : requireVerb(f, 'predict', 'ensembleUncertainty'))(x),
  )
  return deepEnsemblePredict(predictions, { kind: classifier ? 'classification' : 'regression', ...options })
}

/** Options of {@link deepEnsembleAlgorithm} and {@link deepEnsemble}. */
export interface DeepEnsembleOptions extends FitOptions {
  /** The number of members (default 5). */
  readonly members?: number
  /** Fit each member on a bootstrap resample of the rows instead of all of them (default false: only the key differs). */
  readonly bootstrap?: boolean
  /** Inputs to score the ensemble's disagreement on after each member, for the `ensemble/disagreement` path. */
  readonly evaluation?: TensorLike
}

/** Serialisable state of {@link deepEnsembleAlgorithm}: the members fitted so far as JSON parameters. */
export interface DeepEnsembleState {
  readonly members: readonly SerialisedFitted[]
  /** The mean disagreement of the members on the evaluation inputs (0 before there are two). */
  readonly disagreement: number
}

/**
 * Deep ensembles (Lakshminarayanan, Pritzel and Blundell 2017) as a stepped algorithm, one member per step: the same
 * estimator fitted again under its own key (`foldIn(key, m)`), so a stochastic initialisation or training order gives each
 * member its own solution, optionally on a bootstrap resample. The members' spread is the epistemic uncertainty. Logs
 * `member/index`, and `ensemble/disagreement` (the mean over the evaluation inputs of the members' spread) when inputs are
 * given.
 *
 * @param estimator - Any estimator of the library (a network, a tree, a linear model).
 * @param data - The training data.
 * @param options - `members`, `bootstrap`, `evaluation` and fit options.
 * @returns The algorithm.
 * @concept uncertainty-quantification
 * @example
 * ```ts
 * import { deepEnsembleAlgorithm } from 'mdml-engine/learn'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const mean = defineEstimator(
 *   { name: 'mean', kind: 'regressor', fit: (d) => ({ m: d.y.reduce((a, b) => a + b, 0) / d.y.length }), methods: (p) => ({ predict: (x) => x.map(() => p.m) }) },
 *   {},
 * )
 * runAlgorithm(deepEnsembleAlgorithm(mean, { x: [[0], [1]], y: [1, 3] }, { members: 3 }), { maxSteps: 3 }).state.members.length // => 3
 * ```
 */
export function deepEnsembleAlgorithm(
  estimator: Estimator,
  data: LearnData,
  options: DeepEnsembleOptions = {},
): Algorithm<DeepEnsembleState> {
  const who = 'deepEnsemble'
  const count = integerOption(options.members ?? 5, 'members', who, 1)
  const baseKey = keyFor(options, who)
  const rows = matrixView(data.x, who).n
  const evaluation = options.evaluation
  return {
    name: 'deep-ensemble',
    config: { estimator: estimator.name, members: count, bootstrap: options.bootstrap ?? false },
    schema: channels({
      metrics: {
        'member/index': { description: 'the members fitted so far' },
        'ensemble/disagreement': { description: 'mean spread of the members over the evaluation inputs' },
      },
    }),
    init: () => ({ members: [], disagreement: 0 }),
    step: (s) => {
      const memberKey = foldIn(baseKey, s.members.length)
      const sample =
        options.bootstrap === true ? takeData(data, toFlat(bootstrapIndices(foldIn(memberKey, 1), rows))) : data
      const fitted = estimator.fit(sample, innerFitOptions(options, memberKey, options.parent))
      const members = [...s.members, serialiseFitted(fitted)]
      let spread = 0
      if (evaluation !== undefined && members.length > 1) {
        const revived = members.map((m) => revive(estimator, m))
        spread = mean1(toFlat(ensembleUncertainty(revived, evaluation).disagreement))
      }
      return { members, disagreement: spread }
    },
    done: (s) => (s.members.length >= count ? 'completed' : undefined),
    observe: (s) => ({ metrics: { 'member/index': s.members.length, 'ensemble/disagreement': s.disagreement } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as DeepEnsembleState,
  }
}

const mean1 = (xs: readonly number[]): number => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length)

/** A fitted deep ensemble: the members and the mixture predictive. */
export interface DeepEnsemble {
  readonly members: readonly Fitted[]
  /** The uncertainty decomposition of the members' predictions at new inputs. */
  readonly predict: (x: TensorLike, options?: DeepEnsemblePredictOptions) => UncertaintySummary
}

/**
 * Fit a deep ensemble (see {@link deepEnsembleAlgorithm}) and return the members with their mixture predictive.
 *
 * @param estimator - Any estimator of the library.
 * @param data - The training data.
 * @param options - `members`, `bootstrap`, `evaluation` and fit options.
 * @returns The members and `predict`, which decomposes the uncertainty at new inputs.
 * @concept uncertainty-quantification
 * @example
 * ```ts
 * import { deepEnsemble } from 'mdml-engine/learn'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const mean = defineEstimator(
 *   { name: 'mean', kind: 'regressor', fit: (d) => ({ m: d.y.reduce((a, b) => a + b, 0) / d.y.length }), methods: (p) => ({ predict: (x) => x.map(() => p.m) }) },
 *   {},
 * )
 * toFlat(deepEnsemble(mean, { x: [[0], [1]], y: [1, 3] }, { members: 2 }).predict([[5]]).mean) // => [2]
 * ```
 */
export function deepEnsemble(estimator: Estimator, data: LearnData, options: DeepEnsembleOptions = {}): DeepEnsemble {
  const alg = deepEnsembleAlgorithm(estimator, data, options)
  const { state } = runAlgorithm(alg, {
    ...runOptionsFor(options, 'deepEnsemble'),
    maxSteps: integerOption(options.members ?? 5, 'members', 'deepEnsemble', 1),
  })
  const members = state.members.map((m) => revive(estimator, m))
  return { members, predict: (x, o = {}) => ensembleUncertainty(members, x, o) }
}

/**
 * Disagreement across ensemble members per sample: Jensen-Shannon divergence (BALD mutual information) for
 * classification probabilities `[n, k]`, or epistemic variance across members for regression predictions `[n]`.
 *
 * @param members - Member predictions or probability matrices.
 * @returns One disagreement score per row.
 * @concept uncertainty-quantification
 * @example
 * ```ts
 * import { disagreement } from 'mdml-engine/learn'
 * disagreement([[1, 5], [3, 5]]) // => [1, 0]
 * ```
 */
export function disagreement(members: readonly TensorLike[]): number[] {
  return toFlat(deepEnsemblePredict(members).disagreement)
}

/** Output of a single stochastic forward pass in {@link mcDropoutPredict}: either a prediction tensor or mean + variance. */
export type StochasticPassOutput = TensorLike | { readonly mean: TensorLike; readonly variance?: TensorLike }

/** Options of {@link mcDropoutPredict} and {@link mcDropoutAlgorithm}. */
export interface McDropoutOptions extends FitOptions {
  /** Number of stochastic forward passes T (default 20). */
  readonly passes?: number
  /** Task kind: `'auto'`, `'regression'` or `'classification'`. */
  readonly kind?: 'auto' | 'regression' | 'classification'
}

/** Serialisable state of {@link mcDropoutAlgorithm}. */
export interface McDropoutState {
  readonly t: number
  /** The predictions of the passes so far. */
  readonly samples: readonly Tensor<'float64'>[]
  /** The variances the passes reported, when they did. */
  readonly variances: readonly Tensor<'float64'>[]
  readonly hasVariances: boolean
  /** The running sums over passes of the predictions, their squares and the reported variances (flat). */
  readonly sum: Float64Array
  readonly sumSq: Float64Array
  readonly varSum: Float64Array
  /** The largest change of an element of the predictive mean at the last pass. */
  readonly deltaMean: number
}

function unpackPass(out: StochasticPassOutput): {
  readonly mean: Tensor<'float64'>
  readonly variance?: Tensor<'float64'>
} {
  if (typeof out === 'object' && out !== null && 'mean' in out && !('shape' in out) && !Array.isArray(out)) {
    const meanT = asTensor(out.mean)
    const mean64 = fromStorage(Float64Array.from(toFlat(meanT)), [...meanT.shape], 'float64')
    if (out.variance === undefined) return { mean: mean64 }
    const varT = asTensor(out.variance)
    return { mean: mean64, variance: fromStorage(Float64Array.from(toFlat(varT)), [...varT.shape], 'float64') }
  }
  const t = asTensor(out as TensorLike)
  return { mean: fromStorage(Float64Array.from(toFlat(t)), [...t.shape], 'float64') }
}

/**
 * Stepped Monte Carlo dropout {@link Algorithm} (Gal and Ghahramani 2016): each step `t` executes one stochastic
 * forward pass with derived key `foldIn(baseKey, t)` and updates the running predictive mean and variance
 * decomposition.
 *
 * @param forward - Stochastic forward pass taking a PRNG `Key`.
 * @param options - `passes` (default 20), `kind` and `key`.
 * @returns Stepped {@link Algorithm} over {@link McDropoutState}.
 * @concept uncertainty-quantification
 * @example
 * ```ts
 * import { mcDropoutAlgorithm } from 'mdml-engine/learn'
 * import { key, normal } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = mcDropoutAlgorithm((k) => normal(k, [2]), { passes: 4, key: key(0) })
 * runAlgorithm(alg, { maxSteps: 4, key: key(0) }).state.samples.length // => 4
 * ```
 */
export function mcDropoutAlgorithm(
  forward: (key: Key) => StochasticPassOutput,
  options: McDropoutOptions = {},
): Algorithm<McDropoutState> {
  const passes = integerOption(options.passes ?? 20, 'passes', 'mcDropoutPredict', 1)
  const baseKey = keyFor(options, 'mcDropoutPredict')
  return {
    name: 'mc-dropout',
    config: { passes, kind: options.kind ?? 'auto' },
    schema: channels({
      metrics: {
        'pass/index': { description: 'stochastic forward pass index' },
        'predictive/convergence': { description: 'max absolute change in predictive mean', direction: 'lower' },
      },
      tensors: {
        'pass/prediction': { role: 'state', description: 'prediction of the latest stochastic pass' },
        'predictive/mean': { role: 'state', description: 'running predictive mean' },
        'predictive/variance': { role: 'state', description: 'running total predictive variance' },
      },
    }),
    init: () => ({
      t: 0,
      samples: [],
      variances: [],
      hasVariances: false,
      sum: new Float64Array(0),
      sumSq: new Float64Array(0),
      varSum: new Float64Array(0),
      deltaMean: 0,
    }),
    step: (s) => {
      const passKey = foldIn(baseKey, s.t)
      const { mean, variance } = unpackPass(forward(passKey))
      const flat = Float64Array.from(toFlat(mean))
      const n = flat.length
      if (s.t > 0 && n !== s.sum.length) throw new MdmlError('shape', 'mcDropout: the passes differ in shape')
      const sum = new Float64Array(n)
      const sumSq = new Float64Array(n)
      const varSum = new Float64Array(n)
      const reported = variance === undefined ? undefined : toFlat(variance)
      let delta = 0
      for (let i = 0; i < n; i++) {
        const v = flat[i] ?? 0
        sum[i] = (s.sum[i] ?? 0) + v
        sumSq[i] = (s.sumSq[i] ?? 0) + v * v
        varSum[i] = (s.varSum[i] ?? 0) + (reported?.[i] ?? 0)
        if (s.t > 0) delta = Math.max(delta, Math.abs((sum[i] ?? 0) / (s.t + 1) - (s.sum[i] ?? 0) / s.t))
      }
      return {
        t: s.t + 1,
        samples: [...s.samples, mean],
        variances: variance === undefined ? s.variances : [...s.variances, variance],
        hasVariances: s.hasVariances || variance !== undefined,
        sum,
        sumSq,
        varSum,
        deltaMean: delta,
      }
    },
    done: (s) => (s.t >= passes ? 'completed' : undefined),
    observe: (s) => {
      if (s.t === 0) return { metrics: { 'pass/index': 0, 'predictive/convergence': 0 } }
      const latest = s.samples[s.samples.length - 1] as Tensor<'float64'>
      const n = s.sum.length
      const mean = Float64Array.from(s.sum, (v) => v / s.t)
      // Element-wise total variance: the spread of the passes plus the mean of the variances they reported.
      const variance = Float64Array.from(
        { length: n },
        (_, i) => Math.max(0, (s.sumSq[i] ?? 0) / s.t - (mean[i] ?? 0) ** 2) + (s.varSum[i] ?? 0) / s.t,
      )
      return {
        metrics: { 'pass/index': s.t, 'predictive/convergence': s.deltaMean },
        tensors: {
          'pass/prediction': { role: 'state', value: latest, shape: latest.shape },
          'predictive/mean': { role: 'state', value: mean, shape: latest.shape },
          'predictive/variance': { role: 'state', value: variance, shape: latest.shape },
        },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as McDropoutState,
  }
}

/**
 * Monte Carlo dropout prediction (Gal and Ghahramani 2016): run a stochastic forward pass `T` times with derived keys
 * `foldIn(key, t)` for `t = 0, ..., T - 1` and decompose total predictive uncertainty into epistemic and aleatoric
 * components.
 *
 * @param forward - Stochastic forward pass taking a PRNG `Key` and returning predictions (or `{ mean, variance }`).
 * @param options - Number of `passes` (default 20), task `kind`, `key` and tracing options.
 * @returns The {@link UncertaintySummary} together with the `samples` from each pass.
 * @concept uncertainty-quantification
 * @example
 * ```ts
 * import { mcDropoutPredict } from 'mdml-engine/learn'
 * import { key, normal } from 'mdml-engine/random'
 * import { toFlat } from 'mdml-engine/tensor'
 * const res = mcDropoutPredict((k) => normal(k, [2]), { passes: 10, key: key(42) })
 * res.samples.length // => 10
 * toFlat(res.epistemicVariance).every((v) => v > 0) // => true
 * ```
 */
export function mcDropoutPredict(
  forward: (key: Key) => StochasticPassOutput,
  options: McDropoutOptions = {},
): UncertaintySummary & { readonly samples: readonly Tensor<'float64'>[] } {
  const passes = integerOption(options.passes ?? 20, 'passes', 'mcDropoutPredict', 1)
  const alg = mcDropoutAlgorithm(forward, options)
  const { state } = runAlgorithm(alg, { ...runOptionsFor(options, 'mcDropoutPredict'), maxSteps: passes })
  const summary = deepEnsemblePredict(state.samples, {
    ...(options.kind === undefined ? {} : { kind: options.kind }),
    ...(state.hasVariances ? { memberVariances: state.variances } : {}),
  })
  return { ...summary, samples: state.samples }
}

/**
 * Alias of {@link mcDropoutPredict} for the methods layer (`learn/uncertainty.ts`: `monteCarloDropout`).
 *
 * @param forward - Stochastic forward pass taking a PRNG `Key`.
 * @param options - Number of `passes` (default 20), task `kind`, `key` and tracing options.
 * @returns The {@link UncertaintySummary} and per-pass `samples`.
 * @concept uncertainty-quantification
 * @example
 * ```ts
 * import { monteCarloDropout } from 'mdml-engine/learn'
 * import { key } from 'mdml-engine/random'
 * monteCarloDropout(() => [1, 2], { passes: 3, key: key(0) }).samples.length // => 3
 * ```
 */
export function monteCarloDropout(
  forward: (key: Key) => StochasticPassOutput,
  options: McDropoutOptions = {},
): UncertaintySummary & { readonly samples: readonly Tensor<'float64'>[] } {
  return mcDropoutPredict(forward, options)
}
