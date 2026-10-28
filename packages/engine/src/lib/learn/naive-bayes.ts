/**
 * Naive Bayes classifiers as one family over sufficient statistics: Gaussian, multinomial, Bernoulli, categorical and
 * complement naive Bayes, each updated one weighted row at a time (`naiveBayesPartialFit`, the statistics being the
 * fitted parameters), so that fitting is the algorithm that absorbs the rows in order and logs the prequential
 * accuracy; and the two ways of relaxing the independence of features given the class, tree-augmented naive Bayes
 * (Chow-Liu's tree over conditional mutual information, grown one edge per step) and the averaged one-dependence
 * estimators (AODE). A session shows the tables updating and the accuracy climbing; `fittedAt(t)` is the classifier
 * on the first `t` rows.
 */

import { logSumExpOf } from '../base/index.ts'
import { choice, dict, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { classifierFromScores, type ScoreClassifierVerbs } from './_classify.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import type { LearnData } from './estimator.ts'
import { encodeLabels, type MatrixView, matrixView, rowsOf, sampleWeightsOf, targetVector } from './validation.ts'

/** The five likelihood families. */
export type NaiveBayesFamily = 'gaussian' | 'multinomial' | 'bernoulli' | 'categorical' | 'complement'

/** The hyperparameters of {@link naiveBayes}. */
export interface NaiveBayesHyper {
  readonly family: NaiveBayesFamily
  /** Additive (Laplace/Lidstone) smoothing of the count families. */
  readonly alpha: number
  /** Gaussian: the portion of the largest feature variance added to every variance. */
  readonly varSmoothing: number
  /** Bernoulli: the threshold above which a feature is 1 (default 0). */
  readonly binarize: number
  /** Complement: normalise the weights by their sum, as the paper's second variant. */
  readonly norm: boolean
  /** Learn the class priors from the counts (else uniform, or `classPrior`). */
  readonly fitPrior: boolean
  /** The class priors, one per class, overriding the counts. */
  readonly classPrior?: readonly number[]
  /** Categorical: the fewest categories of every feature (default what the data show). */
  readonly minCategories?: number
}

/**
 * The fitted parameters of {@link naiveBayes}: the sufficient statistics, plain JSON. Which fields are used depends on
 * the family; the others are empty.
 */
export type NaiveBayesParams = {
  readonly family: NaiveBayesFamily
  readonly classes: number[]
  readonly nFeatures: number
  /** The (weighted) number of rows of each class. */
  readonly classCount: number[]
  /** Gaussian: the class means `[k][d]`. */
  readonly mean: number[][]
  /** Gaussian: the class sums of squared deviations `[k][d]`. */
  readonly m2: number[][]
  /** Gaussian: the number of rows, mean and sums of squared deviations of all rows, unweighted (for the variance smoothing). */
  readonly overall: { readonly weight: number; readonly mean: number[]; readonly m2: number[] }
  /** Multinomial, Bernoulli (binarised) and complement: the class feature sums `[k][d]`. */
  readonly featureCount: number[][]
  /** Categorical: per feature, per class, the counts of each category `[d][k][categories]`. */
  readonly categoryCount: number[][][]
}

/** What a fitted naive Bayes offers. */
export interface NaiveBayesVerbs extends ScoreClassifierVerbs {
  /** The log-likelihood of each feature value under each class, `[m][k][d]` (what the joint sums, with the log prior). */
  readonly featureLogLikelihood: (x: TensorLike) => number[][][]
  /** The log prior of each class. */
  readonly classLogPrior: readonly number[]
  /** The parameters after one more weighted row (the online update); revive them with the estimator. */
  readonly partialFit: (x: TensorLike, y: TensorLike, weights?: TensorLike) => NaiveBayesParams
}

const MIN_ALPHA = 1e-10

const zeros = (k: number, d: number): number[][] => Array.from({ length: k }, () => new Array<number>(d).fill(0))

/**
 * The empty statistics of a naive Bayes over known classes and a feature count.
 *
 * @param classes - The class labels, ascending.
 * @param nFeatures - The number of features.
 * @param hyper - The family (and `minCategories` for the categorical).
 * @returns Statistics with no rows.
 * @example
 * ```ts
 * import { naiveBayesInit } from 'mdml-engine/learn'
 * naiveBayesInit([0, 1], 2, { family: 'multinomial' }).classCount // => [0, 0]
 * ```
 */
export function naiveBayesInit(
  classes: readonly number[],
  nFeatures: number,
  hyper: Pick<NaiveBayesHyper, 'family' | 'minCategories'>,
): NaiveBayesParams {
  const k = classes.length
  const gaussian = hyper.family === 'gaussian'
  const counting = hyper.family === 'multinomial' || hyper.family === 'bernoulli' || hyper.family === 'complement'
  return {
    family: hyper.family,
    classes: [...classes],
    nFeatures,
    classCount: new Array<number>(k).fill(0),
    mean: gaussian ? zeros(k, nFeatures) : [],
    m2: gaussian ? zeros(k, nFeatures) : [],
    overall: {
      weight: 0,
      mean: gaussian ? new Array<number>(nFeatures).fill(0) : [],
      m2: gaussian ? new Array<number>(nFeatures).fill(0) : [],
    },
    featureCount: counting ? zeros(k, nFeatures) : [],
    categoryCount:
      hyper.family === 'categorical' ? Array.from({ length: nFeatures }, () => zeros(k, hyper.minCategories ?? 0)) : [],
  }
}

function welford(
  weight: number,
  mean: readonly number[],
  m2: readonly number[],
  x: ArrayLike<number>,
  w: number,
): { readonly weight: number; readonly mean: number[]; readonly m2: number[] } {
  const total = weight + w
  const nextMean = mean.map((m, j) => m + (w / total) * ((x[j] ?? 0) - m))
  const nextM2 = m2.map((s, j) => s + w * ((x[j] ?? 0) - (mean[j] ?? 0)) * ((x[j] ?? 0) - (nextMean[j] ?? 0)))
  return { weight: total, mean: nextMean, m2: nextM2 }
}

/**
 * One more weighted row absorbed (`partial_fit` of a single sample): a new statistics value, the old one untouched.
 * Categorical features grow their tables when a larger category appears.
 *
 * @param p - The statistics so far.
 * @param hyper - `family` and `binarize`.
 * @param x - The row's features.
 * @param label - The row's class label (one of `p.classes`).
 * @param weight - The row's weight (default 1).
 * @returns The updated statistics.
 * @example
 * ```ts
 * import { naiveBayesInit, naiveBayesPartialFit } from 'mdml-engine/learn'
 * const h = { family: 'multinomial' } as const
 * const p = naiveBayesPartialFit(naiveBayesInit([0, 1], 2, h), h, [3, 1], 1)
 * p.featureCount // => [[0, 0], [3, 1]]
 * ```
 */
export function naiveBayesPartialFit(
  p: NaiveBayesParams,
  hyper: Pick<NaiveBayesHyper, 'family' | 'binarize'>,
  x: ArrayLike<number>,
  label: number,
  weight = 1,
): NaiveBayesParams {
  const c = p.classes.indexOf(label)
  if (c < 0) throw new Error(`naiveBayesPartialFit: label ${label} is not among the classes ${p.classes.join(', ')}`)
  const classCount = p.classCount.map((n, i) => (i === c ? n + weight : n))
  switch (p.family) {
    case 'gaussian': {
      const row = welford(p.classCount[c] ?? 0, p.mean[c] ?? [], p.m2[c] ?? [], x, weight)
      return {
        ...p,
        classCount,
        mean: p.mean.map((m, i) => (i === c ? row.mean : m)),
        m2: p.m2.map((m, i) => (i === c ? row.m2 : m)),
        // The smoothing's reference variance ignores the sample weights, as scikit-learn's.
        overall: welford(p.overall.weight, p.overall.mean, p.overall.m2, x, 1),
      }
    }
    case 'categorical': {
      const categoryCount = p.categoryCount.map((table, j) => {
        const v = Math.round(x[j] ?? 0)
        const width = Math.max(table[0]?.length ?? 0, v + 1)
        return table.map((row, i) => {
          const grown = row.length < width ? [...row, ...new Array<number>(width - row.length).fill(0)] : row
          return i === c ? grown.map((n, cat) => (cat === v ? n + weight : n)) : grown
        })
      })
      return { ...p, classCount, categoryCount }
    }
    default: {
      const bernoulli = p.family === 'bernoulli'
      const featureCount = p.featureCount.map((row, i) =>
        i === c
          ? row.map((n, j) => n + weight * (bernoulli ? ((x[j] ?? 0) > hyper.binarize ? 1 : 0) : (x[j] ?? 0)))
          : row,
      )
      return { ...p, classCount, featureCount }
    }
  }
}

const logPriorOf = (p: NaiveBayesParams, h: NaiveBayesHyper): number[] => {
  const k = p.classes.length
  if (h.classPrior !== undefined) return h.classPrior.map(Math.log)
  const total = p.classCount.reduce((a, b) => a + b, 0)
  if (!h.fitPrior || total === 0) return new Array<number>(k).fill(-Math.log(k))
  return p.classCount.map((n) => Math.log(n) - Math.log(total))
}

/** Per-feature log-likelihood tables for rows (the part of the joint that depends on the features). */
type FeatureScores = (m: MatrixView, i: number, c: number, j: number) => number

function scorer(p: NaiveBayesParams, h: NaiveBayesHyper): FeatureScores {
  const d = p.nFeatures
  const alpha = Math.max(h.alpha, MIN_ALPHA)
  switch (p.family) {
    case 'gaussian': {
      const total = p.overall.weight
      const epsilon =
        total > 0 ? h.varSmoothing * p.overall.m2.reduce((a, s) => Math.max(a, s / total), 0) : h.varSmoothing
      const variance = p.m2.map((row, c) =>
        row.map((s) => s / Math.max(p.classCount[c] ?? 0, Number.MIN_VALUE) + epsilon),
      )
      return (m, i, c, j) => {
        const v = variance[c]?.[j] ?? 1
        const diff = (m.data[i * d + j] ?? 0) - (p.mean[c]?.[j] ?? 0)
        return -0.5 * Math.log(2 * Math.PI * v) - (0.5 * diff * diff) / v
      }
    }
    case 'multinomial': {
      const log = p.featureCount.map((row) => {
        const smoothed = row.map((n) => n + alpha)
        const sum = smoothed.reduce((a, b) => a + b, 0)
        return smoothed.map((n) => Math.log(n) - Math.log(sum))
      })
      return (m, i, c, j) => (m.data[i * d + j] ?? 0) * (log[c]?.[j] ?? 0)
    }
    case 'complement': {
      const all = Array.from({ length: d }, (_, j) => p.featureCount.reduce((s, row) => s + (row[j] ?? 0), 0))
      const weights = p.featureCount.map((row) => {
        const comp = row.map((n, j) => (all[j] ?? 0) + alpha - n)
        const sum = comp.reduce((a, b) => a + b, 0)
        const logged = comp.map((v) => Math.log(v / sum))
        if (!h.norm) return logged.map((v) => -v)
        const summed = logged.reduce((a, b) => a + b, 0)
        return logged.map((v) => v / summed)
      })
      return (m, i, c, j) => (m.data[i * d + j] ?? 0) * (weights[c]?.[j] ?? 0)
    }
    case 'bernoulli': {
      const log = p.featureCount.map((row, c) =>
        row.map((n) => Math.log(n + alpha) - Math.log((p.classCount[c] ?? 0) + 2 * alpha)),
      )
      const negative = log.map((row) => row.map((v) => Math.log1p(-Math.exp(v))))
      return (m, i, c, j) => ((m.data[i * d + j] ?? 0) > h.binarize ? (log[c]?.[j] ?? 0) : (negative[c]?.[j] ?? 0))
    }
    default: {
      const log = p.categoryCount.map((table) => {
        const width = Math.max(table[0]?.length ?? 0, h.minCategories ?? 0)
        return table.map((row, c) =>
          Array.from(
            { length: width },
            (_, v) => Math.log((row[v] ?? 0) + alpha) - Math.log((p.classCount[c] ?? 0) + alpha * width),
          ),
        )
      })
      return (m, i, c, j) => {
        const v = Math.round(m.data[i * d + j] ?? 0)
        const table = log[j]?.[c]
        // A category never seen counts as having no observations.
        return table?.[v] ?? Math.log(alpha) - Math.log((p.classCount[c] ?? 0) + alpha * (table?.length ?? 1))
      }
    }
  }
}

function naiveBayesVerbs(p: NaiveBayesParams, h: NaiveBayesHyper): NaiveBayesVerbs {
  const k = p.classes.length
  const d = p.nFeatures
  const logPrior = logPriorOf(p, h)
  const score = scorer(p, h)
  const useScores = (m: MatrixView): Float64Array => {
    const out = new Float64Array(m.n * k)
    // Complement naive Bayes ignores the prior except for a single class, as scikit-learn.
    const prior = p.family === 'complement' && k > 1 ? () => 0 : (c: number) => logPrior[c] ?? 0
    for (let i = 0; i < m.n; i++)
      for (let c = 0; c < k; c++) {
        let s = prior(c)
        for (let j = 0; j < d; j++) s += score(m, i, c, j)
        out[i * k + c] = (p.classCount[c] ?? 0) > 0 ? s : Number.NEGATIVE_INFINITY
      }
    return out
  }
  const base = classifierFromScores(p.classes, d, useScores, 'naiveBayes')
  return {
    ...base,
    classLogPrior: logPrior,
    featureLogLikelihood: (x) => {
      const m = matrixView(x, 'naiveBayes', { nFeatures: d })
      return Array.from({ length: m.n }, (_, i) =>
        Array.from({ length: k }, (_, c) => Array.from({ length: d }, (_, j) => score(m, i, c, j))),
      )
    },
    partialFit: (x, y, weights) => {
      const m = matrixView(x, 'naiveBayes.partialFit', { nFeatures: d })
      const labels = targetVector(y, 'naiveBayes.partialFit', m.n)
      const w = sampleWeightsOf(weights, m.n, 'naiveBayes.partialFit')
      let next = p
      for (let i = 0; i < m.n; i++)
        next = naiveBayesPartialFit(next, h, m.data.subarray(i * d, (i + 1) * d), labels[i] ?? 0, w[i] ?? 1)
      return next
    },
  }
}

/** The state of the naive Bayes algorithm: the statistics and the running prequential score. */
export type NaiveBayesState = {
  readonly stats: NaiveBayesParams
  /** Rows absorbed. */
  readonly rows: number
  /** Rows whose class was predicted correctly before they were absorbed. */
  readonly correct: number
  /** The summed log loss of the class before the row was absorbed. */
  readonly logLoss: number
}

/**
 * The naive Bayes fit as an algorithm: one row absorbed per step (the first at the start, so a model exists), each
 * after being predicted by the model so far. It logs the metrics `rows`, `accuracy` (the prequential accuracy, the
 * fraction of rows predicted right before they were seen) and `logLoss` (their mean log loss).
 *
 * @param data - Features, class labels and optional weights.
 * @param hyper - The family and its settings.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { naiveBayesAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = naiveBayesAlgorithm({ x: [[0], [1], [0], [1]], y: [0, 1, 0, 1] }, { family: 'bernoulli', alpha: 1, varSmoothing: 1e-9, binarize: 0, norm: false, fitPrior: true })
 * runAlgorithm(alg, { maxSteps: 10 }).state.correct // => 2
 * ```
 */
export function naiveBayesAlgorithm(data: LearnData, hyper: NaiveBayesHyper): Algorithm<NaiveBayesState> {
  const m = matrixView(data.x, 'naiveBayes')
  const { classes, codes } = encodeLabels(data.y ?? [], 'naiveBayes', m.n)
  const w = sampleWeightsOf(data.weights, m.n, 'naiveBayes')
  const row = (i: number): Float64Array => m.data.subarray(i * m.d, (i + 1) * m.d)
  const categories =
    hyper.family === 'categorical'
      ? Math.max(
          hyper.minCategories ?? 0,
          m.data.reduce((a, v) => Math.max(a, Math.round(v) + 1), 0),
        )
      : 0
  const empty = naiveBayesInit(classes, m.d, { family: hyper.family, minCategories: categories })
  const absorb = (s: NaiveBayesState, i: number): NaiveBayesState => {
    let correct = s.correct
    let logLoss = s.logLoss
    if (i > 0) {
      const verbs = naiveBayesVerbs(s.stats, hyper)
      const proba = verbs.predictProba(fromStorage(Float64Array.from(row(i)), [1, m.d], 'float64'))
      const pr = Float64Array.from((proba as Tensor).data as ArrayLike<number>)
      const truth = codes[i] ?? 0
      let best = 0
      for (let c = 1; c < pr.length; c++) if ((pr[c] ?? 0) > (pr[best] ?? 0)) best = c
      if (best === truth) correct++
      logLoss -= Math.log(Math.max(pr[truth] ?? 0, 1e-12))
    }
    return {
      stats: naiveBayesPartialFit(s.stats, hyper, row(i), classes[codes[i] ?? 0] ?? 0, w[i] ?? 1),
      rows: s.rows + 1,
      correct,
      logLoss,
    }
  }
  return {
    name: `naive-bayes-${hyper.family}`,
    config: { family: hyper.family, alpha: hyper.alpha, rows: m.n },
    schema: channels({
      metrics: {
        rows: { description: 'training rows absorbed so far', direction: 'higher' },
        accuracy: {
          description: 'prequential accuracy: rows predicted right before they were absorbed',
          direction: 'higher',
        },
        logLoss: { description: 'mean log loss of each row’s class before it was absorbed', direction: 'lower' },
      },
    }),
    init: () => absorb({ stats: empty, rows: 0, correct: 0, logLoss: 0 }, 0),
    step: (s) => absorb(s, s.rows),
    done: (s) => (s.rows >= m.n ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        rows: s.rows,
        ...(s.rows > 1 ? { accuracy: s.correct / (s.rows - 1), logLoss: s.logLoss / (s.rows - 1) } : {}),
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as NaiveBayesState,
  }
}

const NB_DEFAULTS: NaiveBayesHyper = {
  family: 'gaussian',
  alpha: 1,
  varSmoothing: 1e-9,
  binarize: 0,
  norm: false,
  fitPrior: true,
}

/**
 * Naive Bayes (`sklearn.naive_bayes`): features independent given the class, five families for the feature
 * likelihoods. `gaussian` fits a mean and variance per class and feature (with `varSmoothing` of the largest variance
 * added); `multinomial` counts, `bernoulli` binarised presence, `categorical` integer codes, each smoothed by `alpha`;
 * `complement` weights by the counts of the other classes (Rennie et al.), made for skewed text classes. The fitted
 * parameters are the sufficient statistics, so `partialFit` adds rows without refitting. A fit is the algorithm
 * that absorbs the rows one per step, logging `solver/rows`, `solver/accuracy` and `solver/logLoss`.
 *
 * @param hyper - `family`, `alpha`, `varSmoothing`, `binarize`, `norm`, `fitPrior`, `classPrior`, `minCategories`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { naiveBayes } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const nb = naiveBayes({ family: 'multinomial' }).fit({ x: [[3, 0], [2, 1], [0, 3], [1, 2]], y: [0, 0, 1, 1] })
 * toFlat(nb.predict([[4, 0], [0, 4]])) // => [0, 1]
 * ```
 */
export function naiveBayes(
  hyper: Partial<NaiveBayesHyper> = {},
): IterativeEstimator<NaiveBayesHyper, NaiveBayesState, NaiveBayesParams, NaiveBayesVerbs> {
  return iterative<NaiveBayesHyper, NaiveBayesState, NaiveBayesParams, NaiveBayesVerbs>(
    {
      name: 'naive-bayes',
      kind: 'classifier',
      defaults: NB_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          family: choice(['gaussian', 'multinomial', 'bernoulli', 'categorical', 'complement'] as const, {
            default: h.family,
          }),
          alpha: real(Math.min(1e-3, h.alpha), Math.max(10, h.alpha), { default: h.alpha, log: true }),
        }),
      algorithm: (data, h) => naiveBayesAlgorithm(data, h),
      maxSteps: (_h, data) => rowsOf(data.x),
      params: (s) => s.stats,
      verbs: naiveBayesVerbs,
      logs: { 'solver/rows': Log(), 'solver/accuracy': Log(), 'solver/logLoss': Log() },
    },
    { ...NB_DEFAULTS, ...hyper },
  )
}

/** The Gaussian family, `sklearn.naive_bayes.GaussianNB`. */
export const gaussianNaiveBayes = (
  hyper: Partial<Omit<NaiveBayesHyper, 'family'>> = {},
): ReturnType<typeof naiveBayes> => naiveBayes({ ...hyper, family: 'gaussian' })
/** The multinomial family, `sklearn.naive_bayes.MultinomialNB`. */
export const multinomialNaiveBayes = (
  hyper: Partial<Omit<NaiveBayesHyper, 'family'>> = {},
): ReturnType<typeof naiveBayes> => naiveBayes({ ...hyper, family: 'multinomial' })
/** The Bernoulli family, `sklearn.naive_bayes.BernoulliNB`. */
export const bernoulliNaiveBayes = (
  hyper: Partial<Omit<NaiveBayesHyper, 'family'>> = {},
): ReturnType<typeof naiveBayes> => naiveBayes({ ...hyper, family: 'bernoulli' })
/** The categorical family, `sklearn.naive_bayes.CategoricalNB`. */
export const categoricalNaiveBayes = (
  hyper: Partial<Omit<NaiveBayesHyper, 'family'>> = {},
): ReturnType<typeof naiveBayes> => naiveBayes({ ...hyper, family: 'categorical' })
/** The complement family, `sklearn.naive_bayes.ComplementNB`. */
export const complementNaiveBayes = (
  hyper: Partial<Omit<NaiveBayesHyper, 'family'>> = {},
): ReturnType<typeof naiveBayes> => naiveBayes({ ...hyper, family: 'complement' })

// ---------------------------------------------------------------------------------------------------------------
// Relaxing the independence: tree-augmented naive Bayes and averaged one-dependence estimators, on discrete features.
// ---------------------------------------------------------------------------------------------------------------

/**
 * The counts of discrete features and classes that tree-augmented naive Bayes and AODE read: how many rows of each
 * class, how many of each class and value of a feature, and of each class and pair of values of two features.
 */
export type DiscreteCounts = {
  readonly classes: number[]
  /** The number of values of each feature (codes `0 .. n - 1`). */
  readonly nCategories: number[]
  readonly classCount: number[]
  /** `[j][c][v]`. */
  readonly single: number[][][]
  /** `[i][j][c][a][b]` for `i < j`: rows of class `c` with feature `i` equal to `a` and feature `j` to `b`. */
  readonly pair: number[][][][][]
}

/** The empty counts for classes and the number of values of each feature. */
function emptyCounts(classes: readonly number[], nCategories: readonly number[]): DiscreteCounts {
  const k = classes.length
  const d = nCategories.length
  const grid = (a: number, b: number): number[][] => Array.from({ length: a }, () => new Array<number>(b).fill(0))
  return {
    classes: [...classes],
    nCategories: [...nCategories],
    classCount: new Array<number>(k).fill(0),
    single: nCategories.map((v) => grid(k, v)),
    pair: Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) =>
        j <= i ? [] : Array.from({ length: k }, () => grid(nCategories[i] ?? 0, nCategories[j] ?? 0)),
      ),
    ),
  }
}

const pairCount = (s: DiscreteCounts, i: number, j: number, c: number, a: number, b: number): number =>
  i < j ? (s.pair[i]?.[j]?.[c]?.[a]?.[b] ?? 0) : (s.pair[j]?.[i]?.[c]?.[b]?.[a] ?? 0)

function countRow(s: DiscreteCounts, x: ArrayLike<number>, c: number, w: number): DiscreteCounts {
  const d = s.nCategories.length
  const v = Array.from({ length: d }, (_, j) => Math.round(x[j] ?? 0))
  v.forEach((value, j) => {
    if (value < 0 || value >= (s.nCategories[j] ?? 0)) {
      throw new Error(
        `discrete counts: value ${value} of feature ${j} is outside 0..${(s.nCategories[j] ?? 0) - 1} (set minCategories)`,
      )
    }
  })
  return {
    ...s,
    classCount: s.classCount.map((n, i) => (i === c ? n + w : n)),
    single: s.single.map((table, j) =>
      table.map((row, i) => (i === c ? row.map((n, a) => (a === v[j] ? n + w : n)) : row)),
    ),
    pair: s.pair.map((byJ, i) =>
      byJ.map((table, j) =>
        j <= i
          ? table
          : table.map((grid, cls) =>
              cls === c ? grid.map((row, a) => (a === v[i] ? row.map((n, b) => (b === v[j] ? n + w : n)) : row)) : grid,
            ),
      ),
    ),
  }
}

/**
 * The counts of discrete features (integer codes) by class.
 *
 * @param x - Features `[n, d]` of integer codes.
 * @param y - Class labels.
 * @param minCategories - The fewest values of each feature (default what the data show).
 * @returns The counts.
 * @example
 * ```ts
 * import { discreteCounts } from 'mdml-engine/learn'
 * const s = discreteCounts([[0, 1], [1, 1], [1, 0]], [0, 0, 1])
 * s.classCount // => [2, 1]
 * s.single[0] // => [[1, 1], [0, 1]]
 * ```
 */
export function discreteCounts(x: TensorLike, y: TensorLike, minCategories = 0): DiscreteCounts {
  const m = matrixView(x, 'discreteCounts')
  const { classes, codes } = encodeLabels(y, 'discreteCounts', m.n)
  const nCategories = Array.from({ length: m.d }, (_, j) => {
    let top = 0
    for (let i = 0; i < m.n; i++) top = Math.max(top, Math.round(m.data[i * m.d + j] ?? 0) + 1)
    return Math.max(top, minCategories, 1)
  })
  let s = emptyCounts(classes, nCategories)
  for (let i = 0; i < m.n; i++) s = countRow(s, m.data.subarray(i * m.d, (i + 1) * m.d), codes[i] ?? 0, 1)
  return s
}

/**
 * The conditional mutual information `I(X_i; X_j | C)` of every pair of features given the class, in nats, from
 * maximum-likelihood probabilities: the weights of the tree in tree-augmented naive Bayes.
 *
 * @param counts - The counts of the training rows.
 * @returns The symmetric matrix `[d][d]` with a zero diagonal.
 * @example
 * ```ts
 * import { conditionalMutualInformation, discreteCounts } from 'mdml-engine/learn'
 * // Features 0 and 1 copy each other inside each class, feature 2 is noise.
 * const x = [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]]
 * const cmi = conditionalMutualInformation(discreteCounts(x, [0, 0, 0, 0, 1, 1, 1, 1]))
 * cmi[0][1] > cmi[0][2] // => true
 * ```
 */
export function conditionalMutualInformation(counts: DiscreteCounts): number[][] {
  const d = counts.nCategories.length
  const n = counts.classCount.reduce((a, b) => a + b, 0)
  const out = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  for (let i = 0; i < d; i++)
    for (let j = i + 1; j < d; j++) {
      let total = 0
      counts.classes.forEach((_, c) => {
        const nc = counts.classCount[c] ?? 0
        if (nc === 0) return
        for (let a = 0; a < (counts.nCategories[i] ?? 0); a++)
          for (let b = 0; b < (counts.nCategories[j] ?? 0); b++) {
            const nab = pairCount(counts, i, j, c, a, b)
            if (nab === 0) continue
            const na = counts.single[i]?.[c]?.[a] ?? 0
            const nb = counts.single[j]?.[c]?.[b] ?? 0
            total += (nab / n) * Math.log((nab * nc) / (na * nb))
          }
      })
      const rowI = out[i]
      const rowJ = out[j]
      if (rowI !== undefined) rowI[j] = total
      if (rowJ !== undefined) rowJ[i] = total
    }
  return out
}

/** The hyperparameters of the discrete Bayesian-network classifiers. */
export interface DiscreteBayesHyper {
  /** Laplace smoothing. */
  readonly alpha: number
  /** The fewest values of each feature (default what the data show). */
  readonly minCategories?: number
  /** AODE: a feature is a parent only when its value was seen at least this often. */
  readonly minCount: number
  /** TAN: the feature the tree grows from. */
  readonly root: number
}

/** An edge of the class-conditional tree: `parent -> child`, with the conditional mutual information that chose it. */
export type TreeEdge = { readonly child: number; readonly parent: number; readonly weight: number }

/** The fitted parameters of {@link treeAugmentedNaiveBayes}: plain JSON. */
export type TanParams = {
  readonly classes: number[]
  readonly nCategories: number[]
  /** The feature parent of each feature, `-1` for none. */
  readonly parent: number[]
  readonly edges: TreeEdge[]
  readonly logPrior: number[]
  /** `[j][c][u][v]`: log P(x_j = v | class c, parent value u) (`u = 0` when there is no parent). */
  readonly logCpt: number[][][][]
}

/** The state of the tree algorithm: the edges so far, in the order they were added. */
export type TanState = { readonly edges: TreeEdge[] }

function tanParams(s: DiscreteCounts, edges: readonly TreeEdge[], alpha: number): TanParams {
  const d = s.nCategories.length
  const k = s.classes.length
  const n = s.classCount.reduce((a, b) => a + b, 0)
  const parent = new Array<number>(d).fill(-1)
  for (const e of edges) parent[e.child] = e.parent
  const logCpt = Array.from({ length: d }, (_, j) => {
    const p = parent[j] ?? -1
    const nv = s.nCategories[j] ?? 1
    return Array.from({ length: k }, (_, c) => {
      const parents = p < 0 ? 1 : (s.nCategories[p] ?? 1)
      return Array.from({ length: parents }, (_, u) => {
        const given = p < 0 ? (s.classCount[c] ?? 0) : (s.single[p]?.[c]?.[u] ?? 0)
        return Array.from({ length: nv }, (_, v) => {
          const joint = p < 0 ? (s.single[j]?.[c]?.[v] ?? 0) : pairCount(s, p, j, c, u, v)
          return Math.log((joint + alpha) / (given + alpha * nv))
        })
      })
    })
  })
  return {
    classes: s.classes,
    nCategories: s.nCategories,
    parent,
    edges: [...edges],
    logPrior: s.classCount.map((nc) => Math.log((nc + alpha) / (n + alpha * k))),
    logCpt,
  }
}

function tanVerbs(p: TanParams): ScoreClassifierVerbs & { readonly edges: readonly TreeEdge[] } {
  const d = p.nCategories.length
  const k = p.classes.length
  const base = classifierFromScores(
    p.classes,
    d,
    (m) => {
      const out = new Float64Array(m.n * k)
      for (let i = 0; i < m.n; i++)
        for (let c = 0; c < k; c++) {
          let s = p.logPrior[c] ?? 0
          for (let j = 0; j < d; j++) {
            const v = Math.round(m.data[i * d + j] ?? 0)
            const pa = p.parent[j] ?? -1
            const u = pa < 0 ? 0 : Math.round(m.data[i * d + pa] ?? 0)
            // A value never seen in training says nothing: the uniform distribution over the feature's values.
            s += p.logCpt[j]?.[c]?.[u]?.[v] ?? -Math.log(p.nCategories[j] ?? 1)
          }
          out[i * k + c] = s
        }
      return out
    },
    'treeAugmentedNaiveBayes',
  )
  return { ...base, edges: p.edges }
}

const DISCRETE_DEFAULTS: DiscreteBayesHyper = { alpha: 1, minCount: 1, root: 0 }

const countsCache = new WeakMap<LearnData, Map<number, DiscreteCounts>>()

function countsOf(data: LearnData, minCategories: number): DiscreteCounts {
  let byMin = countsCache.get(data)
  if (byMin === undefined) {
    byMin = new Map()
    countsCache.set(data, byMin)
  }
  let s = byMin.get(minCategories)
  if (s === undefined) {
    s = discreteCounts(data.x, data.y ?? [], minCategories)
    byMin.set(minCategories, s)
  }
  return s
}

/**
 * The tree-augmented naive Bayes algorithm (Friedman, Geiger and Goldszmidt): Prim's algorithm on the conditional
 * mutual information between features given the class, from the root, one edge per step. Every feature has the class
 * as a parent and, once the edge reaching it is added, one other feature; a state with `t` edges is a classifier
 * (naive Bayes at `t = 0`, the full tree after `d - 1` steps). It logs the metrics `edges` and `weight` (of the edge
 * just added) and the message `edge`.
 *
 * @param data - Integer-coded features and class labels.
 * @param hyper - `root` and `minCategories`.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { tanAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]]
 * const run = runAlgorithm(tanAlgorithm({ x, y: [0, 0, 0, 0, 1, 1, 1, 1] }, {}), { maxSteps: 10 })
 * run.state.edges.map((e) => [e.parent, e.child]) // => [[0, 1], [0, 2]]
 * ```
 */
export function tanAlgorithm(
  data: LearnData,
  hyper: Partial<Pick<DiscreteBayesHyper, 'root' | 'minCategories'>>,
): Algorithm<TanState> {
  const counts = countsOf(data, hyper.minCategories ?? 0)
  const d = counts.nCategories.length
  const cmi = conditionalMutualInformation(counts)
  const root = hyper.root ?? 0
  const inTree = (edges: readonly TreeEdge[], j: number): boolean => j === root || edges.some((e) => e.child === j)
  return {
    name: 'tree-augmented-naive-bayes',
    config: { features: d, root },
    schema: channels({
      metrics: {
        edges: { description: 'edges in the class-conditional tree so far', direction: 'higher' },
        weight: { description: 'the conditional mutual information of the edge just added', direction: 'lower' },
      },
      messages: {
        edge: {
          description: 'the edge added to the tree',
          payload: {
            parent: 'the feature already in the tree',
            child: 'the feature joined to it',
            weight: 'its conditional mutual information',
          },
        },
      },
    }),
    init: () => ({ edges: [] }),
    step: (s) => {
      let best: TreeEdge | undefined
      for (let u = 0; u < d; u++) {
        if (!inTree(s.edges, u)) continue
        for (let v = 0; v < d; v++) {
          if (inTree(s.edges, v)) continue
          const w = cmi[u]?.[v] ?? 0
          if (best === undefined || w > best.weight) best = { parent: u, child: v, weight: w }
        }
      }
      return best === undefined ? s : { edges: [...s.edges, best] }
    },
    done: (s) => (s.edges.length >= d - 1 ? 'converged' : undefined),
    observe: (s) => {
      const last = s.edges[s.edges.length - 1]
      return {
        metrics: { edges: s.edges.length, ...(last === undefined ? {} : { weight: last.weight }) },
        ...(last === undefined ? {} : { messages: [{ kind: 'edge', payload: { ...last } }] }),
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as TanState,
  }
}

/**
 * Tree-augmented naive Bayes on integer-coded features: naive Bayes whose features may each depend on one other
 * feature, the dependences forming the spanning tree of greatest conditional mutual information. The fit is
 * {@link tanAlgorithm} (one edge per step), so a session shows the structure growing and the accuracy against naive
 * Bayes (the state with no edge); the probabilities are smoothed counts.
 *
 * @param hyper - `alpha`, `root`, `minCategories`.
 * @returns The estimator.
 * @reference N. Friedman, D. Geiger and M. Goldszmidt, Bayesian network classifiers, Machine Learning 29 (1997).
 * @example
 * ```ts
 * import { treeAugmentedNaiveBayes } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * // The class is the XOR of two features that naive Bayes cannot separate.
 * const x = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]]
 * const y = [0, 1, 1, 0, 0, 1, 1, 0]
 * toFlat(treeAugmentedNaiveBayes().fit({ x, y }).predict(x)) // => [0, 1, 1, 0, 0, 1, 1, 0]
 * ```
 */
export function treeAugmentedNaiveBayes(
  hyper: Partial<DiscreteBayesHyper> = {},
): IterativeEstimator<DiscreteBayesHyper, TanState, TanParams, ReturnType<typeof tanVerbs>> {
  return iterative<DiscreteBayesHyper, TanState, TanParams, ReturnType<typeof tanVerbs>>(
    {
      name: 'tree-augmented-naive-bayes',
      kind: 'classifier',
      defaults: DISCRETE_DEFAULTS,
      hyperSpace: (h) =>
        dict({ alpha: real(Math.min(1e-3, h.alpha), Math.max(10, h.alpha), { default: h.alpha, log: true }) }),
      algorithm: (data, h) => tanAlgorithm(data, h),
      maxSteps: (_h, data) => Math.max(1, matrixView(data.x, 'treeAugmentedNaiveBayes').d - 1),
      params: (s, data, h) => tanParams(countsOf(data, h.minCategories ?? 0), s.edges, h.alpha),
      verbs: tanVerbs,
      contract: () => ({}),
      logs: { 'solver/edges': Log(), 'solver/weight': Log() },
    },
    { ...DISCRETE_DEFAULTS, ...hyper },
  )
}

/** The fitted parameters of {@link aode}: the counts, plain JSON. */
export type AodeParams = DiscreteCounts

/** The state of the AODE algorithm: the counts and the running prequential score. */
export type AodeState = {
  readonly stats: DiscreteCounts
  readonly rows: number
  readonly correct: number
  readonly logLoss: number
}

/** The verbs of a fitted AODE. */
export interface AodeVerbs extends ScoreClassifierVerbs {
  /** The counts after one more row (the online update); revive them with the estimator. */
  readonly partialFit: (x: TensorLike, y: TensorLike, weights?: TensorLike) => AodeParams
}

function aodeScores(s: DiscreteCounts, h: DiscreteBayesHyper): (m: MatrixView) => Float64Array {
  const d = s.nCategories.length
  const k = s.classes.length
  const n = s.classCount.reduce((a, b) => a + b, 0)
  const alpha = h.alpha
  return (m) => {
    const out = new Float64Array(m.n * k)
    for (let r = 0; r < m.n; r++) {
      const v = Array.from({ length: d }, (_, j) => Math.round(m.data[r * d + j] ?? 0))
      const inRange = v.map((value, j) => value >= 0 && value < (s.nCategories[j] ?? 0))
      const parents = Array.from({ length: d }, (_, i) => i).filter(
        (i) =>
          inRange[i] === true && (s.single[i] ?? []).reduce((t, row) => t + (row[v[i] ?? 0] ?? 0), 0) >= h.minCount,
      )
      for (let c = 0; c < k; c++) {
        if (parents.length === 0 || d === 1) {
          // No parent qualifies (or one feature): naive Bayes with the same smoothing.
          let sc = Math.log(((s.classCount[c] ?? 0) + alpha) / (n + alpha * k))
          for (let j = 0; j < d; j++) {
            const nv = s.nCategories[j] ?? 1
            sc += Math.log(
              ((inRange[j] === true ? (s.single[j]?.[c]?.[v[j] ?? 0] ?? 0) : 0) + alpha) /
                ((s.classCount[c] ?? 0) + alpha * nv),
            )
          }
          out[r * k + c] = sc
          continue
        }
        const terms = parents.map((i) => {
          const vi = s.nCategories[i] ?? 1
          const nci = s.single[i]?.[c]?.[v[i] ?? 0] ?? 0
          let t = Math.log((nci + alpha / (k * vi)) / (n + alpha))
          for (let j = 0; j < d; j++) {
            if (j === i) continue
            const vj = s.nCategories[j] ?? 1
            const ncij = inRange[j] === true ? pairCount(s, i, j, c, v[i] ?? 0, v[j] ?? 0) : 0
            t += Math.log((ncij + alpha / vj) / (nci + alpha))
          }
          return t
        })
        out[r * k + c] = logSumExpOf(terms) - Math.log(terms.length)
      }
    }
    return out
  }
}

function aodeVerbs(p: AodeParams, h: DiscreteBayesHyper): AodeVerbs {
  const d = p.nCategories.length
  return {
    ...classifierFromScores(p.classes, d, aodeScores(p, h), 'aode'),
    partialFit: (x, y, weights) => {
      const m = matrixView(x, 'aode.partialFit', { nFeatures: d })
      const labels = targetVector(y, 'aode.partialFit', m.n)
      const w = sampleWeightsOf(weights, m.n, 'aode.partialFit')
      let next: DiscreteCounts = p
      for (let i = 0; i < m.n; i++) {
        next = countRow(
          next,
          m.data.subarray(i * d, (i + 1) * d),
          Math.max(0, p.classes.indexOf(labels[i] ?? 0)),
          w[i] ?? 1,
        )
      }
      return next
    },
  }
}

/** The AODE fit as an algorithm: one row counted per step after being predicted (prequential accuracy). */
function aodeAlgorithm(data: LearnData, h: DiscreteBayesHyper): Algorithm<AodeState> {
  const m = matrixView(data.x, 'aode')
  const { classes, codes } = encodeLabels(data.y ?? [], 'aode', m.n)
  const w = sampleWeightsOf(data.weights, m.n, 'aode')
  const empty = emptyCounts(classes, countsOf(data, h.minCategories ?? 0).nCategories)
  const row = (i: number): Float64Array => m.data.subarray(i * m.d, (i + 1) * m.d)
  const absorb = (s: AodeState, i: number): AodeState => {
    let correct = s.correct
    let logLoss = s.logLoss
    if (i > 0) {
      const scores = aodeScores(s.stats, h)({ n: 1, d: m.d, data: Float64Array.from(row(i)) })
      const z = logSumExpOf(scores)
      const truth = codes[i] ?? 0
      let best = 0
      for (let c = 1; c < scores.length; c++) if ((scores[c] ?? 0) > (scores[best] ?? 0)) best = c
      if (best === truth) correct++
      logLoss -= Math.max(-27.6, (scores[truth] ?? 0) - z)
    }
    return { stats: countRow(s.stats, row(i), codes[i] ?? 0, w[i] ?? 1), rows: s.rows + 1, correct, logLoss }
  }
  return {
    name: 'aode',
    config: { alpha: h.alpha, rows: m.n },
    schema: channels({
      metrics: {
        rows: { description: 'training rows counted so far', direction: 'higher' },
        accuracy: { description: 'prequential accuracy', direction: 'higher' },
        logLoss: { description: 'mean log loss of each row’s class before it was counted', direction: 'lower' },
      },
    }),
    init: () => absorb({ stats: empty, rows: 0, correct: 0, logLoss: 0 }, 0),
    step: (s) => absorb(s, s.rows),
    done: (s) => (s.rows >= m.n ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        rows: s.rows,
        ...(s.rows > 1 ? { accuracy: s.correct / (s.rows - 1), logLoss: s.logLoss / (s.rows - 1) } : {}),
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as AodeState,
  }
}

/**
 * Averaged one-dependence estimators (Webb, Boughton and Wang) on integer-coded features: the class posterior is
 * proportional to the average, over every feature `x_i` seen at least `minCount` times, of the one-dependence
 * estimator `P(c, x_i) prod_{j != i} P(x_j | c, x_i)` in which every feature depends on `x_i` as well as the class.
 * It needs no structure search and relaxes the independence of naive Bayes; the statistics are counts, so a row at a
 * time is the online mode (logging the prequential accuracy).
 *
 * @param hyper - `alpha`, `minCount`, `minCategories`.
 * @returns The estimator.
 * @reference G. I. Webb, J. R. Boughton and Z. Wang, Not so naive Bayes: aggregating one-dependence estimators,
 * Machine Learning 58 (2005).
 * @example
 * ```ts
 * import { aode } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]]
 * const y = [0, 1, 1, 0, 0, 1, 1, 0]
 * toFlat(aode({ alpha: 0.1 }).fit({ x, y }).predict(x)) // => [0, 1, 1, 0, 0, 1, 1, 0]
 * ```
 */
export function aode(
  hyper: Partial<DiscreteBayesHyper> = {},
): IterativeEstimator<DiscreteBayesHyper, AodeState, AodeParams, AodeVerbs> {
  return iterative<DiscreteBayesHyper, AodeState, AodeParams, AodeVerbs>(
    {
      name: 'aode',
      kind: 'classifier',
      defaults: DISCRETE_DEFAULTS,
      hyperSpace: (h) =>
        dict({ alpha: real(Math.min(1e-3, h.alpha), Math.max(10, h.alpha), { default: h.alpha, log: true }) }),
      algorithm: aodeAlgorithm,
      maxSteps: (_h, data) => rowsOf(data.x),
      params: (s) => s.stats,
      verbs: aodeVerbs,
      logs: { 'solver/rows': Log(), 'solver/accuracy': Log(), 'solver/logLoss': Log() },
    },
    { ...DISCRETE_DEFAULTS, ...hyper },
  )
}
