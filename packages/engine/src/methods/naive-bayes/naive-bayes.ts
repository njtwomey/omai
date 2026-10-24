// TODO: tree-augmented naive Bayes (Chow-Liu), AODE and k-dependence classifiers (bayes-net-classifiers.ts).
import { argmaxOf, logSumExpOf, WhiteboxError } from '../../lib/base/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { type Exposition, type LikelihoodTable, likelihoodTable } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type Fitted,
  featureNamesOf,
  type LearnData,
  type MatrixView,
  matrixView,
  sampleWeightsOf,
} from '../../lib/learn/index.ts'
import { type Structure, structure } from '../../lib/scene/index.ts'
import { bool, dict, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../lib/tensor/index.ts'

/** The naive Bayes variants. */
export type NaiveBayesKind = 'gaussian' | 'multinomial' | 'bernoulli' | 'complement' | 'categorical'

/** Hyperparameters of the naive Bayes classifiers. */
export interface NaiveBayesHyper {
  /** Additive (Laplace or Lidstone) smoothing for the discrete variants (default 1). */
  readonly alpha: number
  /** Learn class priors from the data (default true) or use uniform ones. */
  readonly fitPrior: boolean
  /** Gaussian: the share of the largest feature variance added to every variance (default 1e-9). */
  readonly varSmoothing: number
  /** Bernoulli: the threshold above which a feature counts as present (default 0). */
  readonly binarize: number
  /** Complement: normalise the weights (default false). */
  readonly norm: boolean
}

/**
 * Parameters of a fitted naive Bayes classifier: log priors and, per class, the per-feature parameters (means and
 * variances for Gaussian; log probabilities of features or categories for the discrete variants).
 */
export type NaiveBayesParams = {
  readonly kind: NaiveBayesKind
  readonly classes: number[]
  readonly logPriors: number[]
  /** Gaussian means `[k][d]`. */
  readonly theta: number[][]
  /** Gaussian variances `[k][d]`. */
  readonly sigma: number[][]
  /** Log feature probabilities `[k][d]` (multinomial, Bernoulli, complement). */
  readonly featureLogProb: number[][]
  /** Categorical: logProb[f][k][category]. */
  readonly categoryLogProb: number[][][]
}

/** Verbs of a fitted naive Bayes classifier. */
export interface NaiveBayesVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** The joint log-likelihood log P(c) + sum_f log P(x_f | c), `[m, k]`. */
  readonly predictJointLogProba: (x: TensorLike) => Tensor
  /** Each feature's log-likelihood contribution to each class for one query row. */
  readonly explain: (
    row: readonly number[],
    features?: readonly string[],
    classNames?: readonly string[],
  ) => LikelihoodTable
  /** The model as a Bayes network: the class node with an edge to every feature. */
  readonly describe: (features?: readonly string[]) => Structure
  readonly classes: number[]
  readonly nFeaturesIn: number
}

/** A fitted naive Bayes classifier. */
export type FittedNaiveBayes = Fitted<NaiveBayesParams> & NaiveBayesVerbs

/** contributions[c][f] for one row (without the prior). */
function contributions(p: NaiveBayesParams, row: ArrayLike<number>, h: NaiveBayesHyper): number[][] {
  const k = p.classes.length
  return Array.from({ length: k }, (_, c) => {
    const out: number[] = []
    const d = p.kind === 'categorical' ? p.categoryLogProb.length : (p.theta[c] ?? p.featureLogProb[c] ?? []).length
    for (let f = 0; f < d; f++) {
      const x = row[f] ?? 0
      switch (p.kind) {
        case 'gaussian': {
          const v = p.sigma[c]?.[f] ?? 1
          out.push(-0.5 * Math.log(2 * Math.PI * v) - (0.5 * (x - (p.theta[c]?.[f] ?? 0)) ** 2) / v)
          break
        }
        case 'bernoulli': {
          const lp = p.featureLogProb[c]?.[f] ?? 0
          const neg = Math.log(1 - Math.exp(lp))
          out.push(x > h.binarize ? lp : neg)
          break
        }
        case 'categorical':
          out.push(p.categoryLogProb[f]?.[c]?.[Math.round(x)] ?? Number.NEGATIVE_INFINITY)
          break
        default:
          out.push(x * (p.featureLogProb[c]?.[f] ?? 0))
      }
    }
    return out
  })
}

function jointLogLikelihood(p: NaiveBayesParams, m: MatrixView, h: NaiveBayesHyper): Float64Array {
  const k = p.classes.length
  const out = new Float64Array(m.n * k)
  const priorUsed = p.kind !== 'complement' || k === 1
  for (let i = 0; i < m.n; i++) {
    const contrib = contributions(p, m.data.subarray(i * m.d, (i + 1) * m.d), h)
    for (let c = 0; c < k; c++)
      out[i * k + c] = (priorUsed ? (p.logPriors[c] ?? 0) : 0) + (contrib[c] ?? []).reduce((a, b) => a + b, 0)
  }
  return out
}

function naiveBayesVerbs(p: NaiveBayesParams, h: NaiveBayesHyper): NaiveBayesVerbs {
  const k = p.classes.length
  const d = p.kind === 'categorical' ? p.categoryLogProb.length : (p.theta[0] ?? p.featureLogProb[0] ?? []).length
  const view = (x: TensorLike): MatrixView => matrixView(x, `${p.kind}NB`, { nFeatures: d })
  const proba = (x: TensorLike): Tensor => {
    const m = view(x)
    const j = jointLogLikelihood(p, m, h)
    for (let i = 0; i < m.n; i++) {
      const row = j.subarray(i * k, (i + 1) * k)
      const z = logSumExpOf(row)
      for (let c = 0; c < k; c++) j[i * k + c] = Math.exp((j[i * k + c] ?? 0) - z)
    }
    return fromStorage(j, [m.n, k], 'float64')
  }
  return {
    predict: (x) => {
      const m = view(x)
      const j = jointLogLikelihood(p, m, h)
      return fromStorage(
        Float64Array.from({ length: m.n }, (_, i) => p.classes[argmaxOf(j.subarray(i * k, (i + 1) * k))] ?? 0),
        [m.n],
        'float64',
      )
    },
    predictProba: proba,
    predictDistribution: (x) => categorical({ probs: proba(x) }),
    predictJointLogProba: (x) => {
      const m = view(x)
      return fromStorage(jointLogLikelihood(p, m, h), [m.n, k], 'float64')
    },
    explain: (row, features, classNames) =>
      likelihoodTable({
        classes: classNames ?? p.classes.map(String),
        features: features ?? Array.from({ length: d }, (_, f) => `x${f}`),
        logPriors: p.logPriors,
        contributions: contributions(p, row, h),
        scope: { kind: 'query' },
      }),
    describe: (features) =>
      structure(
        'bayes-net',
        [
          { id: 'class', label: 'class', role: 'latent' },
          ...Array.from({ length: d }, (_, f) => ({
            id: `x${f}`,
            label: features?.[f] ?? `x${f}`,
            role: 'observed' as const,
          })),
        ],
        Array.from({ length: d }, (_, f) => ({ from: 'class', to: `x${f}`, directed: true })),
        { title: `${p.kind} naive Bayes`, direction: 'down' },
      ),
    classes: p.classes,
    nFeaturesIn: d,
  }
}

function fitParams(kind: NaiveBayesKind, data: LearnData, h: NaiveBayesHyper, who: string): NaiveBayesParams {
  const m = matrixView(data.x, who)
  const { classes, codes } = encodeLabels(data.y ?? [], who, m.n)
  const w = sampleWeightsOf(data.weights, m.n, who)
  const k = classes.length
  const { n, d } = m
  const classWeight = new Array<number>(k).fill(0)
  for (let i = 0; i < n; i++) classWeight[codes[i] ?? 0] = (classWeight[codes[i] ?? 0] ?? 0) + (w[i] ?? 0)
  const total = classWeight.reduce((a, b) => a + b, 0)
  const logPriors = h.fitPrior ? classWeight.map((c) => Math.log(c / total)) : classes.map(() => -Math.log(k))
  const empty: NaiveBayesParams = {
    kind,
    classes: [...classes],
    logPriors,
    theta: [],
    sigma: [],
    featureLogProb: [],
    categoryLogProb: [],
  }
  const value = (i: number, f: number): number => {
    const v = m.data[i * d + f] ?? 0
    return kind === 'bernoulli' ? (v > h.binarize ? 1 : 0) : v
  }
  if (kind !== 'gaussian' && kind !== 'categorical' && kind !== 'bernoulli')
    for (const v of m.data) if (v < 0) throw new WhiteboxError('value', `${who}: features must be non-negative counts`)
  // Weighted feature totals per class.
  const counts = Array.from({ length: k }, () => new Array<number>(d).fill(0))
  for (let i = 0; i < n; i++)
    for (let f = 0; f < d; f++) {
      const row = counts[codes[i] ?? 0] as number[]
      row[f] = (row[f] ?? 0) + (w[i] ?? 0) * value(i, f)
    }
  if (kind === 'gaussian') {
    const theta = counts.map((row, c) => row.map((s) => s / (classWeight[c] ?? 1)))
    let maxVar = 0
    for (let f = 0; f < d; f++) {
      let s = 0
      let s2 = 0
      for (let i = 0; i < n; i++) {
        s += m.data[i * d + f] ?? 0
        s2 += (m.data[i * d + f] ?? 0) ** 2
      }
      maxVar = Math.max(maxVar, s2 / n - (s / n) ** 2)
    }
    const epsilon = h.varSmoothing * maxVar
    const sigma = Array.from({ length: k }, () => new Array<number>(d).fill(0))
    for (let i = 0; i < n; i++)
      for (let f = 0; f < d; f++) {
        const c = codes[i] ?? 0
        const row = sigma[c] as number[]
        row[f] =
          (row[f] ?? 0) + ((w[i] ?? 0) * ((m.data[i * d + f] ?? 0) - (theta[c]?.[f] ?? 0)) ** 2) / (classWeight[c] ?? 1)
      }
    return { ...empty, theta, sigma: sigma.map((row) => row.map((v) => v + epsilon)) }
  }
  if (kind === 'multinomial')
    return {
      ...empty,
      featureLogProb: counts.map((row) => {
        const s = row.reduce((a, b) => a + b + h.alpha, 0)
        return row.map((v) => Math.log((v + h.alpha) / s))
      }),
    }
  if (kind === 'bernoulli')
    return {
      ...empty,
      featureLogProb: counts.map((row, c) =>
        row.map((v) => Math.log((v + h.alpha) / ((classWeight[c] ?? 0) + 2 * h.alpha))),
      ),
    }
  if (kind === 'complement') {
    const all = Array.from({ length: d }, (_, f) => counts.reduce((a, row) => a + (row[f] ?? 0), 0))
    const logged = counts.map((row) => {
      const comp = row.map((v, f) => (all[f] ?? 0) + h.alpha - v)
      const s = comp.reduce((a, b) => a + b, 0)
      return comp.map((v) => Math.log(v / s))
    })
    return {
      ...empty,
      featureLogProb: logged.map((row) => {
        if (!h.norm) return row.map((v) => -v)
        const s = row.reduce((a, b) => a + b, 0)
        return row.map((v) => v / s)
      }),
    }
  }
  // Categorical: categories 0..max per feature.
  const categoryLogProb = Array.from({ length: d }, (_, f) => {
    let max = 0
    for (let i = 0; i < n; i++) {
      const v = m.data[i * d + f] ?? 0
      if (!(Number.isInteger(v) && v >= 0)) throw new WhiteboxError('value', `${who}: categories must be integers >= 0`)
      max = Math.max(max, v)
    }
    const table = Array.from({ length: k }, () => new Array<number>(max + 1).fill(0))
    for (let i = 0; i < n; i++) {
      const row = table[codes[i] ?? 0] as number[]
      const v = m.data[i * d + f] ?? 0
      row[v] = (row[v] ?? 0) + (w[i] ?? 0)
    }
    return table.map((row, c) =>
      row.map((v) => Math.log((v + h.alpha) / ((classWeight[c] ?? 0) + h.alpha * (max + 1)))),
    )
  })
  return { ...empty, categoryLogProb }
}

function naiveBayes(
  kind: NaiveBayesKind,
  hyper: Partial<NaiveBayesHyper>,
): ExplainedEstimator<NaiveBayesHyper, FittedNaiveBayes> {
  const name = `${kind}-nb`
  const est: ExplainedEstimator<NaiveBayesHyper, FittedNaiveBayes> = defineExplainedEstimator(
    {
      name,
      kind: 'classifier',
      hyperSpace: (h: NaiveBayesHyper): Space =>
        kind === 'gaussian'
          ? dict({ varSmoothing: real(1e-12, 1e-1, { log: true, default: h.varSmoothing }) })
          : dict({
              alpha: real(1e-3, 10, { log: true, default: h.alpha, label: 'smoothing' }),
              fitPrior: bool({ default: h.fitPrior }),
            }),
      fit: (data: LearnData, h: NaiveBayesHyper) => {
        const params = fitParams(kind, data, h, name)
        const fitted = est.revive(params)
        const m = matrixView(data.x, name)
        const features = featureNamesOf(data, m.d)
        const classNames = classNamesOf(data, params.classes)
        // The average contribution of each feature to each class over that class's training rows.
        const { codes } = encodeLabels(data.y ?? [], name, m.n)
        const k = params.classes.length
        const sums = Array.from({ length: k }, () => new Array<number>(m.d).fill(0))
        const counts = new Array<number>(k).fill(0)
        for (let i = 0; i < m.n; i++) {
          const c = codes[i] ?? 0
          const contrib = contributions(params, m.data.subarray(i * m.d, (i + 1) * m.d), h)[c] ?? []
          counts[c] = (counts[c] ?? 0) + 1
          for (let f = 0; f < m.d; f++) (sums[c] as number[])[f] = ((sums[c] as number[])[f] ?? 0) + (contrib[f] ?? 0)
        }
        const exposition: Exposition[] = [
          likelihoodTable({
            classes: classNames,
            features,
            logPriors: params.logPriors,
            contributions: sums.map((row, c) => row.map((v) => v / Math.max(counts[c] ?? 1, 1))),
            // Each class's row is the average over that class's own training rows.
            scope: { kind: 'average', rows: m.n },
          }),
        ]
        if (m.d === 2) exposition.push(decisionSurfaceOf(fitted, data.x, { featureNames: features, classNames }))
        return { params, exposition }
      },
      methods: (p: NaiveBayesParams, h: NaiveBayesHyper) => naiveBayesVerbs(p, h),
    },
    {
      alpha: hyper.alpha ?? 1,
      fitPrior: hyper.fitPrior ?? true,
      varSmoothing: hyper.varSmoothing ?? 1e-9,
      binarize: hyper.binarize ?? 0,
      norm: hyper.norm ?? false,
    },
  )
  return est
}

/**
 * Gaussian naive Bayes (sklearn.naive_bayes.GaussianNB): per-class means and variances (plus a small share of the
 * largest variance). Exposition: the likelihood table (each feature's average log-likelihood contribution to its
 * class) and, for two features, the decision surface; `explain(row)` gives any query's table, `describe()` the Bayes
 * network.
 *
 * @param hyper - Variance smoothing (default 1e-9) and whether to fit priors.
 * @returns The estimator.
 */
export function gaussianNB(
  hyper: Partial<NaiveBayesHyper> = {},
): ExplainedEstimator<NaiveBayesHyper, FittedNaiveBayes> {
  return naiveBayes('gaussian', hyper)
}

/**
 * Multinomial naive Bayes for counts (sklearn.naive_bayes.MultinomialNB), with additive smoothing. Exposition as
 * {@link gaussianNB}.
 *
 * @param hyper - Smoothing alpha (default 1) and whether to fit priors.
 * @returns The estimator.
 */
export function multinomialNB(
  hyper: Partial<NaiveBayesHyper> = {},
): ExplainedEstimator<NaiveBayesHyper, FittedNaiveBayes> {
  return naiveBayes('multinomial', hyper)
}

/**
 * Bernoulli naive Bayes for binary features (sklearn.naive_bayes.BernoulliNB), features binarised at a threshold.
 *
 * @param hyper - Smoothing alpha (default 1), binarisation threshold (default 0) and whether to fit priors.
 * @returns The estimator.
 */
export function bernoulliNB(
  hyper: Partial<NaiveBayesHyper> = {},
): ExplainedEstimator<NaiveBayesHyper, FittedNaiveBayes> {
  return naiveBayes('bernoulli', hyper)
}

/**
 * Complement naive Bayes (sklearn.naive_bayes.ComplementNB): weights from the complement of each class, suited to
 * imbalanced text.
 *
 * @param hyper - Smoothing alpha (default 1) and weight normalisation.
 * @returns The estimator.
 */
export function complementNB(
  hyper: Partial<NaiveBayesHyper> = {},
): ExplainedEstimator<NaiveBayesHyper, FittedNaiveBayes> {
  return naiveBayes('complement', hyper)
}

/**
 * Categorical naive Bayes (sklearn.naive_bayes.CategoricalNB) for features coded 0, 1, 2, ...
 *
 * @param hyper - Smoothing alpha (default 1) and whether to fit priors.
 * @returns The estimator.
 */
export function categoricalNB(
  hyper: Partial<NaiveBayesHyper> = {},
): ExplainedEstimator<NaiveBayesHyper, FittedNaiveBayes> {
  return naiveBayes('categorical', hyper)
}

/** {@link gaussianNB} as a class. */
export class GaussianNB extends ExplainedEstimatorClass<NaiveBayesHyper, FittedNaiveBayes> {
  /** @param hyper - As {@link gaussianNB}. */
  constructor(hyper: Partial<NaiveBayesHyper> = {}) {
    super(gaussianNB(hyper))
  }
}

/** {@link multinomialNB} as a class. */
export class MultinomialNB extends ExplainedEstimatorClass<NaiveBayesHyper, FittedNaiveBayes> {
  /** @param hyper - As {@link multinomialNB}. */
  constructor(hyper: Partial<NaiveBayesHyper> = {}) {
    super(multinomialNB(hyper))
  }
}

/** {@link bernoulliNB} as a class. */
export class BernoulliNB extends ExplainedEstimatorClass<NaiveBayesHyper, FittedNaiveBayes> {
  /** @param hyper - As {@link bernoulliNB}. */
  constructor(hyper: Partial<NaiveBayesHyper> = {}) {
    super(bernoulliNB(hyper))
  }
}
