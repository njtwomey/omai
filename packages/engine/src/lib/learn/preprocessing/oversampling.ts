/**
 * The SMOTE variants beyond the plain, borderline and ADASYN forms of `resampling.ts`, imbalanced-learn style: SMOTEN
 * for purely categorical features (neighbours by the value difference metric, each feature the most common category
 * among the neighbours), SMOTENC for continuous and categorical features together, SVM-SMOTE (synthesise around the
 * support vectors, extrapolating away from the safe ones) and k-means SMOTE (synthesise inside the clusters where the
 * class is dense enough, more where it is sparse). Categorical features are columns of category codes (any numbers);
 * synthetic rows only take categories that occur among the neighbours.
 */

import { OmaiError } from '../../base/index.ts'
import { kmeansCodebook } from '../../numerics/spatial/index.ts'
import { beta, choice as drawChoice, foldIn, uniform } from '../../random/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { keyFor } from '../_shared.ts'
import type { Estimator } from '../estimator.ts'
import { encodeLabels, type MatrixView, matrixView, takeRows } from '../validation.ts'
import {
  assemble,
  type Classes,
  classesOf,
  distancesFrom,
  drawPairs,
  fromRows,
  interpolate,
  modeOf,
  nearest,
} from './_sampling.ts'
import { type Resampler, type SamplingStrategy, samplingTargets } from './resampling.ts'

/** Options shared by the SMOTE variants. */
export interface SmoteVariantOptions {
  readonly samplingStrategy?: SamplingStrategy
  /** Neighbours within the class to take categories from or interpolate towards (default 5). */
  readonly kNeighbors?: number
}

/** The classes to over-sample, with how many synthetic samples each needs. */
function wanted(c: Classes, strategy: SamplingStrategy): { cls: number; label: number; need: number }[] {
  const targets = samplingTargets(
    c.members.map((m) => m.length),
    c.classes,
    strategy,
    'over',
  )
  return c.members.flatMap((members, cls) => {
    const t = targets[cls]
    return t === undefined || t <= members.length ? [] : [{ cls, label: c.classes[cls] ?? 0, need: t - members.length }]
  })
}

function requireNeighbours(c: Classes, cls: number, k: number, who: string): void {
  if ((c.members[cls]?.length ?? 0) <= k)
    throw new OmaiError('value', `${who}: class ${c.classes[cls]} needs more than ${k} samples`)
}

/** The k nearest of n items by a distance, nearest first, ties by index, skipping `self`. */
function kNearest(n: number, k: number, self: number, distance: (j: number) => number): number[] {
  const d = Array.from({ length: n }, (_, j) => distance(j))
  return Array.from(d.keys())
    .filter((j) => j !== self)
    .sort((p, q) => (d[p] ?? 0) - (d[q] ?? 0) || p - q)
    .slice(0, k)
}

/** Options of {@link valueDifferenceMetric}. */
export interface ValueDifferenceOptions {
  /** The norm `p` of the difference between two class distributions (default 1). */
  readonly k?: number
  /** The power the per-feature distance is raised to (default 2). */
  readonly r?: number
}

/**
 * The value difference metric of Stanfill and Waltz (imblearn ValueDifferenceMetric) for categorical features: two
 * rows are close when, feature by feature, their categories occur with the same class distribution in the training
 * data. For each feature the distance is `|P(class | a) - P(class | b)|_k ** r`, summed over features; a category not
 * seen in the training data is an error.
 *
 * @param x - Training features `[n, d]` of category codes (any numbers).
 * @param y - Training labels `[n]`.
 * @param options - The norm `k` (1) and power `r` (2).
 * @returns A function giving the distance matrix between the rows of `a` and those of `b` (default `a`), `[na, nb]`.
 * @throws OmaiError with code `value` for a category not present in the training data.
 * @example
 * ```ts
 * import { valueDifferenceMetric } from 'omai-engine/learn/preprocessing'
 * import { toFlat } from 'omai-engine/tensor'
 * const vdm = valueDifferenceMetric([[0], [0], [1], [1], [2]], [0, 0, 1, 1, 0])
 * toFlat(vdm([[0], [1], [2]])) // => [0, 4, 0, 4, 0, 4, 0, 4, 0]
 * ```
 */
export function valueDifferenceMetric(
  x: TensorLike,
  y: TensorLike,
  options: ValueDifferenceOptions = {},
): (a: TensorLike, b?: TensorLike) => Tensor {
  const who = 'valueDifferenceMetric'
  const m = matrixView(x, who)
  const { classes, codes } = encodeLabels(y, who, m.n)
  const k = options.k ?? 1
  const r = options.r ?? 2
  const categories = Array.from({ length: m.d }, (_, j) =>
    Array.from(new Set(Array.from({ length: m.n }, (_, i) => m.data[i * m.d + j] ?? 0))).sort((a, b) => a - b),
  )
  const proba = categories.map((cats, j) => {
    const table = new Float64Array(cats.length * classes.length)
    for (let i = 0; i < m.n; i++) {
      const cat = cats.indexOf(m.data[i * m.d + j] ?? 0)
      table[cat * classes.length + (codes[i] ?? 0)] = (table[cat * classes.length + (codes[i] ?? 0)] ?? 0) + 1
    }
    for (let cat = 0; cat < cats.length; cat++) {
      let total = 0
      for (let c = 0; c < classes.length; c++) total += table[cat * classes.length + c] ?? 0
      for (let c = 0; c < classes.length; c++)
        table[cat * classes.length + c] = (table[cat * classes.length + c] ?? 0) / total
    }
    return table
  })
  const lookup = (view: MatrixView): Int32Array =>
    Int32Array.from({ length: view.n * view.d }, (_, t) => {
      const cat = categories[t % view.d]?.indexOf(view.data[t] ?? 0) ?? -1
      if (cat < 0)
        throw new OmaiError(
          'value',
          `${who}: category ${view.data[t]} of feature ${t % view.d} was not in the training data`,
        )
      return cat
    })
  return (a, b) => {
    const va = matrixView(a, who)
    const vb = b === undefined ? va : matrixView(b, who)
    if (va.d !== m.d || vb.d !== m.d) throw new OmaiError('shape', `${who}: expected ${m.d} features`)
    const ca = lookup(va)
    const cb = b === undefined ? ca : lookup(vb)
    const out = new Float64Array(va.n * vb.n)
    for (let j = 0; j < m.d; j++) {
      const table = proba[j] ?? new Float64Array(0)
      for (let p = 0; p < va.n; p++)
        for (let q = 0; q < vb.n; q++) {
          let s = 0
          for (let c = 0; c < classes.length; c++)
            s +=
              Math.abs(
                (table[(ca[p * m.d + j] ?? 0) * classes.length + c] ?? 0) -
                  (table[(cb[q * m.d + j] ?? 0) * classes.length + c] ?? 0),
              ) ** k
          out[p * vb.n + q] = (out[p * vb.n + q] ?? 0) + (s ** (1 / k)) ** r
        }
    }
    return fromStorage(out, [va.n, vb.n], 'float64')
  }
}

/**
 * SMOTEN (Chawla et al. 2002; imblearn SMOTEN): SMOTE for data whose features are all categorical. Neighbours are found
 * with the value difference metric among the class's samples (the sample itself and its k nearest), and a synthetic
 * sample takes, in each feature, the most frequent category among the neighbours of a randomly drawn sample (the
 * smallest code on a tie), so every category it holds occurs in the class.
 *
 * @param options - The sampling strategy and kNeighbors (default 5).
 * @returns The resampler.
 * @concept class-imbalance
 * @index estimator SMOTE for categorical features, by the value difference metric.
 * @example
 * ```ts
 * import { smoten } from 'omai-engine/learn/preprocessing'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [0, 0], [2, 1], [2, 0], [2, 1]]
 * const r = smoten({ kNeighbors: 2 }).fitResample({ x, y: [0, 0, 0, 0, 0, 0, 1, 1, 1] }, { key: key(0) })
 * toFlat(r.y).filter((v) => v === 1).length // => 6
 * ```
 */
export function smoten(options: SmoteVariantOptions = {}): Resampler {
  const k = options.kNeighbors ?? 5
  return {
    name: 'smoten',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'smoten')
      const key = keyFor(fit, 'smoten')
      const distance = valueDifferenceMetric(fromRows(c.m), c.labels)
      const synthetic: number[][] = []
      const labels: number[] = []
      for (const { cls, label, need } of wanted(c, options.samplingStrategy ?? 'auto')) {
        requireNeighbours(c, cls, k, 'smoten')
        const members = c.members[cls] ?? []
        const xc = takeRows(c.m, members)
        const d = toFlat(distance(fromRows(xc)))
        const around = members.map((_, i) => kNearest(xc.n, k + 1, -1, (j) => d[i * xc.n + j] ?? 0))
        for (const pick of toFlat(drawChoice(foldIn(key, cls), xc.n, need, { replace: true }))) {
          synthetic.push(
            Array.from({ length: xc.d }, (_, j) => modeOf((around[pick] ?? []).map((q) => xc.data[q * xc.d + j] ?? 0))),
          )
          labels.push(label)
        }
      }
      return assemble(
        c,
        Array.from({ length: c.m.n }, (_, i) => i),
        synthetic,
        labels,
      )
    },
  }
}

/** Options of {@link smotenc}. */
export interface SmoteNcOptions extends SmoteVariantOptions {
  /** The columns of category codes; the rest are continuous. */
  readonly categorical: readonly number[]
}

/**
 * SMOTE-NC (Chawla et al. 2002; imblearn SMOTENC): SMOTE for continuous and categorical features. Within a class, a
 * mismatch in a categorical feature adds the class's median standard deviation of the continuous features, squared, to
 * the squared distance. A synthetic sample interpolates the continuous features between a sample and one of its k
 * nearest neighbours, and takes each categorical feature as the most frequent category among those k neighbours (ties
 * at random).
 *
 * @param options - The categorical columns, the sampling strategy and kNeighbors (default 5).
 * @returns The resampler.
 * @throws OmaiError with code `value` unless some columns are categorical and some continuous.
 * @concept class-imbalance
 * @index estimator SMOTE for mixed continuous and categorical features.
 * @example
 * ```ts
 * import { smotenc } from 'omai-engine/learn/preprocessing'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 1], [0.1, 1], [0.2, 0], [0.3, 0], [0.4, 1], [5, 0], [5.1, 1], [5.2, 1]]
 * const r = smotenc({ categorical: [1], kNeighbors: 2 }).fitResample({ x, y: [0, 0, 0, 0, 0, 1, 1, 1] }, { key: key(0) })
 * toFlat(r.y).filter((v) => v === 1).length // => 5
 * ```
 */
export function smotenc(options: SmoteNcOptions): Resampler {
  const k = options.kNeighbors ?? 5
  return {
    name: 'smotenc',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'smotenc')
      const cat = [...options.categorical].sort((a, b) => a - b)
      const cont = Array.from({ length: c.m.d }, (_, j) => j).filter((j) => !cat.includes(j))
      if (cat.length === 0 || cont.length === 0)
        throw new OmaiError('value', 'smotenc: needs both categorical and continuous features')
      const key = keyFor(fit, 'smotenc')
      const d = c.m.d
      const synthetic: number[][] = []
      const labels: number[] = []
      for (const { cls, label, need } of wanted(c, options.samplingStrategy ?? 'auto')) {
        requireNeighbours(c, cls, k, 'smotenc')
        const xc = takeRows(c.m, c.members[cls] ?? [])
        const stds = cont.map((j) => {
          let mean = 0
          for (let i = 0; i < xc.n; i++) mean += (xc.data[i * d + j] ?? 0) / xc.n
          let v = 0
          for (let i = 0; i < xc.n; i++) v += ((xc.data[i * d + j] ?? 0) - mean) ** 2 / xc.n
          return Math.sqrt(v)
        })
        const sorted = [...stds].sort((a, b) => a - b)
        const median = ((sorted[(sorted.length - 1) >> 1] ?? 0) + (sorted[sorted.length >> 1] ?? 0)) / 2
        const squared = (i: number, j: number): number => {
          let s = 0
          for (const f of cont) s += ((xc.data[i * d + f] ?? 0) - (xc.data[j * d + f] ?? 0)) ** 2
          for (const f of cat) if (xc.data[i * d + f] !== xc.data[j * d + f]) s += median * median
          return s
        }
        const around = Array.from({ length: xc.n }, (_, i) => kNearest(xc.n, k, i, (j) => squared(i, j)))
        const { row, col, step } = drawPairs(foldIn(key, cls), need, xc.n, k)
        const ties = toFlat(uniform(foldIn(key, 5000 + cls), [need * cat.length]))
        row.forEach((r, s) => {
          const nb = around[r]?.[col[s] ?? 0] ?? 0
          const made = new Array<number>(d).fill(0)
          for (const f of cont) {
            const a = xc.data[r * d + f] ?? 0
            made[f] = a + (step[s] ?? 0) * ((xc.data[nb * d + f] ?? 0) - a)
          }
          cat.forEach((f, q) => {
            const votes = new Map<number, number>()
            for (const t of around[r] ?? [])
              votes.set(xc.data[t * d + f] ?? 0, (votes.get(xc.data[t * d + f] ?? 0) ?? 0) + 1)
            const top = Math.max(...votes.values())
            const tied = Array.from(votes)
              .filter(([, n]) => n === top)
              .map(([v]) => v)
              .sort((a, b) => a - b)
            made[f] = tied[Math.min(tied.length - 1, Math.floor((ties[s * cat.length + q] ?? 0) * tied.length))] ?? 0
          })
          synthetic.push(made)
          labels.push(label)
        })
      }
      return assemble(
        c,
        Array.from({ length: c.m.n }, (_, i) => i),
        synthetic,
        labels,
      )
    },
  }
}

/** Options of {@link svmSmote}. */
export interface SvmSmoteOptions extends SmoteVariantOptions {
  /** A support vector classifier whose fitted parameters list `supportIndex` (the rows that are support vectors), such as `svc` of `omai-engine/svm`. */
  readonly svm: Estimator
  /** Neighbours over all the data that decide whether a support vector is noise or in danger (default 10). */
  readonly mNeighbors?: number
  /** How far beyond a safe support vector, away from its neighbour, to extrapolate, as a fraction of the distance (default 0.5). */
  readonly outStep?: number
}

/**
 * SVM-SMOTE (Nguyen et al. 2011; imblearn SVMSMOTE): fit a support vector machine and synthesise around the class's
 * support vectors. Those whose m nearest samples are all of other classes are noise and are dropped; those with at
 * least half of them in other classes are in danger and interpolate towards their nearest class neighbours (a
 * Beta(10, 10) share of the new samples); the safe ones extrapolate away from them, by `outStep`, to widen the class.
 * When one of the groups is empty the other takes all the new samples (imbalanced-learn then makes fewer).
 *
 * @param options - The SVM, mNeighbors (10), kNeighbors (5), outStep (0.5) and the sampling strategy.
 * @returns The resampler.
 * @throws OmaiError with code `value` when the SVM exposes no `supportIndex` or every support vector is noise.
 * @concept class-imbalance
 * @index estimator SMOTE around the support vectors of an SVM.
 * @example
 * ```ts
 * import { svmSmote } from 'omai-engine/learn/preprocessing'
 * import { svc } from 'omai-engine/svm'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [0.5, 0.2], [1, 0.1], [0.2, 0.9], [1.1, 1], [0.6, 0.6], [0.8, 0.5], [1.4, 1.2]]
 * const y = [0, 0, 0, 0, 0, 1, 1, 1]
 * const r = svmSmote({ svm: svc.with({ kernel: 'rbf' }), kNeighbors: 2, mNeighbors: 4 }).fitResample({ x, y }, { key: key(1) })
 * toFlat(r.y).filter((v) => v === 1).length // => 5
 * ```
 */
export function svmSmote(options: SvmSmoteOptions): Resampler {
  const k = options.kNeighbors ?? 5
  const m = options.mNeighbors ?? 10
  const outStep = options.outStep ?? 0.5
  return {
    name: 'svm-smote',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'svmSmote')
      const key = keyFor(fit, 'svmSmote')
      const jobs = wanted(c, options.samplingStrategy ?? 'auto')
      const synthetic: number[][] = []
      const labels: number[] = []
      if (jobs.length > 0) {
        const fitted = options.svm.fit({ x: data.x, y: data.y ?? [] }, { key: foldIn(key, 9000), explain: false })
        const support = (fitted.params as { readonly supportIndex?: unknown }).supportIndex
        if (!Array.isArray(support))
          throw new OmaiError('value', 'svmSmote: the svm must expose supportIndex in its parameters')
        for (const { cls, label, need } of jobs) {
          requireNeighbours(c, cls, k, 'svmSmote')
          const own = (support as number[]).filter((i) => c.labels[i] === label)
          const others = (i: number): number => nearest(c.m, i, c.m, m, i).filter((t) => c.labels[t] !== label).length
          const mm = Math.min(m, c.m.n - 1)
          const kept = own.filter((i) => others(i) < mm)
          if (kept.length === 0)
            throw new OmaiError('value', 'svmSmote: every support vector is noise; try another SMOTE variant')
          const inDanger = (i: number): boolean => others(i) >= mm / 2
          const danger = kept.filter(inDanger)
          const safe = kept.filter((i) => !inDanger(i))
          const xc = takeRows(c.m, c.members[cls] ?? [])
          const share = toFlat(beta(foldIn(key, 3000 + cls), [1], 10, 10))[0] ?? 0.5
          let fromDanger = Math.floor(share * (need + 1))
          if (danger.length === 0) fromDanger = 0
          if (safe.length === 0) fromDanger = need
          const around = (view: MatrixView): number[][] =>
            view.n === 0 ? [] : Array.from({ length: view.n }, (_, r) => nearest(view, r, xc, k + 1).slice(1))
          const group = (rows: readonly number[], salt: number, count: number, step: number): void => {
            const view = takeRows(c.m, rows)
            for (const p of interpolate(
              foldIn(key, salt + cls),
              view,
              Array.from(rows.keys()),
              around(view),
              xc,
              count,
              step,
            )) {
              synthetic.push(p)
              labels.push(label)
            }
          }
          if (danger.length > 0) group(danger, 100, fromDanger, 1)
          if (safe.length > 0) group(safe, 200, need - fromDanger, -outStep)
        }
      }
      return assemble(
        c,
        Array.from({ length: c.m.n }, (_, i) => i),
        synthetic,
        labels,
      )
    },
  }
}

/** Options of {@link kMeansSmote}. */
export interface KMeansSmoteOptions extends SmoteVariantOptions {
  /** Clusters of k-means over all the data (default 8). */
  readonly nClusters?: number
  /** The least share of the class a cluster must hold to be used: a number, or `'auto'` (half the class's share of the new samples). */
  readonly clusterBalanceThreshold?: 'auto' | number
  /** The exponent of the mean distance in a cluster's sparsity: a number, or `'auto'` (a function of the cluster size). */
  readonly densityExponent?: 'auto' | number
}

/**
 * K-means SMOTE (Last, Douzas and Bacao 2017; imblearn KMeansSMOTE): cluster all the data, keep the clusters in which
 * the class makes up a large enough share and has at least k + 1 samples, share the new samples among them in
 * proportion to each cluster's sparsity of the class (mean distance to the power of the density exponent, over the
 * count), and run SMOTE inside each cluster, so that no sample is made across a cluster of the other class. Each
 * cluster makes the ceiling of its share, so a few more than the target may appear.
 *
 * @param options - nClusters (8), clusterBalanceThreshold, densityExponent, kNeighbors (default 2) and the strategy.
 * @returns The resampler.
 * @throws OmaiError with code `value` when no cluster qualifies.
 * @concept class-imbalance
 * @index estimator SMOTE within k-means clusters of the class.
 * @example
 * ```ts
 * import { kMeansSmote } from 'omai-engine/learn/preprocessing'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [0.1], [0.2], [0.3], [0.4], [0.5], [5], [5.1], [5.2]]
 * const y = [0, 0, 0, 0, 0, 0, 1, 1, 1]
 * const r = kMeansSmote({ nClusters: 2, kNeighbors: 2 }).fitResample({ x, y }, { key: key(0) })
 * toFlat(r.y).filter((v) => v === 1).length >= 6 // => true
 * ```
 */
export function kMeansSmote(options: KMeansSmoteOptions = {}): Resampler {
  const k = options.kNeighbors ?? 2
  return {
    name: 'k-means-smote',
    kind: 'resampler',
    fitResample: (data, fit = {}) => {
      const c = classesOf(data, 'kMeansSmote')
      const key = keyFor(fit, 'kMeansSmote')
      const jobs = wanted(c, options.samplingStrategy ?? 'auto')
      const total = jobs.reduce((s, j) => s + j.need, 0)
      const synthetic: number[][] = []
      const labels: number[] = []
      for (const { cls, label, need } of jobs) {
        const clusters = Math.min(options.nClusters ?? 8, c.m.n)
        const assignment = kmeansCodebook(fromRows(c.m), clusters, { key: foldIn(key, cls) }).assignments
        const valid: { readonly rows: number[]; readonly sparsity: number }[] = []
        for (let cluster = 0; cluster < clusters; cluster++) {
          const inside = Array.from(assignment.keys()).filter((i) => assignment[i] === cluster)
          if (inside.length === 0) continue
          const own = inside.filter((i) => c.labels[i] === label)
          const share = own.length / inside.length
          const threshold =
            clusters === 1
              ? Number.NEGATIVE_INFINITY
              : (options.clusterBalanceThreshold ?? 'auto') === 'auto'
                ? need / total / 2
                : (options.clusterBalanceThreshold as number)
          if (share < threshold || share * inside.length < k + 1) continue
          valid.push({ rows: own, sparsity: sparsityOf(c, own, options.densityExponent ?? 'auto') })
        }
        if (valid.length === 0)
          throw new OmaiError(
            'value',
            `kMeansSmote: no cluster holds enough of class ${label}; lower the threshold or the clusters`,
          )
        const sum = valid.reduce((s, v) => s + v.sparsity, 0)
        valid.forEach((v, t) => {
          const view = takeRows(c.m, v.rows)
          const around = v.rows.map((_, r) => nearest(view, r, view, k, r))
          const count = Math.ceil((need * v.sparsity) / sum)
          for (const p of interpolate(
            foldIn(foldIn(key, 400 + cls), t),
            view,
            Array.from(v.rows.keys()),
            around,
            view,
            count,
            1,
          )) {
            synthetic.push(p)
            labels.push(label)
          }
        })
      }
      return assemble(
        c,
        Array.from({ length: c.m.n }, (_, i) => i),
        synthetic,
        labels,
      )
    },
  }
}

/** A cluster's sparsity of a class: its mean pairwise distance to the density exponent, per sample. */
function sparsityOf(c: Classes, rows: readonly number[], exponent: 'auto' | number): number {
  const n = rows.length
  const view = takeRows(c.m, rows)
  let sum = 0
  for (let i = 0; i < n; i++) for (const d2 of distancesFrom(view, i, view)) sum += Math.sqrt(d2)
  const mean = sum / (n * n - n)
  const power = exponent === 'auto' ? (Math.log(n) / Math.log(1.6)) ** 1.8 * 0.16 : exponent
  return mean ** power / n
}
