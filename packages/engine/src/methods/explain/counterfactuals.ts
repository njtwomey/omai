/**
 * Counterfactual, contrastive and example-based explanations of fitted classifiers as exposition records: growing
 * spheres, FACE, Wachter, DiCE, prototype-guided counterfactuals and CEM's pertinent negatives and positives under
 * actionability constraints, with their robustness to refitting (`counterfactuals`); prototypes and criticisms
 * (`prototypes`); and trust scores against the model's confidence (`curves`).
 */

import { argmaxOf, WhiteboxError } from '../../lib/base/index.ts'
import * as explain from '../../lib/explain/index.ts'
import {
  changedFeatures,
  counterfactualSet,
  curveSet,
  type Explained,
  prototypeSet,
} from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  type Estimator,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  takeData,
  targetVector,
} from '../../lib/learn/index.ts'
import { svd } from '../../lib/numerics/linalg/index.ts'
import { bootstrapIndices, foldIn, type Key, key as makeKey } from '../../lib/random/index.ts'
import {
  add,
  fromStorage,
  matmul,
  sigmoid,
  sliceAxis,
  softmax,
  squeeze,
  sub,
  type Tensor,
  toArray,
  toFlat,
  type Value,
} from '../../lib/tensor/index.ts'
import { linearScoresOf } from './_models.ts'

/** The counterfactual methods {@link explainCounterfactuals} offers. */
export type CounterfactualMethod = 'growing spheres' | 'face' | 'wachter' | 'dice' | 'prototype' | 'cem'

/** A classifier as differentiable functions: its class scores and its probability of one class. */
interface Differentiable {
  readonly scores: (v: Value) => Value
  readonly probability: (k: number) => (v: Value) => Value
}

/**
 * A classifier linear in its inputs (logistic regression, a linear SVM's scores), read off its decision function at
 * d + 1 points and checked at a data row, as differentiable functions; undefined for other models.
 */
function differentiableOf(fitted: Fitted, x: ReturnType<typeof matrixView>): Differentiable | undefined {
  const lin = linearScoresOf(fitted, x)
  if (lin === undefined) return undefined
  const { w, b } = lin
  const k = b.length
  if (k === 1) {
    // A binary score s: class scores [-s/2, s/2], P(1) = sigmoid(s).
    const w2 = w.map((row) => [-(row[0] ?? 0) / 2, (row[0] ?? 0) / 2])
    const b2 = [[-(b[0] ?? 0) / 2, (b[0] ?? 0) / 2]]
    const score = (u: Value): Value =>
      add(
        matmul(
          u,
          w.map((row) => [row[0] ?? 0]),
        ),
        [[b[0] ?? 0]],
      )
    return {
      scores: (u) => add(matmul(u, w2), b2),
      probability: (cls) => (u) => {
        const p = squeeze(sigmoid(score(u)), -1)
        return cls === 1 ? p : sub(1, p)
      },
    }
  }
  const scores = (u: Value): Value => add(matmul(u, w), [b])
  return {
    scores,
    probability: (cls) => (u) => squeeze(sliceAxis(softmax(scores(u), -1), -1, cls, cls + 1), -1),
  }
}

/** Options of {@link explainCounterfactuals}. */
export interface CounterfactualOptions {
  /** The method (default growing spheres). */
  readonly method?: CounterfactualMethod
  /** The class sought (default the most likely class other than the predicted one). */
  readonly desired?: number
  /** Counterfactuals sought, for DiCE (default 3). */
  readonly count?: number
  /** Actionability constraints. */
  readonly constraints?: explain.ActionConstraints
  /** Refits on bootstrap resamples of the data to measure each counterfactual's robustness (with `estimator`). */
  readonly refits?: number
  readonly estimator?: Estimator
  readonly key?: Key
}

/** Counterfactuals of one row. */
export interface CounterfactualExplanation {
  readonly method: CounterfactualMethod
  readonly points: readonly (readonly number[])[]
  readonly predictions: readonly number[]
  readonly valid: readonly boolean[]
}

/**
 * Counterfactual explanations of a fitted classifier's prediction for one row: what would have to change for it to
 * predict the desired class. Growing spheres and FACE need only predictions (any model); Wachter, DiCE, prototype-
 * guided counterfactuals and CEM (a pertinent negative and a pertinent positive) need a model linear in its inputs
 * (logistic regression) to differentiate. Distances use the data's median absolute deviations; actionability
 * constraints are checked (and enforced where the method can); with an estimator and refits, each counterfactual's
 * robustness to refitting on resampled data is measured.
 *
 * @param fitted - A fitted classifier.
 * @param data - The data (features, labels; the row explained and the background).
 * @param row - The row explained.
 * @param options - method, desired, count, constraints, refits, estimator and key.
 * @returns The counterfactuals with a `counterfactuals` record.
 */
export function explainCounterfactuals(
  fitted: Fitted,
  data: LearnData,
  row: number,
  options: CounterfactualOptions = {},
): Explained<CounterfactualExplanation> {
  if (fitted.kind !== 'classifier' || fitted.predictProba === undefined)
    throw new WhiteboxError('value', 'explainCounterfactuals: needs a classifier with probabilities')
  const x = matrixView(data.x, 'explainCounterfactuals')
  if (!(Number.isInteger(row) && row >= 0 && row < x.n))
    throw new WhiteboxError('value', `explainCounterfactuals: no row ${row}`)
  const d = x.d
  const features = featureNamesOf(data, d)
  const classes = (fitted as { readonly classes?: readonly number[] }).classes ?? []
  const names = classNamesOf(data, classes)
  const x0 = Array.from(x.data.subarray(row * d, (row + 1) * d))
  const proba = fitted.predictProba
  const p0 = toFlat(proba([x0]))
  const predicted = argmaxOf(p0)
  const desired =
    options.desired ??
    p0
      .map((p, c) => ({ p, c }))
      .filter((e) => e.c !== predicted)
      .sort((a, b) => b.p - a.p)[0]?.c ??
    0
  const pDesired = (rows: readonly (readonly number[])[]): number[] => {
    const out = proba(rows as number[][])
    const k = out.shape[1] ?? 1
    return toFlat(out).filter((_, i) => i % k === desired)
  }
  const predictDesired = (t: Tensor): Tensor => {
    const out = proba(t)
    const k = out.shape[1] ?? 1
    return fromStorage(
      Float64Array.from(toFlat(out).filter((_, i) => i % k === desired)),
      [out.shape[0] ?? 0],
      'float64',
    )
  }
  const accept = (p: number): boolean => p > 0.5
  const scales = explain.medianAbsoluteDeviations(fromStorage(x.data, [x.n, d], 'float64'))
  const constraints = options.constraints ?? {}
  const method = options.method ?? 'growing spheres'
  const key = options.key ?? makeKey(0)
  const found: { x: number[]; label?: string }[] = []
  const needGradient = (): Differentiable => {
    const g = differentiableOf(fitted, x)
    if (g === undefined)
      throw new WhiteboxError(
        'value',
        `explainCounterfactuals: ${method} needs a model linear in its inputs; use growing spheres or FACE`,
      )
    return g
  }
  const bounds = Array.from({ length: d }, (_, j) => {
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (let i = 0; i < x.n; i++) {
      const v = x.data[i * d + j] ?? 0
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    return [lo, hi] as [number, number]
  })
  if (method === 'growing spheres') {
    const cf = explain.growingSpheres(predictDesired, x0, { accept, scales, constraints, key })
    if (cf !== undefined) found.push({ x: [...cf.x] })
  } else if (method === 'face') {
    const path = explain.faceCounterfactual(predictDesired, x0, fromStorage(x.data, [x.n, d], 'float64'), {
      accept,
      scales,
    })
    if (path !== undefined) found.push({ x: [...path.counterfactual], label: `FACE (${path.path.length} steps)` })
  } else if (method === 'wachter') {
    const g = needGradient()
    const cf = explain.wachterCounterfactual(g.probability(desired), x0, {
      target: 0.6,
      tolerance: 0.08,
      scales,
      ...(constraints.immutable === undefined ? {} : { immutable: constraints.immutable }),
      key,
    })
    found.push({ x: explain.projectActions(x0, cf.x, constraints) })
  } else if (method === 'dice') {
    const g = needGradient()
    const r = explain.diverseCounterfactuals(g.probability(desired), x0, { count: options.count ?? 3, scales, key })
    r.points.forEach((p, i) => {
      found.push({ x: explain.projectActions(x0, p, constraints), label: `#${i + 1}` })
    })
  } else if (method === 'prototype') {
    const g = needGradient()
    const y = data.y === undefined ? [] : Array.from(targetVector(data.y, 'explainCounterfactuals', x.n))
    const labels = y.map((v) => Math.max(0, classes.indexOf(v)))
    const r = explain.prototypeCounterfactual(g.scores, x0, fromStorage(x.data, [x.n, d], 'float64'), labels, {
      targetClass: desired,
      bounds,
    })
    if (r !== undefined) found.push({ x: explain.projectActions(x0, r.x, constraints), label: 'prototype-guided' })
  } else {
    const g = needGradient()
    const pn = explain.pertinentNegative(g.scores, x0, { bounds })
    if (pn !== undefined) found.push({ x: [...pn.x], label: 'pertinent negative' })
    const pp = explain.pertinentPositive(g.scores, x0)
    if (pp !== undefined) found.push({ x: [...pp.x], label: 'pertinent positive' })
  }
  const preds = found.length === 0 ? [] : pDesired(found.map((f) => f.x))
  // Robustness to refitting: the share of bootstrap refits that still give the desired class.
  let robust: number[] | undefined
  if (options.estimator !== undefined && (options.refits ?? 0) > 0 && found.length > 0) {
    const est = options.estimator
    const models = Array.from({ length: options.refits ?? 0 }, (_, r) => {
      const idx = toFlat(bootstrapIndices(foldIn(key, 500 + r), x.n))
      return est.fit(takeData(data, idx), { key: foldIn(key, 900 + r), explain: false })
    })
    robust = found.map((f) => {
      const ok = models.filter((m) => {
        const p = m.predictProba === undefined ? undefined : toFlat(m.predictProba([f.x]))
        return p !== undefined && argmaxOf(p) === desired
      })
      return ok.length / models.length
    })
  }
  const distance = (p: readonly number[]): number =>
    p.reduce((s, v, j) => s + Math.abs(v - (x0[j] ?? 0)) / (scales[j] ?? 1), 0)
  const entries = found.map((f, i) => {
    const pp = method === 'cem' && f.label === 'pertinent positive'
    const prediction = preds[i] ?? Number.NaN
    return {
      x: f.x,
      prediction,
      distance: distance(f.x),
      // A pertinent positive keeps the original class; everything else seeks the desired one.
      valid: pp ? !accept(prediction) : accept(prediction),
      changed: changedFeatures(x0, f.x),
      label: f.label ?? method,
      violations: pp ? 0 : explain.actionViolations(x0, f.x, constraints).length,
      ...(robust === undefined ? {} : { robustness: robust[i] ?? 0 }),
    }
  })
  const desiredName = names[desired] ?? String(desired)
  return {
    method,
    points: entries.map((e) => e.x),
    predictions: entries.map((e) => e.prediction),
    valid: entries.map((e) => e.valid),
    exposition: [
      counterfactualSet({
        method,
        features,
        output: `P(${desiredName})`,
        original: x0,
        originalPrediction: p0[desired] ?? Number.NaN,
        desired: `P(${desiredName}) > 0.5`,
        counterfactuals: entries,
        ...(constraints.immutable === undefined ? {} : { immutable: constraints.immutable }),
        scales,
        index: row,
      }),
    ],
  }
}

/** Options of {@link prototypesAndCriticisms}. */
export interface PrototypeOptions {
  readonly nPrototypes?: number
  readonly nCriticisms?: number
  /** The RBF kernel's gamma on standardised features (default 1 / d). */
  readonly gamma?: number
  /** Rows used, spread over the data (default 400). */
  readonly maxRows?: number
  /** Rows are images of this shape. */
  readonly imageShape?: readonly [number, number]
}

/**
 * Prototypes and criticisms of a dataset (MMD-critic) on standardised features: the rows that summarise it and those
 * it summarises badly, with the witness function at every row and the rows' coordinates on their first two principal
 * components (or images, with `imageShape`), in a `prototypes` record.
 *
 * @param data - The data (labels, when present, colour the scatter).
 * @param options - nPrototypes, nCriticisms, gamma, maxRows and imageShape.
 * @returns The chosen rows with a `prototypes` record.
 */
export function prototypesAndCriticisms(
  data: LearnData,
  options: PrototypeOptions = {},
): Explained<{ readonly prototypes: readonly number[]; readonly criticisms: readonly number[] }> {
  const full = matrixView(data.x, 'prototypesAndCriticisms')
  const n = Math.min(full.n, options.maxRows ?? 400)
  const rowsIdx = Array.from({ length: n }, (_, i) => Math.floor((i * full.n) / n))
  const d = full.d
  const raw = rowsIdx.map((i) => Array.from(full.data.subarray(i * d, (i + 1) * d)))
  const mean = Array.from({ length: d }, (_, j) => raw.reduce((s, r) => s + (r[j] ?? 0), 0) / n)
  const sd = Array.from(
    { length: d },
    (_, j) => Math.sqrt(raw.reduce((s, r) => s + ((r[j] ?? 0) - (mean[j] ?? 0)) ** 2, 0) / n) || 1,
  )
  const z = raw.map((r) => r.map((v, j) => (v - (mean[j] ?? 0)) / (sd[j] ?? 1)))
  const r = explain.mmdCritic(z, {
    nPrototypes: options.nPrototypes ?? 6,
    nCriticisms: options.nCriticisms ?? 3,
    ...(options.gamma === undefined ? {} : { gamma: options.gamma }),
  })
  const labelsAll =
    data.y === undefined
      ? undefined
      : (() => {
          const y = targetVector(data.y, 'prototypesAndCriticisms', full.n)
          const classes = Array.from(new Set(y)).sort((a, b) => a - b)
          const names = classNamesOf(data, classes)
          return rowsIdx.map((i) => names[classes.indexOf(y[i] ?? 0)] ?? String(y[i]))
        })()
  let points: [number, number][] | undefined
  let axes: [string, string] | undefined
  if (options.imageShape === undefined && d >= 2) {
    const { vh } = svd(z)
    const v = toArray(vh as Tensor) as number[][]
    points = z.map((row) => [
      row.reduce((s, u, j) => s + u * (v[0]?.[j] ?? 0), 0),
      row.reduce((s, u, j) => s + u * (v[1]?.[j] ?? 0), 0),
    ])
    axes = ['PC 1', 'PC 2']
  }
  const prototypes = r.prototypes.map((i) => rowsIdx[i] ?? i)
  const criticisms = r.criticisms.map((i) => rowsIdx[i] ?? i)
  return {
    prototypes,
    criticisms,
    exposition: [
      prototypeSet({
        method: 'MMD-critic',
        features: featureNamesOf(data, d),
        prototypes,
        criticisms,
        prototypeData: r.prototypes.map((i) => raw[i] ?? []),
        criticismData: r.criticisms.map((i) => raw[i] ?? []),
        witness: r.witness,
        ...(points === undefined ? {} : { points }),
        ...(axes === undefined ? {} : { axes }),
        ...(labelsAll === undefined ? {} : { labels: labelsAll }),
        ...(options.imageShape === undefined ? {} : { imageShape: options.imageShape }),
      }),
    ],
  }
}

/**
 * Trust scores of a fitted classifier on held-out rows (Jiang et al. 2018): how much closer each row is to its
 * predicted class's dense core than to any other's, and the accuracy among the rows kept as fewer are trusted, by trust
 * score and by the model's own confidence (its largest probability), as a `curves` record: a trust score worth having
 * rises more steeply.
 *
 * @param fitted - A fitted classifier.
 * @param train - Training data (the classes' cores).
 * @param test - Held-out data with labels.
 * @param options - alpha and kFilter of the trust score.
 * @returns The scores with a `curves` record.
 */
export function trustScoreStudy(
  fitted: Fitted,
  train: LearnData,
  test: LearnData,
  options: explain.TrustScoreOptions = {},
): Explained<{ readonly scores: readonly number[] }> {
  if (fitted.predictProba === undefined) throw new WhiteboxError('value', 'trustScoreStudy: needs probabilities')
  const classes = (fitted as { readonly classes?: readonly number[] }).classes ?? []
  const t = matrixView(test.x, 'trustScoreStudy')
  if (test.y === undefined) throw new WhiteboxError('value', 'trustScoreStudy: the test data need labels')
  const y = targetVector(test.y, 'trustScoreStudy', t.n)
  const proba = fitted.predictProba(fromStorage(t.data, [t.n, t.d], 'float64'))
  const k = proba.shape[1] ?? 1
  const pv = toFlat(proba)
  const predictedIdx = Array.from({ length: t.n }, (_, i) => argmaxOf(pv.slice(i * k, (i + 1) * k)))
  const predicted = predictedIdx.map((c) => classes[c] ?? c)
  const confidence = predictedIdx.map((c, i) => pv[i * k + c] ?? 0)
  const r = explain.trustScores(train, fromStorage(t.data, [t.n, t.d], 'float64'), predicted, options)
  const correct = predicted.map((p, i) => p === y[i])
  const curve = (score: readonly number[]): number[] => {
    const order = score.map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s || a.i - b.i)
    return PERCENTILES.map((q) => {
      const keep = Math.max(1, Math.round((q / 100) * t.n))
      return order.slice(0, keep).filter((e) => correct[e.i]).length / keep
    })
  }
  return {
    scores: r.scores,
    exposition: [
      curveSet({
        x: [...PERCENTILES],
        series: [
          { name: 'trust score', y: curve(r.scores.map((s) => (Number.isFinite(s) ? s : 1e300))) },
          { name: 'model confidence', y: curve(confidence) },
        ],
        xLabel: 'rows kept, most trusted first (%)',
        yLabel: 'accuracy among kept rows',
      }),
    ],
  }
}

const PERCENTILES: readonly number[] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
