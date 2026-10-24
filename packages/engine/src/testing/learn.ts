// Reference estimators and golden helpers for the learn tests. Node only; not published.
// The estimators are small textbook learners that match their scikit-learn namesakes exactly, so that the toolkit
// (pipelines, cross-validation, calibration, conformal, inspection) can be checked against sklearn end to end without
// adding model families to the library.
import { describe, expect, it } from 'vitest'
import { argmaxOf, maxOf } from '../lib/base/index.ts'
import { defineEstimator, type Estimator, type Fitted, type LearnData } from '../lib/learn/estimator.ts'
import { encodeLabels, matrixView, sampleWeightsOf, targetVector } from '../lib/learn/validation.ts'
import { lstsq, solve } from '../lib/numerics/linalg/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat, type Value } from '../lib/tensor/index.ts'
import { decode, expectMatches, type GoldenCase, loadFixture } from './golden.ts'

/** What a learn golden case computes: one value per output key of the fixture. */
export type LearnCaseFn = (
  inputs: readonly Tensor[],
  params: Readonly<Record<string, unknown>>,
  c: GoldenCase,
) => Readonly<Record<string, Value | number | readonly number[] | string>>

/**
 * One test per case of a fixture whose outputs are a dictionary of arrays: every key of the expected output must be
 * produced and match within the case's tolerances.
 */
export function learnGolden(title: string, fixture: string, fns: Readonly<Record<string, LearnCaseFn>>): void {
  describe(title, () => {
    for (const c of loadFixture(fixture).cases) {
      it(c.name, () => {
        const f = fns[c.fn]
        if (f === undefined) throw new Error(`no function ${c.fn}`)
        const inputs = c.inputs.map((x) => decode(x) as Tensor)
        const params = Object.fromEntries(Object.entries(c.params).map(([k, v]) => [k, decode(v)]))
        const got = f(inputs, params, c)
        const want = c.output as Readonly<Record<string, unknown>>
        for (const [key, expected] of Object.entries(want)) {
          const value = got[key]
          if (value === undefined) throw new Error(`${c.name}: no output ${key}`)
          if (typeof value === 'string') expect(value, `${c.name}/${key}`).toBe(expected)
          else if (Array.isArray(value)) {
            const arr = value as readonly number[]
            expectMatches(
              fromStorage(Float64Array.from(arr), [arr.length], 'float64'),
              expected,
              c.rtol,
              c.atol,
              `${c.name}/${key}`,
            )
          } else expectMatches(value as Value | number, expected, c.rtol, c.atol, `${c.name}/${key}`)
        }
      })
    }
  })
}

/** A parameter of a golden case as a number. */
export function num(params: Readonly<Record<string, unknown>>, name: string, fallback?: number): number {
  const v = params[name]
  if (typeof v === 'number') return v
  if (v === undefined && fallback !== undefined) return fallback
  throw new Error(`parameter ${name} is not a number`)
}

/** A parameter of a golden case as a string. */
export function str(params: Readonly<Record<string, unknown>>, name: string, fallback?: string): string {
  const v = params[name]
  if (typeof v === 'string') return v
  if (v === undefined && fallback !== undefined) return fallback
  throw new Error(`parameter ${name} is not a string`)
}

/** A tensor's rows as nested number arrays. */
export function rows(t: TensorLike): number[][] {
  const m = matrixView(t, 'rows', { allowNaN: true })
  return Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
}

/** An int tensor's values as numbers. */
export function ints(t: TensorLike): number[] {
  return toFlat(t as Tensor)
}

interface LinearParams {
  readonly coef: number[]
  readonly intercept: number
  readonly [key: string]: number[] | number
}

function centred(data: LearnData, who: string, fitIntercept: boolean) {
  const m = matrixView(data.x, who)
  const y = targetVector(data.y ?? [], who, m.n)
  const w = sampleWeightsOf(data.weights, m.n, who)
  const sw = w.reduce((a, b) => a + b, 0)
  const xm = new Float64Array(m.d)
  let ym = 0
  if (fitIntercept) {
    for (let i = 0; i < m.n; i++) {
      for (let j = 0; j < m.d; j++) xm[j] = (xm[j] ?? 0) + ((w[i] ?? 0) * (m.data[i * m.d + j] ?? 0)) / sw
      ym += ((w[i] ?? 0) * (y[i] ?? 0)) / sw
    }
  }
  const xc = new Float64Array(m.n * m.d)
  const yc = new Float64Array(m.n)
  for (let i = 0; i < m.n; i++) {
    const r = Math.sqrt(w[i] ?? 0)
    for (let j = 0; j < m.d; j++) xc[i * m.d + j] = r * ((m.data[i * m.d + j] ?? 0) - (xm[j] ?? 0))
    yc[i] = r * ((y[i] ?? 0) - ym)
  }
  return { m, xm, ym, xc, yc }
}

interface LinearVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

function linearMethods(p: LinearParams, d: number): LinearVerbs {
  return {
    predict: (x: TensorLike): Tensor => {
      const m = matrixView(x, 'linear', { nFeatures: d })
      const out = new Float64Array(m.n)
      for (let i = 0; i < m.n; i++) {
        let s = p.intercept
        for (let j = 0; j < d; j++) s += (p.coef[j] ?? 0) * (m.data[i * d + j] ?? 0)
        out[i] = s
      }
      return fromStorage(out, [m.n], 'float64')
    },
    nFeaturesIn: d,
    featureImportances: p.coef.map(Math.abs),
  }
}

/** Ordinary least squares with intercept (sklearn.linear_model.LinearRegression), weighted. */
export function leastSquares(
  hyper: { readonly fitIntercept?: boolean } = {},
): Estimator<{ readonly fitIntercept: boolean }, Fitted<LinearParams> & LinearVerbs> {
  return defineEstimator(
    {
      name: 'least-squares',
      kind: 'regressor',
      fit: (data, h) => {
        const { m, xm, ym, xc, yc } = centred(data, 'leastSquares', h.fitIntercept)
        const coef = toFlat(lstsq(fromStorage(xc, [m.n, m.d], 'float64'), fromStorage(yc, [m.n], 'float64')))
        const intercept = h.fitIntercept ? ym - coef.reduce((s, c, j) => s + c * (xm[j] ?? 0), 0) : 0
        return { coef, intercept }
      },
      methods: (p) => linearMethods(p, p.coef.length),
    },
    { fitIntercept: hyper.fitIntercept ?? true },
  )
}

/** Ridge regression with an unpenalised intercept (sklearn.linear_model.Ridge), weighted. */
export function ridge(
  hyper: { readonly alpha?: number; readonly fitIntercept?: boolean } = {},
): Estimator<{ readonly alpha: number; readonly fitIntercept: boolean }, Fitted<LinearParams> & LinearVerbs> {
  return defineEstimator(
    {
      name: 'ridge',
      kind: 'regressor',
      fit: (data, h) => {
        const { m, xm, ym, xc, yc } = centred(data, 'ridge', h.fitIntercept)
        const g = new Float64Array(m.d * m.d)
        const b = new Float64Array(m.d)
        for (let i = 0; i < m.n; i++) {
          for (let j = 0; j < m.d; j++) {
            b[j] = (b[j] ?? 0) + (xc[i * m.d + j] ?? 0) * (yc[i] ?? 0)
            for (let k = 0; k < m.d; k++)
              g[j * m.d + k] = (g[j * m.d + k] ?? 0) + (xc[i * m.d + j] ?? 0) * (xc[i * m.d + k] ?? 0)
          }
        }
        for (let j = 0; j < m.d; j++) g[j * m.d + j] = (g[j * m.d + j] ?? 0) + h.alpha
        const coef = toFlat(solve(fromStorage(g, [m.d, m.d], 'float64'), fromStorage(b, [m.d], 'float64')))
        const intercept = h.fitIntercept ? ym - coef.reduce((s, c, j) => s + c * (xm[j] ?? 0), 0) : 0
        return { coef, intercept }
      },
      methods: (p) => linearMethods(p, p.coef.length),
    },
    { alpha: hyper.alpha ?? 1, fitIntercept: hyper.fitIntercept ?? true },
  )
}

interface CentroidParams {
  readonly classes: number[]
  readonly centroids: number[][]
  readonly [key: string]: number[] | number[][]
}

/** The nearest-centroid classifier (sklearn.neighbors.NearestCentroid, Euclidean). */
export function nearestCentroid(): Estimator<Record<string, never>, Fitted<CentroidParams>> {
  return defineEstimator(
    {
      name: 'nearest-centroid',
      kind: 'classifier',
      fit: (data) => {
        const m = matrixView(data.x, 'nearestCentroid')
        const { classes, codes } = encodeLabels(data.y ?? [], 'nearestCentroid', m.n)
        const sums = classes.map(() => new Float64Array(m.d))
        const counts = classes.map(() => 0)
        for (let i = 0; i < m.n; i++) {
          const c = codes[i] ?? 0
          counts[c] = (counts[c] ?? 0) + 1
          const s = sums[c] ?? new Float64Array(m.d)
          for (let j = 0; j < m.d; j++) s[j] = (s[j] ?? 0) + (m.data[i * m.d + j] ?? 0)
        }
        return { classes: [...classes], centroids: sums.map((s, c) => Array.from(s, (v) => v / (counts[c] ?? 1))) }
      },
      methods: (p) => ({
        classes: p.classes,
        nFeaturesIn: p.centroids[0]?.length ?? 0,
        predict: (x) => {
          const m = matrixView(x, 'nearestCentroid')
          const out = new Float64Array(m.n)
          for (let i = 0; i < m.n; i++) {
            let best = 0
            let bestD = Number.POSITIVE_INFINITY
            p.centroids.forEach((c, k) => {
              let d = 0
              for (let j = 0; j < m.d; j++) d += ((m.data[i * m.d + j] ?? 0) - (c[j] ?? 0)) ** 2
              if (d < bestD) {
                bestD = d
                best = k
              }
            })
            out[i] = p.classes[best] ?? 0
          }
          return fromStorage(out, [m.n], 'float64')
        },
      }),
    },
    {},
  )
}

interface NeighbourParams {
  readonly x: number[][]
  readonly y: number[]
  readonly classes: number[]
  readonly [key: string]: number[] | number[][]
}

function neighbours(p: NeighbourParams, k: number, x: TensorLike): number[][] {
  const m = matrixView(x, 'kNeighbours')
  const out: number[][] = []
  for (let i = 0; i < m.n; i++) {
    const d = p.x.map((r, t) => ({ t, d: r.reduce((s, v, j) => s + (v - (m.data[i * m.d + j] ?? 0)) ** 2, 0) }))
    d.sort((a, b) => a.d - b.d || a.t - b.t)
    out.push(d.slice(0, k).map((e) => e.t))
  }
  return out
}

/** k-nearest-neighbour classifier with uniform weights (sklearn.neighbors.KNeighborsClassifier, brute force). */
export function kNeighborsClassifier(
  hyper: { readonly k?: number } = {},
): Estimator<{ readonly k: number }, Fitted<NeighbourParams>> {
  return defineEstimator(
    {
      name: 'k-neighbors-classifier',
      kind: 'classifier',
      fit: (data) => {
        const m = matrixView(data.x, 'kNeighborsClassifier')
        const { classes, codes } = encodeLabels(data.y ?? [], 'kNeighborsClassifier', m.n)
        return { x: rows(data.x), y: Array.from(codes), classes: [...classes] }
      },
      methods: (p, h) => {
        const proba = (x: TensorLike): Tensor => {
          const nb = neighbours(p, h.k, x)
          const out = new Float64Array(nb.length * p.classes.length)
          nb.forEach((idx, i) => {
            for (const t of idx)
              out[i * p.classes.length + (p.y[t] ?? 0)] = (out[i * p.classes.length + (p.y[t] ?? 0)] ?? 0) + 1 / h.k
          })
          return fromStorage(out, [nb.length, p.classes.length], 'float64')
        }
        return {
          classes: p.classes,
          nFeaturesIn: p.x[0]?.length ?? 0,
          predictProba: proba,
          predict: (x) => {
            const pr = rows(proba(x))
            return fromStorage(
              Float64Array.from(pr, (r) => p.classes[argmaxOf(r)] ?? 0),
              [pr.length],
              'float64',
            )
          },
        }
      },
    },
    { k: hyper.k ?? 5 },
  )
}

/** k-nearest-neighbour regressor with uniform weights (sklearn.neighbors.KNeighborsRegressor, brute force). */
export function kNeighborsRegressor(
  hyper: { readonly k?: number } = {},
): Estimator<{ readonly k: number }, Fitted<NeighbourParams>> {
  return defineEstimator(
    {
      name: 'k-neighbors-regressor',
      kind: 'regressor',
      fit: (data) => ({
        x: rows(data.x),
        y: Array.from(targetVector(data.y ?? [], 'kNeighborsRegressor')),
        classes: [],
      }),
      methods: (p, h) => ({
        nFeaturesIn: p.x[0]?.length ?? 0,
        predict: (x) => {
          const nb = neighbours(p, h.k, x)
          return fromStorage(
            Float64Array.from(nb, (idx) => idx.reduce((s, t) => s + (p.y[t] ?? 0), 0) / h.k),
            [nb.length],
            'float64',
          )
        },
      }),
    },
    { k: hyper.k ?? 5 },
  )
}

interface GaussianNbParams {
  readonly classes: number[]
  readonly prior: number[]
  readonly theta: number[][]
  readonly var: number[][]
  readonly [key: string]: number[] | number[][]
}

/** Gaussian naive Bayes (sklearn.naive_bayes.GaussianNB with its variance smoothing). */
export function gaussianNB(
  hyper: { readonly varSmoothing?: number } = {},
): Estimator<{ readonly varSmoothing: number }, Fitted<GaussianNbParams>> {
  return defineEstimator(
    {
      name: 'gaussian-nb',
      kind: 'classifier',
      fit: (data, h) => {
        const m = matrixView(data.x, 'gaussianNB')
        const { classes, codes } = encodeLabels(data.y ?? [], 'gaussianNB', m.n)
        let maxVar = 0
        for (let j = 0; j < m.d; j++) {
          let s = 0
          for (let i = 0; i < m.n; i++) s += m.data[i * m.d + j] ?? 0
          const mu = s / m.n
          let v = 0
          for (let i = 0; i < m.n; i++) v += ((m.data[i * m.d + j] ?? 0) - mu) ** 2
          maxVar = Math.max(maxVar, v / m.n)
        }
        const eps = h.varSmoothing * maxVar
        const theta: number[][] = []
        const variance: number[][] = []
        const prior: number[] = []
        classes.forEach((_, c) => {
          const idx = Array.from(codes.keys()).filter((i) => codes[i] === c)
          prior.push(idx.length / m.n)
          const mu = Array.from(
            { length: m.d },
            (_, j) => idx.reduce((s, i) => s + (m.data[i * m.d + j] ?? 0), 0) / idx.length,
          )
          theta.push(mu)
          variance.push(
            mu.map((u, j) => idx.reduce((s, i) => s + ((m.data[i * m.d + j] ?? 0) - u) ** 2, 0) / idx.length + eps),
          )
        })
        return { classes: [...classes], prior, theta, var: variance }
      },
      methods: (p) => {
        const proba = (x: TensorLike): Tensor => {
          const m = matrixView(x, 'gaussianNB')
          const k = p.classes.length
          const out = new Float64Array(m.n * k)
          for (let i = 0; i < m.n; i++) {
            const jll = p.classes.map((_, c) => {
              let s = Math.log(p.prior[c] ?? 0)
              for (let j = 0; j < m.d; j++) {
                const v = p.var[c]?.[j] ?? 1
                s -=
                  0.5 * Math.log(2 * Math.PI * v) +
                  (0.5 * ((m.data[i * m.d + j] ?? 0) - (p.theta[c]?.[j] ?? 0)) ** 2) / v
              }
              return s
            })
            const top = maxOf(jll)
            const z = jll.reduce((s, v) => s + Math.exp(v - top), 0)
            jll.forEach((v, c) => {
              out[i * k + c] = Math.exp(v - top) / z
            })
          }
          return fromStorage(out, [m.n, k], 'float64')
        }
        return {
          classes: p.classes,
          nFeaturesIn: p.theta[0]?.length ?? 0,
          predictProba: proba,
          predict: (x) => {
            const pr = rows(proba(x))
            return fromStorage(
              Float64Array.from(pr, (r) => p.classes[argmaxOf(r)] ?? 0),
              [pr.length],
              'float64',
            )
          },
        }
      },
    },
    { varSmoothing: hyper.varSmoothing ?? 1e-9 },
  )
}

/** Binary logistic regression with an L2 penalty 1/(2C) on the coefficients (sklearn LogisticRegression), by Newton. */
export function logisticRegression(
  hyper: { readonly C?: number } = {},
): Estimator<{ readonly C: number }, Fitted<LinearParams & { readonly classes: number[] }>> {
  return defineEstimator(
    {
      name: 'logistic-regression',
      kind: 'classifier',
      fit: (data, h) => {
        const m = matrixView(data.x, 'logisticRegression')
        const { classes, codes } = encodeLabels(data.y ?? [], 'logisticRegression', m.n)
        if (classes.length !== 2) throw new Error('logisticRegression: binary only')
        const d = m.d + 1
        let w = new Float64Array(d)
        for (let iter = 0; iter < 100; iter++) {
          const g = new Float64Array(d)
          const hm = new Float64Array(d * d)
          for (let i = 0; i < m.n; i++) {
            let z = w[m.d] ?? 0
            for (let j = 0; j < m.d; j++) z += (w[j] ?? 0) * (m.data[i * m.d + j] ?? 0)
            const pr = 1 / (1 + Math.exp(-z))
            const r = pr - (codes[i] ?? 0)
            const s = pr * (1 - pr)
            for (let a = 0; a < d; a++) {
              const xa = a === m.d ? 1 : (m.data[i * m.d + a] ?? 0)
              g[a] = (g[a] ?? 0) + h.C * r * xa
              for (let b = 0; b < d; b++) {
                const xb = b === m.d ? 1 : (m.data[i * m.d + b] ?? 0)
                hm[a * d + b] = (hm[a * d + b] ?? 0) + h.C * s * xa * xb
              }
            }
          }
          for (let j = 0; j < m.d; j++) {
            g[j] = (g[j] ?? 0) + (w[j] ?? 0)
            hm[j * d + j] = (hm[j * d + j] ?? 0) + 1
          }
          const step = toFlat(solve(fromStorage(hm, [d, d], 'float64'), fromStorage(g, [d], 'float64')))
          w = Float64Array.from(w, (v, j) => v - (step[j] ?? 0))
          if (maxOf(step.map(Math.abs)) < 1e-13) break
        }
        return { coef: Array.from(w.subarray(0, m.d)), intercept: w[m.d] ?? 0, classes: [...classes] }
      },
      methods: (p) => {
        const lin = linearMethods(p, p.coef.length)
        const proba = (x: TensorLike): Tensor => {
          const z = toFlat(lin.predict(x))
          const out = new Float64Array(z.length * 2)
          z.forEach((v, i) => {
            const q = 1 / (1 + Math.exp(-v))
            out[2 * i] = 1 - q
            out[2 * i + 1] = q
          })
          return fromStorage(out, [z.length, 2], 'float64')
        }
        return {
          classes: p.classes,
          nFeaturesIn: p.coef.length,
          featureImportances: lin.featureImportances,
          decisionFunction: lin.predict,
          predictProba: proba,
          predict: (x) =>
            fromStorage(
              Float64Array.from(toFlat(lin.predict(x)), (v) => (v > 0 ? p.classes[1] : p.classes[0]) ?? 0),
              [rowsOfT(x)],
              'float64',
            ),
        }
      },
    },
    { C: hyper.C ?? 1 },
  )
}

function rowsOfT(x: TensorLike): number {
  return matrixView(x, 'rows').n
}
