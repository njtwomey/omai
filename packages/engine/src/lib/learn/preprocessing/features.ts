/**
 * Feature expansions and transforms, scikit-learn style: polynomial and interaction features, B-spline basis
 * expansions, binning of continuous features by width, quantiles or k-means, thresholding to 0 or 1, and
 * `functionTransformer`, which wraps any function so it composes in a pipeline.
 */

import { WhiteboxError } from '../../base/index.ts'
import { bspline } from '../../numerics/interpolate/index.ts'
import { kmeansCodebook } from '../../numerics/spatial/index.ts'
import { bool, choice, dict, int, type Space } from '../../space/index.ts'
import { quantile } from '../../stats/index.ts'
import { fromStorage, linspace, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { inputNames, present } from '../_shared.ts'
import { defineEstimator, type Estimator, type Fitted } from '../estimator.ts'
import { columnOf, fromView, matrixView } from '../validation.ts'

/** The verbs of a fitted feature transformer. */
export interface FeatureVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

/** Hyperparameters of {@link polynomialFeatures}. */
export interface PolynomialHyper {
  /** The largest total degree, or `[min, max]`. */
  readonly degree: number | readonly [number, number]
  /** Only products of distinct features (no powers). */
  readonly interactionOnly: boolean
  /** Include the constant column. */
  readonly includeBias: boolean
}

/** Parameters of fitted polynomial features: the exponent of each input in each output column. */
export type PolynomialParams = { readonly powers: number[][] }

/** Index combinations of size k of 0..n-1, with or without replacement, in itertools order. */
function combos(n: number, k: number, replacement: boolean): number[][] {
  const out: number[][] = []
  const rec = (start: number, acc: number[]): void => {
    if (acc.length === k) {
      out.push([...acc])
      return
    }
    for (let i = start; i < n; i++) {
      acc.push(i)
      rec(replacement ? i : i + 1, acc)
      acc.pop()
    }
  }
  rec(0, [])
  return out
}

/** The most output columns {@link polynomialFeatures} builds. */
export const MAX_POLYNOMIAL_FEATURES = 100000

/**
 * All monomials of the features up to a degree (sklearn.preprocessing.PolynomialFeatures), in sklearn's order: the
 * bias, then degree 1, 2, ... with each degree's combinations (with replacement unless `interactionOnly`) in
 * lexicographic order.
 *
 * @param hyper - degree (default 2), interactionOnly (default false) and includeBias (default true).
 * @returns The estimator.
 * @throws WhiteboxError with code `value` beyond {@link MAX_POLYNOMIAL_FEATURES} columns.
 * @concept feature-engineering
 * @example
 * ```ts
 * import { polynomialFeatures } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const p = polynomialFeatures().fit({ x: [[2, 3]] })
 * toArray(p.transform([[2, 3]])) // => [[1, 2, 3, 4, 6, 9]]
 * p.featureNamesOut() // => ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']
 * ```
 */
export function polynomialFeatures(
  hyper: Partial<PolynomialHyper> = {},
): Estimator<PolynomialHyper, Fitted<PolynomialParams> & FeatureVerbs> {
  return defineEstimator(
    {
      name: 'polynomial-features',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ degree: int(1, 4), interactionOnly: bool(), includeBias: bool() }),
      fit: (data, h) => {
        const d = matrixView(data.x, 'polynomialFeatures').d
        const [lo, hi] = typeof h.degree === 'number' ? [0, h.degree] : h.degree
        if (!(Number.isInteger(lo) && Number.isInteger(hi) && lo >= 0 && hi >= lo)) {
          throw new WhiteboxError('value', 'polynomialFeatures: degree must be integers 0 <= min <= max')
        }
        const powers: number[][] = []
        if (h.includeBias) powers.push(new Array<number>(d).fill(0))
        for (let k = Math.max(1, lo); k <= hi; k++) {
          for (const c of combos(d, k, !h.interactionOnly)) {
            const row = new Array<number>(d).fill(0)
            for (const i of c) row[i] = (row[i] ?? 0) + 1
            powers.push(row)
            if (powers.length > MAX_POLYNOMIAL_FEATURES) {
              throw new WhiteboxError('value', `polynomialFeatures: more than ${MAX_POLYNOMIAL_FEATURES} columns`)
            }
          }
        }
        return { powers }
      },
      methods: (p) => {
        const d = p.powers[0]?.length ?? 0
        return {
          nFeaturesIn: d,
          transform: (x) => {
            const m = matrixView(x, 'polynomialFeatures', { nFeatures: d, allowNaN: true })
            const k = p.powers.length
            const out = new Float64Array(m.n * k)
            for (let i = 0; i < m.n; i++) {
              p.powers.forEach((row, c) => {
                let v = 1
                for (let j = 0; j < d; j++) if ((row[j] ?? 0) > 0) v *= (m.data[i * d + j] ?? 0) ** (row[j] ?? 0)
                out[i * k + c] = v
              })
            }
            return fromStorage(out, [m.n, k], 'float64')
          },
          featureNamesOut: (input) => {
            const names = inputNames(d, input, 'polynomialFeatures')
            return p.powers.map((row) => {
              const terms = row.flatMap((e, j) => (e === 0 ? [] : [e === 1 ? (names[j] ?? '') : `${names[j]}^${e}`]))
              return terms.length === 0 ? '1' : terms.join(' ')
            })
          },
        }
      },
    },
    {
      degree: hyper.degree ?? 2,
      interactionOnly: hyper.interactionOnly ?? false,
      includeBias: hyper.includeBias ?? true,
    },
  )
}

/**
 * Pairwise products of features: every `x_i x_j` with i < j (or the given pairs), without the features themselves.
 *
 * @param hyper - Explicit pairs of feature indices (default all pairs).
 * @returns The estimator.
 * @example
 * ```ts
 * import { interactionFeatures } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(interactionFeatures().fit({ x: [[2, 3, 5]] }).transform([[2, 3, 5]])) // => [[6, 10, 15]]
 * interactionFeatures({ pairs: [[0, 2]] }).fit({ x: [[2, 3, 5]] }).featureNamesOut() // => ['x0 x2']
 * ```
 */
export function interactionFeatures(
  hyper: { readonly pairs?: readonly (readonly [number, number])[] } = {},
): Estimator<
  { readonly pairs?: readonly (readonly [number, number])[] },
  Fitted<{ readonly pairs: number[][]; readonly d: number }> & FeatureVerbs
> {
  return defineEstimator(
    {
      name: 'interaction-features',
      kind: 'transformer',
      fit: (data, h) => {
        const d = matrixView(data.x, 'interactionFeatures').d
        const pairs = h.pairs?.map(([a, b]) => [a, b]) ?? combos(d, 2, false)
        for (const [a = -1, b = -1] of pairs) {
          if (!(a >= 0 && a < d && b >= 0 && b < d))
            throw new WhiteboxError('value', `interactionFeatures: pair ${a}, ${b} out of range`)
        }
        return { pairs, d }
      },
      methods: (p) => ({
        nFeaturesIn: p.d,
        transform: (x) => {
          const m = matrixView(x, 'interactionFeatures', { nFeatures: p.d, allowNaN: true })
          const k = p.pairs.length
          const out = new Float64Array(m.n * k)
          for (let i = 0; i < m.n; i++) {
            p.pairs.forEach(([a = 0, b = 0], c) => {
              out[i * k + c] = (m.data[i * p.d + a] ?? 0) * (m.data[i * p.d + b] ?? 0)
            })
          }
          return fromStorage(out, [m.n, k], 'float64')
        },
        featureNamesOut: (input) => {
          const names = inputNames(p.d, input, 'interactionFeatures')
          return p.pairs.map(([a = 0, b = 0]) => `${names[a]} ${names[b]}`)
        },
      }),
    },
    hyper,
  )
}

/** How {@link splineFeatures} treats values beyond the knots. */
export type SplineExtrapolation = 'error' | 'constant' | 'linear' | 'continue' | 'periodic'

/** Hyperparameters of {@link splineFeatures}. */
export interface SplineHyper {
  /** The number of knots of the base interval (default 5). */
  readonly nKnots: number
  /** The polynomial degree (default 3). */
  readonly degree: number
  /** Knots evenly spaced between the data's extremes, or at its quantiles. */
  readonly knots: 'uniform' | 'quantile'
  readonly extrapolation: SplineExtrapolation
  /** Keep the last basis function of each feature (default true; dropping it removes the partition of unity). */
  readonly includeBias: boolean
}

/** Parameters of fitted spline features: the full knot vector of each feature. */
export type SplineParams = { readonly knots: number[][]; readonly nSplines: number }

/**
 * B-spline basis expansion of each feature (sklearn.preprocessing.SplineTransformer): base knots uniform or at
 * quantiles, `degree` extra knots beyond each end at the end spacing (or wrapped for periodic splines), and
 * extrapolation by error, constants, linear continuation, the polynomial pieces, or periodicity.
 *
 * @param hyper - nKnots, degree, knots, extrapolation and includeBias.
 * @returns The estimator.
 * @example
 * ```ts
 * import { splineFeatures } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const s = splineFeatures({ nKnots: 3, degree: 1 }).fit({ x: [[0], [2]] })
 * toArray(s.transform([[0.5]])) // => [[0.5, 0.5, 0]]
 * s.featureNamesOut() // => ['x0_sp_0', 'x0_sp_1', 'x0_sp_2']
 * ```
 */
export function splineFeatures(
  hyper: Partial<SplineHyper> = {},
): Estimator<SplineHyper, Fitted<SplineParams> & FeatureVerbs> {
  return defineEstimator(
    {
      name: 'spline-features',
      kind: 'transformer',
      hyperSpace: (): Space =>
        dict({ nKnots: int(3, 20), degree: int(1, 5), knots: choice(['uniform', 'quantile']), includeBias: bool() }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'splineFeatures', { minSamples: 2 })
        if (h.nKnots < 2) throw new WhiteboxError('value', 'splineFeatures: nKnots must be at least 2')
        if (h.extrapolation === 'periodic' && h.nKnots <= h.degree) {
          throw new WhiteboxError('value', 'splineFeatures: periodic splines need degree < nKnots')
        }
        const k = h.degree
        const knots: number[][] = []
        for (let j = 0; j < m.d; j++) {
          const col = present(columnOf(m, j))
          const base =
            h.knots === 'quantile'
              ? toFlat(quantile(col, toFlat(linspace(0, 1, h.nKnots))))
              : toFlat(
                  linspace(
                    col.reduce((a, b) => Math.min(a, b)),
                    col.reduce((a, b) => Math.max(a, b)),
                    h.nKnots,
                  ),
                )
          for (let i = 1; i < base.length; i++) {
            if (!((base[i] ?? 0) > (base[i - 1] ?? 0)))
              throw new WhiteboxError('value', 'splineFeatures: knots must be distinct')
          }
          const nb = base.length
          if (h.extrapolation === 'periodic') {
            const period = (base[nb - 1] ?? 0) - (base[0] ?? 0)
            knots.push([
              ...base.slice(nb - 1 - k, nb - 1).map((v) => v - period),
              ...base,
              ...base.slice(1, k + 1).map((v) => v + period),
            ])
          } else {
            const dMin = (base[1] ?? 0) - (base[0] ?? 0)
            const dMax = (base[nb - 1] ?? 0) - (base[nb - 2] ?? 0)
            const before = k === 0 ? [] : toFlat(linspace((base[0] ?? 0) - k * dMin, (base[0] ?? 0) - dMin, k))
            const after = k === 0 ? [] : toFlat(linspace((base[nb - 1] ?? 0) + dMax, (base[nb - 1] ?? 0) + k * dMax, k))
            knots.push([...before, ...base, ...after])
          }
        }
        return { knots, nSplines: h.extrapolation === 'periodic' ? h.nKnots - 1 : h.nKnots + h.degree - 1 }
      },
      methods: (p, h) => {
        const d = p.knots.length
        const k = h.degree
        const ns = p.nSplines
        const width = d * (ns - (h.includeBias ? 0 : 1))
        // One spline per feature whose coefficient matrix is the identity (periodic: wrapped onto the first columns).
        const splines = p.knots.map((t) => {
          const nCoef = t.length - k - 1
          // Row r of the coefficients selects basis column r mod ns: the identity, wrapped for periodic splines.
          const coef = Float64Array.from({ length: nCoef * ns }, (_, i) => (i % ns === Math.floor(i / ns) % ns ? 1 : 0))
          const extrapolate = h.extrapolation === 'periodic' || h.extrapolation === 'continue'
          return bspline(t, fromStorage(coef, [nCoef, ns], 'float64'), k, extrapolate)
        })
        const basisAt = (j: number, values: readonly number[], nu = 0): number[][] => {
          if (values.length === 0) return []
          const r = toFlat(splines[j]?.evaluate(values, nu) as Tensor)
          return values.map((_, i) => r.slice(i * ns, (i + 1) * ns))
        }
        return {
          nFeaturesIn: d,
          transform: (x) => {
            const m = matrixView(x, 'splineFeatures', { nFeatures: d })
            const full = new Float64Array(m.n * d * ns)
            for (let j = 0; j < d; j++) {
              const t = p.knots[j] as number[]
              const xmin = t[k] ?? 0
              const xmax = t[t.length - k - 1] ?? 0
              const col = Array.from(columnOf(m, j))
              let values = col
              if (h.extrapolation === 'periodic') {
                const period = xmax - xmin
                values = col.map((v) => (period > 0 ? xmin + ((((v - xmin) % period) + period) % period) : 0))
              }
              const inside = col.map((v) => v >= xmin && v <= xmax)
              const evalAll = h.extrapolation === 'continue' || h.extrapolation === 'periodic'
              const idx = col.flatMap((_, i) => (evalAll || inside[i] === true ? [i] : []))
              if (h.extrapolation === 'error' && idx.length < col.length) {
                throw new WhiteboxError('value', `splineFeatures: values beyond the knots in feature ${j}`)
              }
              const b = basisAt(
                j,
                idx.map((i) => values[i] ?? 0),
              )
              idx.forEach((i, r) => {
                full.set(b[r] as number[], (i * d + j) * ns)
              })
              if (h.extrapolation === 'constant' || h.extrapolation === 'linear') {
                const [fMin = [], fMax = []] = basisAt(j, [xmin, xmax])
                const [dMin = [], dMax = []] = h.extrapolation === 'linear' ? basisAt(j, [xmin, xmax], 1) : [[], []]
                const kk = h.extrapolation === 'linear' && k <= 1 ? k + 1 : k
                col.forEach((v, i) => {
                  if (v < xmin) {
                    for (let s = 0; s < kk; s++) {
                      full[(i * d + j) * ns + s] =
                        h.extrapolation === 'linear' ? (fMin[s] ?? 0) + (v - xmin) * (dMin[s] ?? 0) : (fMin[s] ?? 0)
                    }
                  } else if (v > xmax) {
                    for (let s = 0; s < kk; s++) {
                      const c = ns - 1 - s
                      full[(i * d + j) * ns + c] =
                        h.extrapolation === 'linear' ? (fMax[c] ?? 0) + (v - xmax) * (dMax[c] ?? 0) : (fMax[c] ?? 0)
                    }
                  }
                })
              }
            }
            if (h.includeBias) return fromStorage(full, [m.n, d * ns], 'float64')
            const out = new Float64Array(m.n * width)
            let c = 0
            for (let i = 0; i < m.n; i++) {
              for (let col = 0; col < d * ns; col++) if ((col + 1) % ns !== 0) out[c++] = full[i * d * ns + col] ?? 0
            }
            return fromStorage(out, [m.n, width], 'float64')
          },
          featureNamesOut: (input) => {
            const names = inputNames(d, input, 'splineFeatures')
            return names.flatMap((nm) =>
              Array.from({ length: ns - (h.includeBias ? 0 : 1) }, (_, s) => `${nm}_sp_${s}`),
            )
          },
        }
      },
    },
    {
      nKnots: hyper.nKnots ?? 5,
      degree: hyper.degree ?? 3,
      knots: hyper.knots ?? 'uniform',
      extrapolation: hyper.extrapolation ?? 'constant',
      includeBias: hyper.includeBias ?? true,
    },
  )
}

/** Hyperparameters of {@link kBinsDiscretizer}. */
export interface KBinsHyper {
  /** Bins per feature (default 5). */
  readonly nBins: number | readonly number[]
  /** Bin codes (`'ordinal'`) or one indicator column per bin (`'onehot'`). */
  readonly encode: 'ordinal' | 'onehot'
  /** Equal widths, equal counts, or 1-D k-means centres (default `'quantile'`). */
  readonly strategy: 'uniform' | 'quantile' | 'kmeans'
  /** The quantile definition for `'quantile'` (default `'averaged_inverted_cdf'`, sklearn's). */
  readonly quantileMethod: 'linear' | 'inverted_cdf' | 'averaged_inverted_cdf'
}

/** Parameters of a fitted discretiser: bin edges per feature. */
export type KBinsParams = { readonly edges: number[][] }

/**
 * Bin continuous features (sklearn.preprocessing.KBinsDiscretizer): edges by equal width, quantiles or 1-D k-means
 * (Lloyd from uniform centres), bins narrower than 1e-8 merged, values coded by `searchsorted` on the inner edges, and
 * the inverse mapping a code to its bin's centre. A constant feature gets one bin.
 *
 * @param hyper - nBins, encode, strategy and quantileMethod.
 * @returns The estimator.
 * @example
 * ```ts
 * import { kBinsDiscretizer } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const b = kBinsDiscretizer({ nBins: 2, strategy: 'uniform' }).fit({ x: [[0], [1], [4]] })
 * toArray(b.transform([[1], [3]])) // => [[0], [1]]
 * toArray(b.inverseTransform([[1]])) // => [[3]]
 * ```
 */
export function kBinsDiscretizer(
  hyper: Partial<KBinsHyper> = {},
): Estimator<
  KBinsHyper,
  Fitted<KBinsParams> & FeatureVerbs & { readonly inverseTransform: (x: TensorLike) => Tensor }
> {
  return defineEstimator(
    {
      name: 'k-bins-discretizer',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ nBins: int(2, 20), strategy: choice(['uniform', 'quantile', 'kmeans']) }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'kBinsDiscretizer')
        const edges: number[][] = []
        for (let j = 0; j < m.d; j++) {
          const col = columnOf(m, j)
          const nb = typeof h.nBins === 'number' ? h.nBins : (h.nBins[j] ?? 5)
          if (!(Number.isInteger(nb) && nb >= 2))
            throw new WhiteboxError('value', 'kBinsDiscretizer: nBins must be integers >= 2')
          const lo = col.reduce((a, b) => Math.min(a, b))
          const hi = col.reduce((a, b) => Math.max(a, b))
          if (lo === hi) {
            edges.push([Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY])
            continue
          }
          let e: number[]
          if (h.strategy === 'uniform') e = toFlat(linspace(lo, hi, nb + 1))
          else if (h.strategy === 'quantile')
            e = toFlat(quantile(col, toFlat(linspace(0, 1, nb + 1)), { method: h.quantileMethod }))
          else {
            const u = toFlat(linspace(lo, hi, nb + 1))
            const init = u.slice(1).map((v, i) => [(v + (u[i] ?? 0)) * 0.5])
            const cb = kmeansCodebook(fromStorage(col, [col.length, 1], 'float64'), nb, {
              init,
              tol: 0,
              maxIterations: 300,
            })
            const centres = toFlat(cb.centroids).sort((a, b) => a - b)
            e = [lo, ...centres.slice(1).map((c, i) => (c + (centres[i] ?? 0)) * 0.5), hi]
          }
          if (h.strategy !== 'uniform') e = e.filter((v, i) => i === 0 || v - (e[i - 1] ?? 0) > 1e-8)
          edges.push(e)
        }
        return { edges }
      },
      methods: (p, h) => {
        const d = p.edges.length
        const nBins = p.edges.map((e) => e.length - 1)
        const width = h.encode === 'onehot' ? nBins.reduce((a, b) => a + b, 0) : d
        const code = (j: number, v: number): number => {
          const e = p.edges[j] as number[]
          let c = 0
          for (let i = 1; i < e.length - 1; i++) if ((e[i] ?? 0) <= v) c = i
          return c
        }
        return {
          nFeaturesIn: d,
          transform: (x) => {
            const m = matrixView(x, 'kBinsDiscretizer', { nFeatures: d })
            const out = new Float64Array(m.n * width)
            for (let i = 0; i < m.n; i++) {
              let offset = 0
              for (let j = 0; j < d; j++) {
                const c = code(j, m.data[i * d + j] ?? 0)
                if (h.encode === 'onehot') out[i * width + offset + c] = 1
                else out[i * d + j] = c
                offset += nBins[j] ?? 0
              }
            }
            return fromStorage(out, [m.n, width], 'float64')
          },
          inverseTransform: (x) => {
            const m = matrixView(x, 'kBinsDiscretizer', { nFeatures: width })
            const out = new Float64Array(m.n * d)
            for (let i = 0; i < m.n; i++) {
              let offset = 0
              for (let j = 0; j < d; j++) {
                let c = 0
                if (h.encode === 'onehot') {
                  for (let b = 1; b < (nBins[j] ?? 0); b++)
                    if ((m.data[i * width + offset + b] ?? 0) > (m.data[i * width + offset + c] ?? 0)) c = b
                } else c = Math.min(Math.max(Math.round(m.data[i * d + j] ?? 0), 0), (nBins[j] ?? 1) - 1)
                const e = p.edges[j] as number[]
                out[i * d + j] = ((e[c] ?? 0) + (e[c + 1] ?? 0)) / 2
                offset += nBins[j] ?? 0
              }
            }
            return fromStorage(out, [m.n, d], 'float64')
          },
          featureNamesOut: (input) => {
            const names = inputNames(d, input, 'kBinsDiscretizer')
            return h.encode === 'onehot'
              ? names.flatMap((nm, j) => Array.from({ length: nBins[j] ?? 0 }, (_, b) => `${nm}_${b}`))
              : names
          },
        }
      },
    },
    {
      nBins: hyper.nBins ?? 5,
      encode: hyper.encode ?? 'ordinal',
      strategy: hyper.strategy ?? 'quantile',
      quantileMethod: hyper.quantileMethod ?? 'averaged_inverted_cdf',
    },
  )
}

/**
 * Threshold features to 0 or 1 (sklearn.preprocessing.Binarizer): 1 where x > threshold.
 *
 * @param hyper - The threshold (default 0).
 * @returns The estimator.
 * @example
 * ```ts
 * import { binarizer } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(binarizer({ threshold: 1 }).fit({ x: [[0]] }).transform([[0.5, 1, 2]])) // => [[0, 0, 1]]
 * ```
 */
export function binarizer(
  hyper: { readonly threshold?: number } = {},
): Estimator<{ readonly threshold: number }, Fitted<{ readonly d: number }> & FeatureVerbs> {
  return defineEstimator(
    {
      name: 'binarizer',
      kind: 'transformer',
      fit: (data) => ({ d: matrixView(data.x, 'binarizer').d }),
      methods: (p, h) => ({
        nFeaturesIn: p.d,
        transform: (x) => {
          const m = matrixView(x, 'binarizer')
          return fromView({ ...m, data: Float64Array.from(m.data, (v) => (v > h.threshold ? 1 : 0)) })
        },
        featureNamesOut: (input) => inputNames(p.d, input, 'binarizer'),
      }),
    },
    { threshold: hyper.threshold ?? 0 },
  )
}

/** Hyperparameters of {@link functionTransformer}. */
export interface FunctionTransformerHyper {
  /** The forward map of a feature matrix (default the identity). */
  readonly func?: (x: Tensor) => TensorLike
  /** Its inverse. */
  readonly inverseFunc?: (x: Tensor) => TensorLike
  /** Check on the fit data that inverse(func(x)) recovers x (default true when both are given). */
  readonly checkInverse?: boolean
  /** Names of the output features, given the input names. */
  readonly featureNamesOut?: (input: readonly string[]) => string[]
}

/**
 * Wrap any function of a feature matrix as a transformer (sklearn.preprocessing.FunctionTransformer), so it composes
 * in pipelines; with an inverse, `fit` checks the round trip on the training data.
 *
 * @param hyper - func, inverseFunc, checkInverse and featureNamesOut.
 * @returns The estimator.
 * @throws WhiteboxError with code `value` in fit when the inverse does not invert.
 * @example
 * ```ts
 * import { functionTransformer } from 'whitebox-engine/learn/preprocessing'
 * import { exp, log, toArray } from 'whitebox-engine/tensor'
 * const f = functionTransformer({ func: log, inverseFunc: exp }).fit({ x: [[1], [Math.E]] })
 * toArray(f.transform([[Math.E]])) // => [[1]]
 * ```
 */
export function functionTransformer(
  hyper: FunctionTransformerHyper = {},
): Estimator<
  FunctionTransformerHyper,
  Fitted<{ readonly d: number }> & FeatureVerbs & { readonly inverseTransform: (x: TensorLike) => Tensor }
> {
  const apply = (f: ((x: Tensor) => TensorLike) | undefined, x: TensorLike, who: string): Tensor => {
    const t = fromView(matrixView(x, who, { allowNaN: true, allowInfinite: true }))
    return f === undefined ? t : fromView(matrixView(f(t), who, { allowNaN: true, allowInfinite: true }))
  }
  return defineEstimator(
    {
      name: 'function-transformer',
      kind: 'transformer',
      fit: (data, h) => {
        const m = matrixView(data.x, 'functionTransformer', { allowNaN: true, allowInfinite: true })
        if (h.func !== undefined && h.inverseFunc !== undefined && h.checkInverse !== false) {
          const back = matrixView(
            apply(h.inverseFunc, apply(h.func, fromView(m), 'functionTransformer'), 'functionTransformer'),
            'functionTransformer',
            { allowNaN: true, allowInfinite: true },
          )
          for (let i = 0; i < m.data.length; i++) {
            const a = m.data[i] ?? 0
            const b = back.data[i] ?? 0
            if (Number.isFinite(a) && Math.abs(a - b) > 1e-7 * (1 + Math.abs(a))) {
              throw new WhiteboxError('value', 'functionTransformer: inverseFunc does not invert func on the fit data')
            }
          }
        }
        return { d: m.d }
      },
      methods: (p, h) => ({
        nFeaturesIn: p.d,
        transform: (x) => apply(h.func, x, 'functionTransformer'),
        inverseTransform: (x) => apply(h.inverseFunc, x, 'functionTransformer'),
        featureNamesOut: (input) => {
          const names = inputNames(p.d, input, 'functionTransformer')
          return h.featureNamesOut === undefined ? names : h.featureNamesOut(names)
        },
      }),
    },
    hyper,
  )
}
