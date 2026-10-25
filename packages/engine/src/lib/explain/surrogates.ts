/**
 * Surrogate explanations: LIME's locally weighted linear models (with its feature selection), anchors (high-precision
 * rules found by a bandit search), and global surrogate trees with their fidelity to the model.
 */
import { WhiteboxError } from '../base/index.ts'
import { keyFor, runOptionsFor } from '../learn/_shared.ts'
import type { FitOptions } from '../learn/estimator.ts'
import { columnOf, type MatrixView, matrixView, takeColumns, targetVector } from '../learn/validation.ts'
import { pinv } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal, permutation, randint } from '../random/index.ts'
import { asTensor, fromStorage, matmul, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import type { PredictFunction, TreeModel } from './shapley.ts'

/** A weighted ridge fit with an unpenalised intercept (sklearn Ridge with sample weights). */
interface RidgeFit {
  readonly coef: number[]
  readonly intercept: number
  /** The weighted R^2 on the fitting data. */
  readonly score: number
}

function weightedRidge(m: MatrixView, y: Float64Array, w: Float64Array, alpha: number): RidgeFit {
  const sw = w.reduce((a, b) => a + b, 0)
  const xm = Array.from({ length: m.d }, (_, j) => {
    let s = 0
    for (let i = 0; i < m.n; i++) s += (w[i] ?? 0) * (m.data[i * m.d + j] ?? 0)
    return s / sw
  })
  let ym = 0
  for (let i = 0; i < m.n; i++) ym += ((w[i] ?? 0) * (y[i] ?? 0)) / sw
  const g = new Float64Array(m.d * m.d)
  const b = new Float64Array(m.d)
  for (let i = 0; i < m.n; i++) {
    const wi = w[i] ?? 0
    for (let a = 0; a < m.d; a++) {
      const xa = (m.data[i * m.d + a] ?? 0) - (xm[a] ?? 0)
      b[a] = (b[a] ?? 0) + wi * xa * ((y[i] ?? 0) - ym)
      for (let c = 0; c < m.d; c++)
        g[a * m.d + c] = (g[a * m.d + c] ?? 0) + wi * xa * ((m.data[i * m.d + c] ?? 0) - (xm[c] ?? 0))
    }
  }
  for (let a = 0; a < m.d; a++) g[a * m.d + a] = (g[a * m.d + a] ?? 0) + alpha
  const coef =
    m.d === 0 ? [] : toFlat(matmul(pinv(fromStorage(g, [m.d, m.d], 'float64')), fromStorage(b, [m.d], 'float64')))
  const intercept = ym - coef.reduce((s, c, j) => s + c * (xm[j] ?? 0), 0)
  let ssr = 0
  let sst = 0
  for (let i = 0; i < m.n; i++) {
    let p = intercept
    for (let j = 0; j < m.d; j++) p += (coef[j] ?? 0) * (m.data[i * m.d + j] ?? 0)
    ssr += (w[i] ?? 0) * ((y[i] ?? 0) - p) ** 2
    sst += (w[i] ?? 0) * ((y[i] ?? 0) - ym) ** 2
  }
  return { coef, intercept, score: sst === 0 ? 1 : 1 - ssr / sst }
}

/** How LIME picks the features of its explanation. */
export type LimeFeatureSelection = 'auto' | 'none' | 'forward_selection' | 'highest_weights'

/** Options of {@link limeFit}. */
export interface LimeFitOptions {
  /** The width of the exponential kernel sqrt(exp(-d^2 / width^2)) (default sqrt(d) 0.75, LIME tabular's). */
  readonly kernelWidth?: number
  /** Features kept in the explanation (default 10). */
  readonly nFeatures?: number
  readonly featureSelection?: LimeFeatureSelection
  /** The ridge penalty of the explanation model (default 1). */
  readonly alpha?: number
}

/** A LIME explanation: the local linear model and its fit. */
export interface LimeExplanation {
  readonly intercept: number
  /** `[feature, weight]` pairs, by decreasing |weight|. */
  readonly weights: readonly (readonly [number, number])[]
  /** The weighted R^2 of the local model on the neighbourhood. */
  readonly score: number
  /** The local model's prediction for the explained point (the first row). */
  readonly localPrediction: number
}

function forwardSelection(m: MatrixView, y: Float64Array, w: Float64Array, k: number): number[] {
  const used: number[] = []
  for (let r = 0; r < Math.min(k, m.d); r++) {
    let best = 0
    let bestScore = -100000000
    for (let j = 0; j < m.d; j++) {
      if (used.includes(j)) continue
      const s = weightedRidge(takeColumns(m, [...used, j]), y, w, 0).score
      if (s > bestScore) {
        bestScore = s
        best = j
      }
    }
    used.push(best)
  }
  return used
}

/**
 * Fit LIME's local explanation to a given neighbourhood (Ribeiro, Singh and Guestrin 2016; lime's
 * LimeBase.explain_instance_with_data): weight the perturbed points by an exponential kernel of their distance to the
 * explained point (the first row), select features (forward selection, or the largest weighted coefficients) and fit
 * a weighted ridge regression to the model's outputs.
 *
 * @param neighbourhood - Perturbed points `[m, d]` (row 0 is the explained point), in the explanation's space.
 * @param targets - The model's outputs at them `[m]`.
 * @param distances - Their distances to the explained point `[m]`.
 * @param options - kernelWidth, nFeatures, featureSelection and alpha.
 * @returns The explanation.
 * @concept local-surrogates
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * const z = [[0, 0], [1, 0], [0, 1], [1, 1]]
 * const r = explain.limeFit(z, [0, 2, 1, 3], [0, 1, 1, Math.SQRT2], { alpha: 0, featureSelection: 'none' })
 * r.weights.map(([j, w]) => [j, Math.round(w * 1e9) / 1e9]) // => [[0, 2], [1, 1]]
 * ```
 */
export function limeFit(
  neighbourhood: TensorLike,
  targets: TensorLike,
  distances: TensorLike,
  options: LimeFitOptions = {},
): LimeExplanation {
  const m = matrixView(neighbourhood, 'limeFit')
  const y = targetVector(targets, 'limeFit', m.n)
  const dist = targetVector(distances, 'limeFit', m.n)
  const width = options.kernelWidth ?? Math.sqrt(m.d) * 0.75
  const w = dist.map((d) => Math.sqrt(Math.exp(-(d * d) / (width * width))))
  const k = options.nFeatures ?? 10
  const method = options.featureSelection ?? 'auto'
  const chosen = method === 'auto' ? (k <= 6 ? 'forward_selection' : 'highest_weights') : method
  let used: number[]
  if (chosen === 'none') used = Array.from({ length: m.d }, (_, j) => j)
  else if (chosen === 'forward_selection') used = forwardSelection(m, y, w, k)
  else {
    const fit = weightedRidge(m, y, w, 0.01)
    const weighted = fit.coef.map((c, j) => ({ j, v: c * (m.data[j] ?? 0) }))
    // Python's stable sort with reverse=True keeps ties in their original order.
    used = weighted
      .map((e, i) => ({ ...e, i }))
      .sort((a, b) => Math.abs(b.v) - Math.abs(a.v) || a.i - b.i)
      .slice(0, k)
      .map((e) => e.j)
  }
  const sub = takeColumns(m, used)
  const fit = weightedRidge(sub, y, w, options.alpha ?? 1)
  let local = fit.intercept
  used.forEach((_, c) => {
    local += (fit.coef[c] ?? 0) * (sub.data[c] ?? 0)
  })
  const pairs = used.map((j, c) => [j, fit.coef[c] ?? 0] as const)
  const weights = pairs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => Math.abs(b.p[1]) - Math.abs(a.p[1]) || a.i - b.i)
    .map((e) => e.p)
  return { intercept: fit.intercept, weights, score: fit.score, localPrediction: local }
}

/** Options of {@link lime}. */
export interface LimeOptions extends LimeFitOptions {
  /** Perturbations drawn (default 5000). */
  readonly nSamples?: number
  /** Sample around the explained point (default false: around the training mean, as lime's default). */
  readonly sampleAroundInstance?: boolean
  /** The model output to explain (a column of `[m, k]` outputs; default 0). */
  readonly label?: number
  readonly key?: Key
}

/**
 * LIME for tabular data with continuous features (lime's LimeTabularExplainer with discretize_continuous=False):
 * draw Gaussian perturbations with the training features' scale (around their mean or around the point), standardise
 * them, weight by distance to the point and fit the local linear model with {@link limeFit}.
 *
 * @param predict - The model (its outputs `[m]` or `[m, k]`).
 * @param x - The point to explain `[d]`.
 * @param training - Training rows `[n, d]` (their mean and standard deviation set the sampling scale).
 * @param options - nSamples, sampleAroundInstance, label, key and the {@link limeFit} options.
 * @returns The explanation (weights refer to standardised features).
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * const linear = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => 3 * x.data[2 * i]))
 * const train = [[0, 0], [1, 5], [2, 1], [3, 7]]
 * explain.lime(linear, [1, 1], train, { nSamples: 200, nFeatures: 1 }).weights[0][0] // => 0
 * ```
 */
export function lime(
  predict: PredictFunction,
  x: TensorLike,
  training: TensorLike,
  options: LimeOptions = {},
): LimeExplanation {
  const xv = toFlat(asTensor(x))
  const tr = matrixView(training, 'lime', { nFeatures: xv.length })
  const d = xv.length
  const mean = Array.from({ length: d }, (_, j) => columnOf(tr, j).reduce((a, b) => a + b, 0) / tr.n)
  const scale = Array.from({ length: d }, (_, j) => {
    const c = columnOf(tr, j)
    const s = Math.sqrt(c.reduce((a, b) => a + (b - (mean[j] ?? 0)) ** 2, 0) / tr.n)
    return s === 0 ? 1 : s
  })
  const ns = options.nSamples ?? 5000
  const z = toFlat(normal(options.key ?? keyFor({}, 'lime'), [ns, d]))
  const centre = options.sampleAroundInstance === true ? xv : mean
  const data = new Float64Array(ns * d)
  for (let i = 0; i < ns; i++) {
    for (let j = 0; j < d; j++)
      data[i * d + j] = i === 0 ? (xv[j] ?? 0) : (z[i * d + j] ?? 0) * (scale[j] ?? 1) + (centre[j] ?? 0)
  }
  const out = predict(fromStorage(data, [ns, d], 'float64'))
  const k = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
  const ov = toFlat(out)
  const label = options.label ?? 0
  const scaled = Float64Array.from(data, (v, i) => (v - (mean[i % d] ?? 0)) / (scale[i % d] ?? 1))
  const distances = Float64Array.from({ length: ns }, (_, i) => {
    let s = 0
    for (let j = 0; j < d; j++) s += ((scaled[i * d + j] ?? 0) - (scaled[j] ?? 0)) ** 2
    return Math.sqrt(s)
  })
  return limeFit(
    fromStorage(scaled, [ns, d], 'float64'),
    Float64Array.from({ length: ns }, (_, i) => ov[i * k + label] ?? 0),
    distances,
    options,
  )
}

/** Options of {@link limeMasked}. */
export interface LimeMaskedOptions extends LimeFitOptions {
  /** Perturbations drawn (default 1000). */
  readonly nSamples?: number
  /** The kernel width on the scaled cosine distance (default 25, lime text's; lime image uses 0.25 on the raw one). */
  readonly kernelWidth?: number
  /** The factor on the cosine distance (default 100, lime text's; 1 for lime image's convention). */
  readonly distanceScale?: number
  readonly key?: Key
}

/**
 * LIME over interpretable on/off features (lime's LimeTextExplainer and LimeImageExplainer): the explained instance is
 * the all-ones row over its tokens, superpixels or windows; each perturbation switches off a random subset of a
 * uniformly drawn size, the model scores it through a predictor over groups (`tokenPredictor`, `maskedPredictor`),
 * points are weighted by their cosine distance to the instance, and a weighted ridge regression on the binary rows
 * gives each unit's weight.
 *
 * @param predict - A predictor over `[m, nFeatures]` rows of 0/1 presence.
 * @param nFeatures - The interpretable features (tokens, superpixels, windows).
 * @param options - nSamples, kernelWidth, distanceScale, key and the {@link limeFit} options.
 * @returns The explanation (weights per interpretable feature).
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * // Only the second unit matters.
 * const f = (z) => tensor(Array.from({ length: z.shape[0] }, (_, i) => 2 * z.data[3 * i + 1]))
 * explain.limeMasked(f, 3, { nSamples: 200, nFeatures: 1 }).weights[0][0] // => 1
 * ```
 */
export function limeMasked(
  predict: PredictFunction,
  nFeatures: number,
  options: LimeMaskedOptions = {},
): LimeExplanation {
  const d = nFeatures
  if (!(Number.isInteger(d) && d >= 1)) throw new WhiteboxError('value', 'limeMasked: needs at least one feature')
  const ns = options.nSamples ?? 1000
  const key = options.key ?? keyFor({}, 'limeMasked')
  const data = new Float64Array(ns * d).fill(1)
  for (let i = 1; i < ns; i++) {
    const k = foldIn(key, i)
    const size = d === 1 ? 1 : 1 + (toFlat(randint(foldIn(k, 0), [1], 0, d - 1))[0] ?? 0)
    const order = toFlat(permutation(foldIn(k, 1), d))
    for (let t = 0; t < size; t++) data[i * d + (order[t] ?? 0)] = 0
  }
  const out = predict(fromStorage(data, [ns, d], 'float64'))
  const kOut = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
  const ov = toFlat(out)
  const scale = options.distanceScale ?? 100
  const distances = Float64Array.from({ length: ns }, (_, i) => {
    let on = 0
    for (let j = 0; j < d; j++) on += data[i * d + j] ?? 0
    // Cosine distance to the all-ones row: 1 - on / sqrt(on * d).
    return on === 0 ? scale : scale * (1 - on / Math.sqrt(on * d))
  })
  return limeFit(
    fromStorage(data, [ns, d], 'float64'),
    Float64Array.from({ length: ns }, (_, i) => ov[i * kOut] ?? 0),
    distances,
    { kernelWidth: 25, ...options },
  )
}

/** Options of {@link anchors}. */
export interface AnchorOptions extends FitOptions {
  /** The precision an anchor must reach (default 0.95). */
  readonly threshold?: number
  /** Candidates kept per rule length (default 2). */
  readonly beamSize?: number
  /** The confidence parameter of the bandit bounds (default 0.05) and its tolerance (default 0.1). */
  readonly delta?: number
  readonly epsilon?: number
  /** Samples per bandit pull (default 32) and the most samples per candidate (default 2048). */
  readonly batchSize?: number
  readonly maxSamples?: number
  /** Quantile bins per feature (default 4). */
  readonly bins?: number
  /** The longest anchor (default d). */
  readonly maxLength?: number
}

/** An anchor: a conjunction of feature-in-interval predicates that (almost) fixes the model's prediction. */
export interface Anchor {
  /** The predicates: feature j lies in [lower, upper]. */
  readonly predicates: readonly { readonly feature: number; readonly lower: number; readonly upper: number }[]
  /** The estimated precision (agreement with the model's prediction at x) and its lower confidence bound. */
  readonly precision: number
  readonly precisionLower: number
  /** The fraction of training rows the anchor covers. */
  readonly coverage: number
  /** Samples drawn in total. */
  readonly samples: number
}

function klBernoulli(p: number, q: number): number {
  const a = Math.min(Math.max(p, 1e-12), 1 - 1e-12)
  const b = Math.min(Math.max(q, 1e-12), 1 - 1e-12)
  return a * Math.log(a / b) + (1 - a) * Math.log((1 - a) / (1 - b))
}

function klBound(p: number, level: number, upper: boolean): number {
  let lo = upper ? p : 0
  let hi = upper ? 1 : p
  for (let it = 0; it < 40; it++) {
    const mid = (lo + hi) / 2
    if (klBernoulli(p, mid) > level === upper) hi = mid
    else lo = mid
  }
  return upper ? hi : lo
}

/**
 * Anchors (Ribeiro, Singh and Guestrin 2018): a high-precision rule around a point, built by beam search over
 * conjunctions of quantile-bin predicates containing the point, with KL-LUCB bandit sampling to tell candidates'
 * precisions apart; among rules whose lower confidence bound reaches the threshold, the one with the widest coverage
 * of the training data wins. Samples come from training rows, with the anchored features redrawn from rows in the
 * same bins. A traced algorithm with one step per bandit round.
 *
 * @param predict - The model (labels or scores; outputs are compared for equality).
 * @param x - The point `[d]`.
 * @param training - Training rows `[n, d]`.
 * @param options - threshold, beamSize, delta, epsilon, batchSize, maxSamples, bins, maxLength, key and tracing.
 * @returns The anchor.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * const firstPositive = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => (x.data[2 * i] > 0 ? 1 : 0)))
 * const train = Array.from({ length: 40 }, (_, i) => [((i * 7) % 20) - 10, (i * 3) % 5])
 * explain.anchors(firstPositive, [8, 1], train).predicates.map((p) => p.feature) // => [0]
 * ```
 */
export function anchors(
  predict: PredictFunction,
  x: TensorLike,
  training: TensorLike,
  options: AnchorOptions = {},
): Anchor {
  const xv = toFlat(asTensor(x))
  const tr = matrixView(training, 'anchors', { nFeatures: xv.length })
  const d = xv.length
  const nBins = options.bins ?? 4
  const tau = options.threshold ?? 0.95
  const beam = options.beamSize ?? 2
  const delta = options.delta ?? 0.05
  const eps = options.epsilon ?? 0.1
  const batch = options.batchSize ?? 32
  const maxSamples = options.maxSamples ?? 2048
  const maxLength = Math.min(options.maxLength ?? d, d)
  const target = toFlat(predict(fromStorage(Float64Array.from(xv), [1, d], 'float64')))[0] ?? 0
  // The bin of x on each feature: [lower, upper] from training quantiles.
  const bounds = Array.from({ length: d }, (_, j) => {
    const s = Float64Array.from(columnOf(tr, j)).sort()
    const q = (p: number): number => {
      const h = (s.length - 1) * p
      const lo = Math.floor(h)
      return (s[lo] ?? 0) + (h - lo) * ((s[Math.min(lo + 1, s.length - 1)] ?? 0) - (s[lo] ?? 0))
    }
    const edges = Array.from({ length: nBins + 1 }, (_, b) => q(b / nBins))
    let b = 0
    while (b < nBins - 1 && (xv[j] ?? 0) > (edges[b + 1] ?? 0)) b++
    return {
      lower: b === 0 ? Number.NEGATIVE_INFINITY : (edges[b] ?? 0),
      upper: b === nBins - 1 ? Number.POSITIVE_INFINITY : (edges[b + 1] ?? 0),
    }
  })
  const inBin = (j: number, v: number): boolean => {
    const bnd = bounds[j] as { lower: number; upper: number }
    return v > bnd.lower && v <= bnd.upper
  }
  const pools = Array.from({ length: d }, (_, j) => Array.from(columnOf(tr, j)).filter((v) => inBin(j, v)))
  const coverage = (rule: readonly number[]): number => {
    let c = 0
    for (let i = 0; i < tr.n; i++) if (rule.every((j) => inBin(j, tr.data[i * d + j] ?? 0))) c++
    return c / tr.n
  }
  interface Arm {
    readonly rule: number[]
    hits: number
    pulls: number
  }
  const pull = (arm: Arm, k: Key): void => {
    const rows = toFlat(randint(foldIn(k, 0), [batch], 0, tr.n))
    const data = new Float64Array(batch * d)
    rows.forEach((r, s) => {
      data.set(tr.data.subarray(r * d, (r + 1) * d), s * d)
    })
    arm.rule.forEach((j, t) => {
      const pool = pools[j] ?? []
      const picks = toFlat(randint(foldIn(k, 1 + t), [batch], 0, Math.max(1, pool.length)))
      for (let s = 0; s < batch; s++) data[s * d + j] = pool.length === 0 ? (xv[j] ?? 0) : (pool[picks[s] ?? 0] ?? 0)
    })
    const out = toFlat(predict(fromStorage(data, [batch, d], 'float64')))
    arm.hits += out.filter((v) => v === target).length
    arm.pulls += batch
  }
  const level = (arm: Arm, t: number): number => Math.log((1.1 * d * t ** 1.1) / delta) / Math.max(arm.pulls, 1)
  type State = {
    readonly beam: readonly Arm[]
    readonly length: number
    readonly best?: Arm
    readonly samples: number
    readonly done: boolean
  }
  const algorithm: Algorithm<State> = {
    name: 'anchors',
    config: { features: d, threshold: tau, beamSize: beam },
    schema: channels({
      metrics: {
        length: { description: 'the rule length being searched' },
        precision: { description: 'the best candidate precision' },
      },
    }),
    init: () => ({ beam: [{ rule: [], hits: 0, pulls: 0 }], length: 0, samples: 0, done: maxLength === 0 }),
    step: (s, ctx) => {
      // Candidates: each beam rule extended by one new feature.
      const seen = new Set<string>()
      const arms: Arm[] = []
      for (const b of s.beam) {
        for (let j = 0; j < d; j++) {
          if (b.rule.includes(j)) continue
          const rule = [...b.rule, j].sort((p, q) => p - q)
          const id = rule.join(',')
          if (seen.has(id)) continue
          seen.add(id)
          arms.push({ rule, hits: 0, pulls: 0 })
        }
      }
      let samples = s.samples
      arms.forEach((a, i) => {
        pull(a, foldIn(ctx.key, i))
        samples += batch
      })
      // KL-LUCB: sample the weakest of the top and strongest of the rest until their bounds separate.
      const top = Math.min(beam, arms.length)
      for (let round = 1; round < 1000; round++) {
        const ranked = [...arms].sort((p, q) => q.hits / q.pulls - p.hits / p.pulls)
        const head = ranked.slice(0, top)
        const tail = ranked.slice(top)
        const weakest = head.reduce((m, a) =>
          klBound(a.hits / a.pulls, level(a, round), false) < klBound(m.hits / m.pulls, level(m, round), false) ? a : m,
        )
        const strongest =
          tail.length === 0
            ? undefined
            : tail.reduce((m, a) =>
                klBound(a.hits / a.pulls, level(a, round), true) > klBound(m.hits / m.pulls, level(m, round), true)
                  ? a
                  : m,
              )
        const gap =
          strongest === undefined
            ? 0
            : klBound(strongest.hits / strongest.pulls, level(strongest, round), true) -
              klBound(weakest.hits / weakest.pulls, level(weakest, round), false)
        if (gap <= eps || weakest.pulls >= maxSamples) break
        pull(weakest, foldIn(ctx.key, 1000 + 2 * round))
        if (strongest !== undefined) pull(strongest, foldIn(ctx.key, 1001 + 2 * round))
        samples += batch * (strongest === undefined ? 1 : 2)
      }
      const ranked = [...arms].sort((p, q) => q.hits / q.pulls - p.hits / p.pulls).slice(0, top)
      // Confirm the threshold for the leaders, sampling until the bound decides or the budget runs out.
      for (const a of ranked) {
        for (let r = 1; r < 1000 && a.pulls < maxSamples; r++) {
          const lb = klBound(a.hits / a.pulls, level(a, r), false)
          const ub = klBound(a.hits / a.pulls, level(a, r), true)
          if (lb >= tau || ub < tau) break
          pull(a, foldIn(ctx.key, 5000 + r + a.rule.length * 7919))
          samples += batch
        }
      }
      const valid = ranked.filter((a) => klBound(a.hits / a.pulls, level(a, 1), false) >= tau)
      const best = valid.length === 0 ? s.best : valid.reduce((m, a) => (coverage(a.rule) > coverage(m.rule) ? a : m))
      const length = s.length + 1
      return {
        beam: ranked,
        length,
        ...(best === undefined ? {} : { best }),
        samples,
        done: valid.length > 0 || length >= maxLength,
      }
    },
    done: (s) => (s.done ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        length: s.length,
        precision: s.beam[0] === undefined || s.beam[0].pulls === 0 ? 0 : s.beam[0].hits / s.beam[0].pulls,
      },
    }),
  }
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'anchors'), maxSteps: maxLength })
  const chosen = state.best ?? state.beam[0] ?? { rule: [], hits: 0, pulls: 0 }
  const precision = chosen.pulls === 0 ? 0 : chosen.hits / chosen.pulls
  return {
    predicates: chosen.rule.map((j) => ({
      feature: j,
      lower: bounds[j]?.lower ?? Number.NEGATIVE_INFINITY,
      upper: bounds[j]?.upper ?? Number.POSITIVE_INFINITY,
    })),
    precision,
    precisionLower: chosen.pulls === 0 ? 0 : klBound(precision, level(chosen, 1), false),
    coverage: coverage(chosen.rule),
    samples: state.samples,
  }
}

/** Options of {@link surrogateTree}. */
export interface SurrogateTreeOptions {
  /** Fit a regression tree (variance) or a classification tree (Gini) (default regression). */
  readonly task?: 'regression' | 'classification'
  readonly maxDepth?: number
  readonly minSamplesLeaf?: number
  readonly minSamplesSplit?: number
}

/**
 * Fit a CART tree (Breiman et al. 1984; sklearn DecisionTreeRegressor and DecisionTreeClassifier's best splitter) to a
 * model's predictions: a global surrogate that approximates the model by a readable tree. Splits minimise the
 * children's weighted variance (or Gini impurity) at midpoints between consecutive distinct values.
 *
 * @param x - Features `[n, d]`.
 * @param targets - The model's predictions `[n]` (labels for classification).
 * @param options - task, maxDepth (default 3), minSamplesLeaf (default 1) and minSamplesSplit (default 2).
 * @returns The tree as arrays (leaf values are means, or for classification the majority class).
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const t = explain.surrogateTree([[0], [1], [2], [3]], [0, 0, 5, 5], { maxDepth: 1 })
 * t.threshold[0] // => 1.5
 * toFlat(explain.treePredict(t, [[0.5], [2.5]])) // => [0, 5]
 * ```
 */
export function surrogateTree(x: TensorLike, targets: TensorLike, options: SurrogateTreeOptions = {}): TreeModel {
  const m = matrixView(x, 'surrogateTree')
  const y = targetVector(targets, 'surrogateTree', m.n)
  const classification = options.task === 'classification'
  const classes = classification ? Array.from(new Set(y)).sort((a, b) => a - b) : []
  const maxDepth = options.maxDepth ?? 3
  const minLeaf = options.minSamplesLeaf ?? 1
  const minSplit = options.minSamplesSplit ?? 2
  const tree = {
    childrenLeft: [] as number[],
    childrenRight: [] as number[],
    feature: [] as number[],
    threshold: [] as number[],
    value: [] as number[],
    cover: [] as number[],
  }
  const impurityOf = (rows: readonly number[]): { readonly value: number; readonly impurity: number } => {
    if (!classification) {
      const mu = rows.reduce((s, i) => s + (y[i] ?? 0), 0) / rows.length
      return { value: mu, impurity: rows.reduce((s, i) => s + ((y[i] ?? 0) - mu) ** 2, 0) / rows.length }
    }
    const counts = classes.map((c) => rows.filter((i) => y[i] === c).length)
    let best = 0
    counts.forEach((c, k) => {
      if (c > (counts[best] ?? 0)) best = k
    })
    return { value: classes[best] ?? 0, impurity: 1 - counts.reduce((s, c) => s + (c / rows.length) ** 2, 0) }
  }
  const grow = (rows: number[], depth: number): number => {
    const id = tree.value.length
    const { value, impurity } = impurityOf(rows)
    tree.childrenLeft.push(-1)
    tree.childrenRight.push(-1)
    tree.feature.push(-2)
    tree.threshold.push(-2)
    tree.value.push(value)
    tree.cover.push(rows.length)
    if (depth >= maxDepth || rows.length < minSplit || rows.length < 2 * minLeaf || impurity <= 2.220446049250313e-16)
      return id
    let best: { readonly f: number; readonly t: number; readonly score: number } | undefined
    for (let f = 0; f < m.d; f++) {
      const sorted = [...rows].sort((a, b) => (m.data[a * m.d + f] ?? 0) - (m.data[b * m.d + f] ?? 0))
      // Running sums give each split's weighted child impurity in O(n).
      let sumL = 0
      let sqL = 0
      const countsL = classes.map(() => 0)
      const total = sorted.reduce((s, i) => s + (y[i] ?? 0), 0)
      const totalSq = sorted.reduce((s, i) => s + (y[i] ?? 0) ** 2, 0)
      const countsAll = classes.map((c) => sorted.filter((i) => y[i] === c).length)
      for (let p = 1; p < sorted.length; p++) {
        const prev = sorted[p - 1] ?? 0
        const yi = y[prev] ?? 0
        sumL += yi
        sqL += yi * yi
        if (classification) countsL[classes.indexOf(yi)] = (countsL[classes.indexOf(yi)] ?? 0) + 1
        const a = m.data[prev * m.d + f] ?? 0
        const b = m.data[(sorted[p] ?? 0) * m.d + f] ?? 0
        if (b <= a + 1e-7 || p < minLeaf || sorted.length - p < minLeaf) continue
        const nl = p
        const nr = sorted.length - p
        let score: number
        if (classification) {
          const gl = 1 - countsL.reduce((s, c) => s + (c / nl) ** 2, 0)
          const gr = 1 - countsAll.reduce((s, c, k) => s + ((c - (countsL[k] ?? 0)) / nr) ** 2, 0)
          score = nl * gl + nr * gr
        } else {
          score = sqL - (sumL * sumL) / nl + (totalSq - sqL) - ((total - sumL) * (total - sumL)) / nr
        }
        // sklearn's threshold: the midpoint, or the lower value when rounding lands on the upper one.
        const mid = a / 2 + b / 2
        if (best === undefined || score < best.score - 1e-12) best = { f, t: mid === b ? a : mid, score }
      }
    }
    if (best === undefined) return id
    const split = best
    const left = rows.filter((i) => (m.data[i * m.d + split.f] ?? 0) <= split.t)
    const right = rows.filter((i) => (m.data[i * m.d + split.f] ?? 0) > split.t)
    tree.feature[id] = split.f
    tree.threshold[id] = split.t
    tree.childrenLeft[id] = grow(left, depth + 1)
    tree.childrenRight[id] = grow(right, depth + 1)
    return id
  }
  grow(
    Array.from({ length: m.n }, (_, i) => i),
    0,
  )
  return tree
}

/**
 * How faithfully a surrogate reproduces a model on data: R^2 of the surrogate's outputs against the model's (or
 * agreement, for labels).
 *
 * @param model - The model's outputs `[n]`.
 * @param surrogate - The surrogate's outputs `[n]`.
 * @param task - `'regression'` (R^2, default) or `'classification'` (agreement).
 * @returns The fidelity (1 is perfect).
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * explain.fidelity([1, 2, 3], [1, 2, 3]) // => 1
 * explain.fidelity([0, 1, 1, 0], [0, 1, 0, 0], 'classification') // => 0.75
 * ```
 */
export function fidelity(
  model: TensorLike,
  surrogate: TensorLike,
  task: 'regression' | 'classification' = 'regression',
): number {
  const a = targetVector(model, 'fidelity')
  const b = targetVector(surrogate, 'fidelity', a.length)
  if (task === 'classification') return a.filter((v, i) => v === b[i]).length / a.length
  const mu = a.reduce((s, v) => s + v, 0) / a.length
  let ssr = 0
  let sst = 0
  a.forEach((v, i) => {
    ssr += (v - (b[i] ?? 0)) ** 2
    sst += (v - mu) ** 2
  })
  if (sst === 0) return ssr === 0 ? 1 : 0
  return 1 - ssr / sst
}

/**
 * A global surrogate of a model in one call: predict on the data, fit {@link surrogateTree} to the predictions and
 * report its fidelity.
 *
 * @param predict - The model.
 * @param x - Data `[n, d]` (the region the surrogate should describe).
 * @param options - The tree options.
 * @returns The tree and its fidelity on the data.
 * @example
 * ```ts
 * import * as explain from 'whitebox-engine/explain'
 * import { tensor } from 'whitebox-engine/tensor'
 * const step = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => (x.data[i] > 1 ? 1 : 0)))
 * explain.globalSurrogate(step, [[0], [1], [2], [3]], { maxDepth: 1 }).fidelity // => 1
 * ```
 */
export function globalSurrogate(
  predict: PredictFunction,
  x: TensorLike,
  options: SurrogateTreeOptions = {},
): { readonly tree: TreeModel; readonly fidelity: number } {
  const m = matrixView(x, 'globalSurrogate')
  const xt = fromStorage(m.data, [m.n, m.d], 'float64')
  const out = predict(xt)
  const k = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
  const ov = toFlat(out)
  const target = Float64Array.from({ length: m.n }, (_, i) => ov[i * k] ?? 0)
  const tree = surrogateTree(xt, target, options)
  const leaf = (i: number): number => {
    let node = 0
    for (let g = 0; (tree.childrenLeft[node] ?? -1) >= 0 && g < tree.value.length; g++) {
      node =
        (m.data[i * m.d + (tree.feature[node] ?? 0)] ?? 0) <= (tree.threshold[node] ?? 0)
          ? (tree.childrenLeft[node] ?? 0)
          : (tree.childrenRight[node] ?? 0)
    }
    return tree.value[node] ?? 0
  }
  const pred = Float64Array.from({ length: m.n }, (_, i) => leaf(i))
  return { tree, fidelity: fidelity(target, pred, options.task ?? 'regression') }
}
