import { describe, expect, it } from 'vitest'
import { logSumExpOf } from '../../base/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'
import { bocpd } from './changepoint.ts'
import type { ConjugateModel } from './segment-models.ts'

/**
 * BOCPD against exact enumeration: the evidence is the sum over every segmentation of the series of its prior (the
 * hazard at each step) times the product of the segments' marginal likelihoods, and the final run-length posterior is
 * the share of that sum whose last segment has each length. The marginal likelihoods here are the closed forms of each
 * conjugate model, not the recursion of the predictives that BOCPD runs.
 */

const lg = (x: number): number => item(lgamma(x))
const LOG_2PI = Math.log(2 * Math.PI)
const sum = (xs: readonly number[]): number => xs.reduce((a, v) => a + v, 0)

/** log p(segment) for the observations y[a..b) under a model, by the closed form of its marginal likelihood. */
function logMarginal(model: ConjugateModel, y: readonly number[], a: number, b: number): number {
  const seg = y.slice(a, b)
  const n = seg.length
  switch (model.kind) {
    case 'normal-known-variance': {
      const s2 = model.sigma ** 2
      const t2 = model.priorStd ** 2
      const z = seg.map((v) => v - model.priorMean)
      const quad = (sum(z.map((v) => v * v)) - (t2 * sum(z) ** 2) / (s2 + n * t2)) / s2
      return -0.5 * n * Math.log(2 * Math.PI * s2) - 0.5 * Math.log(1 + (n * t2) / s2) - 0.5 * quad
    }
    case 'normal-known-mean': {
      const ss = sum(seg.map((v) => (v - model.mean) ** 2))
      return (
        model.alpha * Math.log(model.beta) +
        lg(model.alpha + n / 2) -
        lg(model.alpha) -
        0.5 * n * LOG_2PI -
        (model.alpha + n / 2) * Math.log(model.beta + ss / 2)
      )
    }
    case 'normal-gamma': {
      const m = sum(seg) / n
      const ss = sum(seg.map((v) => (v - m) ** 2))
      const kn = model.kappa0 + n
      const an = model.alpha0 + n / 2
      const bn = model.beta0 + 0.5 * ss + (model.kappa0 * n * (m - model.mu0) ** 2) / (2 * kn)
      return (
        lg(an) -
        lg(model.alpha0) +
        model.alpha0 * Math.log(model.beta0) -
        an * Math.log(bn) +
        0.5 * Math.log(model.kappa0 / kn) -
        0.5 * n * LOG_2PI
      )
    }
    case 'poisson-gamma': {
      const s = sum(seg)
      return (
        -sum(seg.map((v) => lg(v + 1))) +
        model.shape * Math.log(model.rate) +
        lg(model.shape + s) -
        lg(model.shape) -
        (model.shape + s) * Math.log(model.rate + n)
      )
    }
    case 'bernoulli-beta': {
      const k = sum(seg.map((v) => (v > 0.5 ? 1 : 0)))
      const lbeta = (p: number, q: number): number => lg(p) + lg(q) - lg(p + q)
      return lbeta(model.alpha + k, model.beta + n - k) - lbeta(model.alpha, model.beta)
    }
    case 'ar1-known-noise': {
      const s2 = model.sigma ** 2
      const t2 = model.priorStd ** 2
      const x = seg.map((_, i) => (a + i > 0 ? (y[a + i - 1] ?? 0) : 0))
      const sxx = sum(x.map((v) => v * v))
      const sxy = sum(x.map((v, i) => v * (seg[i] ?? 0)))
      const syy = sum(seg.map((v) => v * v))
      const quad = (syy - (t2 * sxy * sxy) / (s2 + t2 * sxx)) / s2
      return -0.5 * n * Math.log(2 * Math.PI * s2) - 0.5 * Math.log(1 + (t2 * sxx) / s2) - 0.5 * quad
    }
    case 'trend-known-noise': {
      const s2 = model.sigma ** 2
      const aa = model.interceptStd ** 2
      const bb = model.slopeStd ** 2
      const z = seg.map((v) => v - model.priorMean)
      const u = seg.map((_, i) => i)
      // G = X'X for X = [1, u]; b = X'z.
      const g00 = n
      const g01 = sum(u)
      const g11 = sum(u.map((v) => v * v))
      const b0 = sum(z)
      const b1 = sum(u.map((v, i) => v * (z[i] ?? 0)))
      const det = (1 + (aa * g00) / s2) * (1 + (bb * g11) / s2) - (aa * bb * g01 * g01) / (s2 * s2)
      const m00 = s2 / aa + g00
      const m11 = s2 / bb + g11
      const detM = m00 * m11 - g01 * g01
      const solve = (m11 * b0 * b0 - 2 * g01 * b0 * b1 + m00 * b1 * b1) / detM
      const quad = (sum(z.map((v) => v * v)) - solve) / s2
      return -0.5 * n * Math.log(2 * Math.PI * s2) - 0.5 * Math.log(det) - 0.5 * quad
    }
  }
}

/** The evidence and the final run-length posterior by summing over all 2^(n-1) segmentations. */
function enumerate(
  model: ConjugateModel,
  y: readonly number[],
  hazard: (runLength: number) => number,
): { readonly logEvidence: number; readonly lastRun: readonly number[] } {
  const n = y.length
  const logs: number[] = []
  const lasts: number[] = []
  for (let mask = 0; mask < 2 ** (n - 1); mask++) {
    let start = 0
    let logp = 0
    const bounds = [0]
    for (let t = 1; t < n; t++) {
      const cut = (mask >> (t - 1)) & 1
      const h = hazard(t - start)
      logp += cut === 1 ? Math.log(h) : Math.log(1 - h)
      if (cut === 1) {
        start = t
        bounds.push(t)
      }
    }
    if (logp === Number.NEGATIVE_INFINITY) continue
    bounds.push(n)
    for (let k = 0; k + 1 < bounds.length; k++) logp += logMarginal(model, y, bounds[k] ?? 0, bounds[k + 1] ?? n)
    logs.push(logp)
    lasts.push(n - (bounds[bounds.length - 2] ?? 0))
  }
  const total = logSumExpOf(logs)
  const lastRun = new Array<number>(n + 1).fill(0)
  for (const [i, l] of logs.entries()) lastRun[lasts[i] ?? 0] = (lastRun[lasts[i] ?? 0] ?? 0) + Math.exp(l - total)
  return { logEvidence: total, lastRun }
}

const cases: readonly { readonly name: string; readonly model: ConjugateModel; readonly y: readonly number[] }[] = [
  {
    name: 'a shift of the mean',
    model: { kind: 'normal-known-variance', sigma: 0.7, priorMean: 0.5, priorStd: 2 },
    y: [0.1, -0.4, 0.3, 2.9, 3.4, 3.1, 0.2],
  },
  {
    name: 'a shift of the variance',
    model: { kind: 'normal-known-mean', mean: 0.2, alpha: 3, beta: 2 },
    y: [0.3, -0.5, 0.2, 3.1, -2.7, 2.2, 0.1],
  },
  {
    name: 'a shift of the mean and the variance',
    model: { kind: 'normal-gamma', mu0: 0.5, kappa0: 0.4, alpha0: 1.5, beta0: 0.8 },
    y: [0.3, -0.5, 0.2, 3.1, 3.7, 2.2, -1.1],
  },
  {
    name: 'a shift of the rate of counts',
    model: { kind: 'poisson-gamma', shape: 2, rate: 0.5 },
    y: [2, 1, 3, 9, 11, 8, 1],
  },
  {
    name: 'a shift of a binary frequency',
    model: { kind: 'bernoulli-beta', alpha: 0.7, beta: 1.3 },
    y: [0, 0, 1, 0, 1, 1, 1],
  },
  {
    name: 'a shift of the AR(1) coefficient',
    model: { kind: 'ar1-known-noise', sigma: 0.6, priorStd: 0.5 },
    y: [0.5, 0.3, 0.2, -1.0, 0.9, -0.8, 0.7],
  },
  {
    name: 'a shift of the trend',
    model: { kind: 'trend-known-noise', sigma: 0.5, priorMean: 0.3, interceptStd: 2, slopeStd: 0.3 },
    y: [0.2, 0.4, 0.9, 3.0, 2.8, 2.5, 2.4],
  },
]

describe('BOCPD equals the sum over every segmentation', () => {
  for (const c of cases) {
    it(`${c.name}: the evidence and the final run-length posterior`, () => {
      const r = bocpd(c.y, c.model, 0.2)
      const truth = enumerate(c.model, c.y, () => 0.2)
      expect(r.logEvidence).toBeCloseTo(truth.logEvidence, 9)
      const last = r.runLength[c.y.length - 1] ?? new Float64Array()
      for (const [len, p] of truth.lastRun.entries()) expect(last[len] ?? 0).toBeCloseTo(p, 9)
    })

    it(`${c.name}: with a hazard that depends on the run length`, () => {
      const hazard = (run: number): number => (run >= 2 ? 0.15 + 0.05 * Math.min(run, 4) : 0)
      const r = bocpd(c.y, c.model, hazard)
      const truth = enumerate(c.model, c.y, hazard)
      expect(r.logEvidence).toBeCloseTo(truth.logEvidence, 9)
    })
  }
})
