import { describe, expect, it } from 'vitest'
import { gamma, gammaEta, gaussian, gaussianEta } from '../families.ts'
import { messageArg, valueArg } from './args.ts'
import { gaussianGammaPrecision } from './gaussian.ts'

/** The moments of x (cavity N(m, v)) and of lambda (gamma cavity) under the factor N(x; 0, 1 / (scale lambda + offset)), by a two-dimensional grid. */
function brute(m: number, v: number, shape: number, rate: number, scale: number, offset: number) {
  const nx = 1600
  const nt = 1600
  const sd = Math.sqrt(v)
  const xs = Array.from({ length: nx }, (_, i) => m - 10 * sd + (20 * sd * i) / (nx - 1))
  const dx = (20 * sd) / (nx - 1)
  const lo = -12
  const hi = 8
  const dt = (hi - lo) / (nt - 1)
  let z = 0
  let ex = 0
  let ex2 = 0
  let el = 0
  let ell = 0
  for (let k = 0; k < nt; k++) {
    const t = lo + k * dt
    const lam = Math.exp(t)
    const logGamma = shape * t - rate * lam
    const p = scale * lam + offset
    for (const x of xs) {
      const w = Math.exp(logGamma - (0.5 * (x - m) ** 2) / v + 0.5 * Math.log(p) - 0.5 * p * x * x) * dx * dt
      z += w
      ex += w * x
      ex2 += w * x * x
      el += w * lam
      ell += w * t
    }
  }
  const mean = ex / z
  return { mean, variance: ex2 / z - mean * mean, lambda: el / z, logLambda: ell / z }
}

describe('gaussian-gamma-precision with a scale and an offset', () => {
  it('the tilted moments of the value and the precision match a two-dimensional grid', () => {
    for (const [scale, offset] of [
      [1, 0],
      [2.5, 0],
      [0.3, 0.8],
      [0, 2],
    ] as const) {
      const args = () => [
        messageArg(gaussian, gaussianEta(0.7, 2)),
        valueArg(gaussian, 0),
        messageArg(gamma, gammaEta(3, 1.5)),
      ]
      const ref = brute(0.7, 2, 3, 1.5, scale, offset)
      const params = { scale, offset }
      const x = gaussianGammaPrecision.ep?.({ mode: 'ep', to: 0, args: args(), params, power: 1 })
      const j = 1 / 2 + (x?.message[1] ?? 0)
      const mean = (0.7 / 2 + (x?.message[0] ?? 0)) / j
      expect(mean).toBeCloseTo(ref.mean, 4)
      expect(1 / j).toBeCloseTo(ref.variance, 4)
      const l = gaussianGammaPrecision.ep?.({ mode: 'ep', to: 2, args: args(), params, power: 1 })
      // The site plus the cavity is the gamma with these moments: check E lambda through the shape and rate.
      const shape = 3 + 0 + ((l?.message[0] ?? 0) as number)
      const rate = 1.5 - ((l?.message[1] ?? 0) as number)
      expect(shape / rate).toBeCloseTo(ref.lambda, 3)
    }
  })

  it('a variational message for scale 2 is the scale times the precision', () => {
    const args = [valueArg(gaussian, 1), valueArg(gaussian, 0), messageArg(gamma, gammaEta(3, 1.5))]
    const a = gaussianGammaPrecision.vmp?.({ mode: 'vmp', to: 2, args, params: { scale: 2 }, power: 1 })
    expect(a?.message).toEqual([0.5, -1])
    const b = gaussianGammaPrecision.vmp?.({
      mode: 'vmp',
      to: 0,
      args: [messageArg(gaussian, gaussian.flat), valueArg(gaussian, 0), messageArg(gamma, gammaEta(3, 1.5))],
      params: { scale: 2 },
      power: 1,
    })
    expect(b?.message).toEqual([0, 4])
  })
})

import { gammaShapeRate } from '../families.ts'
import { gaussianMv, gaussianMvEta, gaussianMvMoments } from '../families-vector.ts'
import { gaussianMvPrecisionScale } from './scale.ts'

/** The moments of beta (cavity N(m, V), two coordinates) and lambda under prod N(beta_j; 0, 1 / (s_j lambda + o_j)), by a three-dimensional grid. */
function brute3(m: number[], v: number[], shape: number, rate: number, scale: number[], offset: number[]) {
  const n = 220
  const sd = [Math.sqrt(v[0] as number), Math.sqrt(v[3] as number)]
  const det = (v[0] as number) * (v[3] as number) - (v[1] as number) ** 2
  const prec = [(v[3] as number) / det, -(v[1] as number) / det, -(v[2] as number) / det, (v[0] as number) / det]
  const nt = 500
  const lo = -9
  const hi = 7
  let z = 0
  let e1 = 0
  let e2 = 0
  let e11 = 0
  let e22 = 0
  let e12 = 0
  let el = 0
  for (let k = 0; k < nt; k++) {
    const t = lo + ((hi - lo) * k) / (nt - 1)
    const lam = Math.exp(t)
    const p = [(scale[0] as number) * lam + (offset[0] as number), (scale[1] as number) * lam + (offset[1] as number)]
    for (let a = 0; a < n; a++) {
      const x = (m[0] as number) - 9 * (sd[0] as number) + (18 * (sd[0] as number) * a) / (n - 1)
      for (let b = 0; b < n; b++) {
        const y = (m[1] as number) - 9 * (sd[1] as number) + (18 * (sd[1] as number) * b) / (n - 1)
        const dx = x - (m[0] as number)
        const dy = y - (m[1] as number)
        const quad = (prec[0] as number) * dx * dx + 2 * (prec[1] as number) * dx * dy + (prec[3] as number) * dy * dy
        const w = Math.exp(
          shape * t -
            rate * lam -
            0.5 * quad +
            0.5 * Math.log((p[0] as number) * (p[1] as number)) -
            0.5 * ((p[0] as number) * x * x + (p[1] as number) * y * y),
        )
        z += w
        e1 += w * x
        e2 += w * y
        e11 += w * x * x
        e22 += w * y * y
        e12 += w * x * y
        el += w * lam
      }
    }
  }
  const m1 = e1 / z
  const m2 = e2 / z
  return {
    mean: [m1, m2],
    cov: [e11 / z - m1 * m1, e12 / z - m1 * m2, e12 / z - m1 * m2, e22 / z - m2 * m2],
    lambda: el / z,
  }
}

describe('gaussian-mv-precision-scale', () => {
  const v2 = gaussianMv(2)
  it('the dense message and the multiplier message match a three-dimensional grid, with correlated cavities and offsets', () => {
    const cavity = gaussianMvEta([0.8, -0.4], [1.2, 0.5, 0.5, 0.9])
    const params = { scale: [1, 0.2], offset: [0, 0.5] }
    const ref = brute3([0.8, -0.4], [1.2, 0.5, 0.5, 0.9], 2, 1.5, params.scale, params.offset)
    const toBeta = gaussianMvPrecisionScale.ep?.({
      mode: 'ep',
      to: 0,
      args: [messageArg(v2, cavity), messageArg(gamma, gammaEta(2, 1.5))],
      params,
      power: 1,
    })
    const tilted = cavity.map((x, i) => x + (toBeta?.message[i] ?? 0))
    const mo = gaussianMvMoments(tilted, 2)
    expect(mo.mean[0]).toBeCloseTo(ref.mean[0] as number, 3)
    expect(mo.mean[1]).toBeCloseTo(ref.mean[1] as number, 3)
    for (let i = 0; i < 4; i++) expect(mo.covariance[i]).toBeCloseTo(ref.cov[i] as number, 3)
    const toLambda = gaussianMvPrecisionScale.ep?.({
      mode: 'ep',
      to: 1,
      args: [messageArg(v2, cavity), messageArg(gamma, gammaEta(2, 1.5))],
      params,
      power: 1,
    })
    const { shape, rate } = gammaShapeRate([1 + (toLambda?.message[0] ?? 0), -1.5 + (toLambda?.message[1] ?? 0)])
    expect(shape / rate).toBeCloseTo(ref.lambda, 3)
  })

  it('a flat cavity gives the prior scale mixture: the covariance is E[1 / (s lambda + o)]', () => {
    const params = { scale: [2, 0.5], offset: [0, 1] }
    const out = gaussianMvPrecisionScale.ep?.({
      mode: 'ep',
      to: 0,
      args: [messageArg(v2, v2.flat), messageArg(gamma, gammaEta(3, 2))],
      params,
      power: 1,
    })
    const mo = gaussianMvMoments(out?.message ?? [], 2)
    let e0 = 0
    let e1 = 0
    let z = 0
    const nt = 20000
    for (let k = 0; k < nt; k++) {
      const t = -14 + (24 * k) / (nt - 1)
      const lam = Math.exp(t)
      const w = Math.exp(3 * t - 2 * lam)
      z += w
      e0 += w / (2 * lam)
      e1 += w / (0.5 * lam + 1)
    }
    expect(mo.covariance[0]).toBeCloseTo(e0 / z, 2)
    expect(mo.covariance[3]).toBeCloseTo(e1 / z, 3)
    expect(mo.covariance[1]).toBeCloseTo(0, 8)
  })
})

describe('the evidence of gaussian-gamma-precision (what a gate weighs its branches by)', () => {
  it('is the masses of the cavities plus the log of the expected density of the observation, against a grid over the precision', () => {
    const [y, m, v, shape, rate] = [1.3, 0.4, 0.5, 3, 2]
    const args = [
      valueArg(gaussian, y),
      messageArg(gaussian, gaussianEta(m, v)),
      messageArg(gamma, gammaEta(shape, rate)),
    ]
    const got = gaussianGammaPrecision.logEvidence?.({ mode: 'ep', to: 0, args, params: {}, power: 1 }) as number
    // E over the gamma cavity of N(y; m, v + 1 / lambda), the mean integrated out, on a grid in log lambda.
    const nt = 20000
    const lo = -14
    const hi = 8
    let z = 0
    for (let k = 0; k < nt; k++) {
      const t = lo + ((hi - lo) * (k + 0.5)) / nt
      const lam = Math.exp(t)
      const s2 = v + 1 / lam
      const density = Math.exp(shape * Math.log(rate) - (0.5 * (y - m) ** 2) / s2) / Math.sqrt(2 * Math.PI * s2)
      // the gamma density in log lambda: lambda^shape exp(-rate lambda) / Gamma(shape) * rate^shape
      z += (Math.exp(shape * t - rate * lam) * density * ((hi - lo) / nt)) / Math.exp(shape * Math.log(rate))
    }
    // The gamma cavity enters unnormalised (its mass is in the grid), the Gaussian cavity's mass is added.
    expect(got).toBeCloseTo(gaussian.logPartition(gaussianEta(m, v)) + Math.log(z), 6)
  })
})
