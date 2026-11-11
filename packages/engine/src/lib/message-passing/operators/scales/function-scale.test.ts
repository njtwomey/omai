import { describe, expect, it } from 'vitest'
import { messageArg } from '../../core/args.ts'
import { gamma, gammaEta, gaussian, gaussianEta, gaussianMoments } from '../../core/families.ts'
import { gaussianMv, gaussianMvEta, gaussianMvMoments } from '../../core/families-vector.ts'
import { gaussianMvFunctionScale } from './function-scale.ts'
import { gaussianMvPrecisionScale } from './scale.ts'

const v2 = gaussianMv(2)

/** The coefficient chains `1 / sqrt(d_j exp(x) + n_j)`, as the compiler writes them. */
const chainsOf = (d: number[], n: number[]): unknown =>
  d.map((dj, j) => [['exp'], ['mul', dj], ['add', n[j] ?? 0], ['sqrt'], ['recip']])

/**
 * The moments of `beta` (cavity N(m, V), two coordinates) and of `x` (cavity N(mx, vx)) under `beta_j ~ N(0, 1 / (d_j e^x + n_j))`,
 * by a three-dimensional grid, with the log mass.
 */
function brute(m: number[], v: number[], mx: number, vx: number, d: number[], n: number[]) {
  const grid = 200
  const sd = [Math.sqrt(v[0] as number), Math.sqrt(v[3] as number)]
  const det = (v[0] as number) * (v[3] as number) - (v[1] as number) ** 2
  const prec = [(v[3] as number) / det, -(v[1] as number) / det, -(v[2] as number) / det, (v[0] as number) / det]
  const nt = 400
  const lo = mx - 9 * Math.sqrt(vx)
  const hi = mx + 9 * Math.sqrt(vx)
  const dt = (hi - lo) / (nt - 1)
  const acc = { z: 0, e1: 0, e2: 0, e11: 0, e22: 0, e12: 0, ex: 0, exx: 0 }
  for (let k = 0; k < nt; k++) {
    const t = lo + dt * k
    const p = [(d[0] as number) * Math.exp(t) + (n[0] as number), (d[1] as number) * Math.exp(t) + (n[1] as number)]
    for (let a = 0; a < grid; a++) {
      const x = (m[0] as number) - 9 * (sd[0] as number) + (18 * (sd[0] as number) * a) / (grid - 1)
      for (let b = 0; b < grid; b++) {
        const y = (m[1] as number) - 9 * (sd[1] as number) + (18 * (sd[1] as number) * b) / (grid - 1)
        const dx = x - (m[0] as number)
        const dy = y - (m[1] as number)
        const quad = (prec[0] as number) * dx * dx + 2 * (prec[1] as number) * dx * dy + (prec[3] as number) * dy * dy
        const w = Math.exp(
          -0.5 * quad -
            (0.5 * (t - mx) ** 2) / vx +
            0.5 * Math.log((p[0] as number) * (p[1] as number)) -
            0.5 * ((p[0] as number) * x * x + (p[1] as number) * y * y),
        )
        acc.z += w
        acc.e1 += w * x
        acc.e2 += w * y
        acc.e11 += w * x * x
        acc.e22 += w * y * y
        acc.e12 += w * x * y
        acc.ex += w * t
        acc.exx += w * t * t
      }
    }
  }
  const m1 = acc.e1 / acc.z
  const m2 = acc.e2 / acc.z
  const mt = acc.ex / acc.z
  return {
    mean: [m1, m2],
    cov: [acc.e11 / acc.z - m1 * m1, acc.e12 / acc.z - m1 * m2, acc.e12 / acc.z - m1 * m2, acc.e22 / acc.z - m2 * m2],
    x: { mean: mt, variance: acc.exx / acc.z - mt * mt },
  }
}

describe('gaussian-mv-function-scale', () => {
  it('the dense message and the message to a Gaussian x match a three-dimensional grid, with a correlated cavity', () => {
    const [m, v] = [
      [0.8, -0.4],
      [1.2, 0.5, 0.5, 0.9],
    ]
    const [mx, vx, d, n] = [1.1, 0.8, [1, 0.2], [0, 0.5]]
    const ref = brute(m as number[], v as number[], mx as number, vx as number, d as number[], n as number[])
    const cavity = gaussianMvEta(m as number[], v as number[])
    const params = { steps: chainsOf(d as number[], n as number[]) } as never
    const args = [messageArg(v2, cavity), messageArg(gaussian, gaussianEta(mx as number, vx as number))]
    const toBeta = gaussianMvFunctionScale.ep?.({ mode: 'ep', to: 0, args, params, power: 1 })
    const tilted = cavity.map((x, i) => x + (toBeta?.message[i] ?? 0))
    const mo = gaussianMvMoments(tilted, 2)
    for (let i = 0; i < 2; i++) expect(mo.mean[i]).toBeCloseTo(ref.mean[i] as number, 3)
    for (let i = 0; i < 4; i++) expect(mo.covariance[i]).toBeCloseTo(ref.cov[i] as number, 3)
    const toX = gaussianMvFunctionScale.ep?.({ mode: 'ep', to: 1, args, params, power: 1 })
    const xEta = gaussianEta(mx as number, vx as number).map((e, i) => e + (toX?.message[i] ?? 0))
    const xo = gaussianMoments(xEta)
    expect(xo.mean).toBeCloseTo(ref.x.mean, 3)
    expect(xo.variance).toBeCloseTo(ref.x.variance, 3)
  })

  it('a gamma input with the chain of a precision is the factor of gaussian-mv-precision-scale', () => {
    const cavity = gaussianMvEta([0.8, -0.4], [1.2, 0.5, 0.5, 0.9])
    const steps = [
      [['mul', 1], ['add', 0], ['sqrt'], ['recip']],
      [['mul', 0.2], ['add', 0.5], ['sqrt'], ['recip']],
    ]
    const args = [messageArg(v2, cavity), messageArg(gamma, gammaEta(2, 1.5))]
    for (const to of [0, 1] as const) {
      const own = gaussianMvFunctionScale.ep?.({ mode: 'ep', to, args, params: { steps } as never, power: 1 })
      const reference = gaussianMvPrecisionScale.ep?.({
        mode: 'ep',
        to,
        args,
        params: { scale: [1, 0.2], offset: [0, 0.5] },
        power: 1,
      })
      for (const [i, value] of (reference?.message ?? []).entries()) expect(own?.message[i]).toBeCloseTo(value, 5)
    }
  })

  it('a flat cavity gives the prior scale mixture: the covariance is E[1 / (d exp(x) + n)]', () => {
    const [mx, vx, d, n] = [0.5, 1.5, [2, 0.5], [0, 1]]
    const args = [messageArg(v2, v2.flat), messageArg(gaussian, gaussianEta(mx as number, vx as number))]
    const out = gaussianMvFunctionScale.ep?.({
      mode: 'ep',
      to: 0,
      args,
      params: { steps: chainsOf(d as number[], n as number[]) } as never,
      power: 1,
    })
    const mo = gaussianMvMoments(out?.message ?? [], 2)
    let z = 0
    let e0 = 0
    let e1 = 0
    const nt = 20000
    for (let k = 0; k < nt; k++) {
      const t = -20 + (40 * k) / (nt - 1)
      const w = Math.exp(-(0.5 * (t - (mx as number)) ** 2) / (vx as number))
      z += w
      e0 += w / (2 * Math.exp(t))
      e1 += w / (0.5 * Math.exp(t) + 1)
    }
    expect(mo.covariance[1]).toBeCloseTo(0, 8)
    expect(mo.covariance[3]).toBeCloseTo(e1 / z, 3)
    // The first coefficient has no offset, so its prior variance is the log-normal mean exp(-x) / 2, heavy-tailed on the grid.
    expect(mo.covariance[0]).toBeCloseTo(e0 / z, 1)
  })

  it('the evidence is the log mass against the unnormalised cavities', () => {
    const [mx, vx] = [0.3, 0.5]
    const cavity = gaussianMvEta([0.5, 0.2], [1.5, 0.3, 0.3, 1])
    const args = [messageArg(v2, cavity), messageArg(gaussian, gaussianEta(mx, vx))]
    const steps = chainsOf([1, 0.4], [0, 0.3])
    const got = gaussianMvFunctionScale.logEvidence?.({ mode: 'ep', to: 0, args, params: { steps } as never, power: 1 })
    // The vector in closed form for each x on a grid, the Gaussian message of x taken unnormalised (its mass is in the grid).
    const nt = 6000
    const [h0, h1, j00, j01, j11] = [cavity[0] ?? 0, cavity[1] ?? 0, cavity[2] ?? 0, cavity[3] ?? 0, cavity[5] ?? 0]
    const dt = (20 * Math.sqrt(vx)) / (nt - 1)
    // the Gaussian message exp(h t - J t^2 / 2) with h = mx / vx, J = 1 / vx, not normalised.
    const logMessage = (t: number): number => (mx / vx) * t - (0.5 * t * t) / vx
    let zz = 0
    for (let k = 0; k < nt; k++) {
      const t = mx - 10 * Math.sqrt(vx) + dt * k
      const p0 = Math.exp(t)
      const p1 = 0.4 * Math.exp(t) + 0.3
      const a = j00 + p0
      const b = j01
      const c = j11 + p1
      const det = a * c - b * b
      const quad = (c * h0 * h0 - 2 * b * h0 * h1 + a * h1 * h1) / det
      zz += Math.exp(logMessage(t) + 0.5 * Math.log((p0 * p1) / det) + 0.5 * quad) * dt
    }
    expect(got).toBeCloseTo(Math.log(zz), 5)
  })
})
