// A non-centred ARD regression (a coefficient `z_j / sqrt(a_j)` with a gamma relevance `a_j` of its own) against its exact posterior, which
// two coefficients make a two-dimensional integral: given the relevances the coefficients are Gaussian with a closed-form evidence, so the
// posterior of the relevances is a density on a grid and the coefficients' is the mixture it implies. Expectation propagation and variational
// message passing (the products assembled into one vector, `gaussian-mv-stack`) are checked against it.
import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { epSpecMethod, vmpSpecMethod } from '../inference/mp.ts'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import type { Posterior } from '../posterior.ts'

const [shape, rate, sigma] = [1, 1, 0.7]
const X = [
  [1, 0.2],
  [0.4, 1],
  [0.9, -0.3],
  [-0.5, 0.8],
  [0.2, 0.6],
  [1.1, 0.1],
  [-0.7, -0.4],
  [0.3, -0.9],
].slice(0, 5)
// The first coefficient is large, the second is near zero (the relevance of the second should grow).
const y = X.map((r, i) => 2.2 * (r[0] as number) + 0.1 * (r[1] as number) + 0.3 * Math.sin(3 * i))

const model = defineModel('ard-two', (m) => {
  const row = m.range('row')
  const coef = m.range('coef')
  const x = m.data('X', { dims: [row, coef] })
  const obs = m.data('y', { dims: [row] })
  const a = m.latent('a', d.gamma(shape, rate), { dims: [coef] })
  const z = m.latent('z', d.normal(0, 1), { dims: [coef] })
  const beta = m.derived('beta', z.div(a.sqrt()))
  m.observe('y', d.normal(x.matmul(beta), sigma), obs)
})

/** The exact posterior mean and standard deviation of the two coefficients, by a grid over the log relevances. */
function exact(): { mean: number[]; sd: number[] } {
  const grid = Array.from({ length: 140 }, (_, i) => -7 + (11 * i) / 139)
  const step = 11 / 139
  const XtX = [0, 0, 0, 0]
  const Xty = [0, 0]
  for (const [i, r] of X.entries()) {
    for (let p = 0; p < 2; p++) {
      Xty[p] = (Xty[p] as number) + (r[p] as number) * (y[i] as number)
      for (let q = 0; q < 2; q++) XtX[p * 2 + q] = (XtX[p * 2 + q] as number) + (r[p] as number) * (r[q] as number)
    }
  }
  const yty = y.reduce((s, v) => s + v * v, 0)
  const n = y.length
  let total = 0
  const firstMoment = [0, 0]
  const secondMoment = [0, 0]
  const lg = (a: number): number => shape * Math.log(rate) - lgamma(shape) + (shape - 1) * Math.log(a) - rate * a
  for (const u of grid)
    for (const v of grid) {
      const a = [Math.exp(u), Math.exp(v)]
      // The conditional: precision diag(a) + X'X / sigma^2, mean precision^-1 X'y / sigma^2; the evidence of the data up to a constant.
      const s2 = sigma * sigma
      const p00 = (a[0] as number) + (XtX[0] as number) / s2
      const p01 = (XtX[1] as number) / s2
      const p11 = (a[1] as number) + (XtX[3] as number) / s2
      const det = p00 * p11 - p01 * p01
      const h = [(Xty[0] as number) / s2, (Xty[1] as number) / s2]
      const mean = [
        (p11 * (h[0] as number) - p01 * (h[1] as number)) / det,
        (p00 * (h[1] as number) - p01 * (h[0] as number)) / det,
      ]
      const variance = [p11 / det, p00 / det]
      // log N(y; 0, s2 I + X diag(1/a) X') = -n/2 log(2 pi s2) - yty / (2 s2) + h' mean / 2 - log det(P) / 2 + (log a_1 + log a_2) / 2.
      const logEvidence =
        -0.5 * n * Math.log(2 * Math.PI * s2) -
        yty / (2 * s2) +
        0.5 * ((h[0] as number) * (mean[0] as number) + (h[1] as number) * (mean[1] as number)) -
        0.5 * Math.log(det) +
        0.5 * (u + v)
      // The density of the log relevances: the gamma density times the Jacobian a.
      const w = Math.exp(logEvidence + lg(a[0] as number) + u + lg(a[1] as number) + v) * step * step
      total += w
      for (let p = 0; p < 2; p++) {
        firstMoment[p] = (firstMoment[p] as number) + w * (mean[p] as number)
        secondMoment[p] = (secondMoment[p] as number) + w * ((variance[p] as number) + (mean[p] as number) ** 2)
      }
    }
  const m = firstMoment.map((s) => s / total)
  return { mean: m, sd: secondMoment.map((s, p) => Math.sqrt(s / total - (m[p] as number) ** 2)) }
}

function lgamma(x: number): number {
  // Shape 2: log Gamma(2) = 0; the test uses integer shapes only.
  let r = 0
  for (let k = 2; k < x; k++) r += Math.log(k)
  return r
}

const config = { damping: 0.3, tol: 1e-8, maxSweeps: 3000 } as never

function posterior(kind: 'ep' | 'vmp'): { mean: number[]; sd: number[]; status: string } {
  const bound = model.bind({ X, y })
  let p: Posterior | undefined
  if (kind === 'ep') {
    const method = epSpecMethod(bound)
    const s = session(method, { steps: 3000, mode: 'eager', config })
    p = method.posterior?.(s.state, s.config)
  } else {
    const method = vmpSpecMethod(bound)
    const s = session(method, { steps: 3000, mode: 'eager', config })
    p = method.posterior?.(s.state, s.config)
  }
  if (p === undefined) throw new Error('no posterior')
  return {
    mean: Array.from(toFlat(p.mean('beta') as never)),
    sd: Array.from(toFlat(p.stddev('beta') as never)),
    status: p.status,
  }
}

describe('a non-centred ARD regression against its exact posterior (two coefficients)', () => {
  const truth = exact()

  it('under expectation propagation: the means and the spreads of the coefficients (exact to 1e-3)', () => {
    const ep = posterior('ep')
    expect(ep.status).toBe('converged')
    for (let j = 0; j < 2; j++) {
      expect(Math.abs((ep.mean[j] as number) - (truth.mean[j] as number))).toBeLessThan(1e-3)
      expect((ep.sd[j] as number) / (truth.sd[j] as number)).toBeCloseTo(1, 3)
    }
  })

  it('under variational message passing: the means of the coefficients, the spread of the mean field (a quarter wider than the exact one for the large coefficient)', () => {
    const vmp = posterior('vmp')
    for (let j = 0; j < 2; j++) {
      expect(Math.abs((vmp.mean[j] as number) - (truth.mean[j] as number))).toBeLessThan(0.12)
      expect((vmp.sd[j] as number) / (truth.sd[j] as number)).toBeGreaterThan(0.9)
      expect((vmp.sd[j] as number) / (truth.sd[j] as number)).toBeLessThan(1.4)
    }
  })

  it('has the relevant coefficient large and the irrelevant one small, as the exact posterior has', () => {
    expect(Math.abs(truth.mean[0] as number)).toBeGreaterThan(1)
    expect(Math.abs(truth.mean[1] as number)).toBeLessThan(0.3)
  })
})
