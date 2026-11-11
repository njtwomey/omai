// A hurdle count (a zero with probability `zero`, else a zero-truncated Poisson of `exp(b0 + b1 x)`): the family's own mass function,
// and EP on the compiled model against the exact posterior. The hurdle's zero part and positive part share no parameter, so the exact
// posterior of the zero probability is a Beta of the counts, and that of the two coefficients a two-dimensional grid of the truncated
// Poisson likelihood over the positive counts.
import { describe, expect, it } from 'vitest'
import { lgamma } from '../../numerics/special/index.ts'
import { item, tensor } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { epSpecMethod } from '../inference/index.ts'
import { analysisOf } from '../inference/model-input.ts'
import { distributionFamilies } from './dists/families.ts'
import * as d from './dists/index.ts'
import { defineModel, ops } from './index.ts'

const X = [-1, -0.6, -0.2, 0.1, 0.5, 0.9, 1.3, -0.8, 0.3, 1.1, -0.4, 0.7]
const Y = [0, 0, 1, 0, 2, 4, 6, 1, 3, 0, 1, 5]
const PRIOR_SD = 1.5

const model = defineModel('hurdle', (m) => {
  const row = m.range('row')
  const x = m.data('x', { dims: [row] })
  const y = m.data('y', { dims: [row], role: 'response' })
  const pi = m.latent('pi', d.beta(1, 1))
  const b0 = m.latent('b0', d.normal(0, PRIOR_SD))
  const b1 = m.latent('b1', d.normal(0, PRIOR_SD))
  m.observe('y', d.hurdlePoisson(ops.exp(ops.add(b0, ops.mul(b1, x))), pi), y)
})

describe('the hurdle Poisson family', () => {
  it('has the mass function of a zero with probability `zero` and a zero-truncated Poisson otherwise', () => {
    const dist = distributionFamilies()['hurdlePoisson']?.make({ rate: tensor(2), zero: tensor(0.3) })
    const rate = 2
    const lf = (k: number): number => item(lgamma(k + 1))
    expect(Number(item(dist?.prob(0) as never))).toBeCloseTo(0.3, 12)
    for (const k of [1, 2, 5])
      expect(Number(item(dist?.prob(k) as never))).toBeCloseTo(
        (0.7 * Math.exp(-rate + k * Math.log(rate) - lf(k))) / (1 - Math.exp(-rate)),
        10,
      )
  })
})

describe('EP on a hurdle regression', () => {
  const bound = model.bind({ x: X, y: Y })
  const method = epSpecMethod(bound)

  it('is exact in the zero probability and close on the coefficients: against the Beta and the grid', () => {
    expect(method.check?.(analysisOf(bound)).applies).toBe(true)
    const run = session(method, { steps: 400, mode: 'eager', config: { tol: 1e-10 } })
    const post = method.posterior?.(run.state, run.config)
    if (post === undefined) throw new Error('no posterior')
    // The zero probability: Beta(1 + zeros, 1 + positives).
    const zeros = Y.filter((v) => v === 0).length
    const a = 1 + zeros
    const b = 1 + (Y.length - zeros)
    const row = post.summary().find((r) => r.name === 'pi')
    expect(row?.mean).toBeCloseTo(a / (a + b), 6)
    expect(row?.sd).toBeCloseTo(Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1))), 6)
    // The coefficients: a grid of the truncated Poisson likelihood of the positive counts.
    const n = 241
    const lo = -3
    const hi = 3
    const w: number[] = []
    const pts: [number, number][] = []
    let top = Number.NEGATIVE_INFINITY
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const [b0, b1] = [lo + ((hi - lo) * i) / (n - 1), lo + ((hi - lo) * j) / (n - 1)]
        let lp = -(b0 * b0 + b1 * b1) / (2 * PRIOR_SD ** 2)
        for (const [r, y] of Y.entries()) {
          if (y === 0) continue
          const mu = Math.exp(b0 + b1 * (X[r] as number))
          lp += y * Math.log(mu) - mu - Math.log(-Math.expm1(-mu))
        }
        w.push(lp)
        pts.push([b0, b1])
        top = Math.max(top, lp)
      }
    const mass = w.map((v) => Math.exp(v - top))
    const total = mass.reduce((s, v) => s + v, 0)
    for (const [k, name] of ['b0', 'b1'].entries()) {
      const mean = mass.reduce((s, v, g) => s + (v * ((pts[g] as number[])[k] as number)) / total, 0)
      const sd = Math.sqrt(
        mass.reduce((s, v, g) => s + (v * (((pts[g] as number[])[k] as number) - mean) ** 2) / total, 0),
      )
      const r = post.summary().find((q) => q.name === name)
      expect(Math.abs((r?.mean ?? Number.NaN) - mean) / sd).toBeLessThan(0.1)
      expect(Math.abs((r?.sd ?? Number.NaN) / sd - 1)).toBeLessThan(0.1)
    }
  })
})
