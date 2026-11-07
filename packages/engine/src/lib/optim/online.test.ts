import { describe, expect, it } from 'vitest'
import { maxOf, toRows } from '../base/index.ts'
import { key, uniform } from '../random/index.ts'
import { square, sub, sum, type Tensor, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import { metricSeries, recorder } from '../trace/index.ts'
import * as O from './index.ts'

/** The plays of a run as rows of plain numbers. */
const playRows = (r: O.OnlineResult): number[][] => toRows(toFlat(r.plays), r.plays.shape[0] ?? 0, r.plays.shape[1] ?? 0)

const T = 200
const n = 5
const lossMatrix = uniform(key(3), [T, n]) as Tensor
// Expert 2 is better on average.
const biased = toArray(lossMatrix) as number[][]
for (const row of biased) row[2] = (row[2] ?? 0) * 0.5

describe('prediction with expert advice', () => {
  it('Hedge keeps its regret below ln n / eta + eta T / 8', () => {
    const eta = Math.sqrt((8 * Math.log(n)) / T)
    const r = O.runOnline(O.hedge({ learningRate: eta }), O.linearLosses(biased), { dim: n, experts: true })
    const bound = Math.log(n) / eta + (eta * T) / 8
    expect(maxOf(toFlat(r.regret))).toBeLessThanOrEqual(bound)
    expect(toFlat(r.regret).at(-1)).toBeCloseTo(r.cumulativeLoss - r.comparatorLoss, 12)
    // the anytime rate also has sublinear regret
    const anytime = O.runOnline(O.hedge(), O.linearLosses(biased), { dim: n, experts: true })
    expect((toFlat(anytime.regret).at(-1) ?? Infinity) / T).toBeLessThan(0.15)
  })

  it('Hedge and exponentiated gradient agree on linear losses with a constant rate', () => {
    const a = O.runOnline(O.hedge({ learningRate: 0.3 }), O.linearLosses(biased.slice(0, 30)), { dim: n })
    const b = O.runOnline(O.exponentiatedGradient({ learningRate: 0.3 }), O.linearLosses(biased.slice(0, 30)), {
      dim: n,
    })
    const other = playRows(b)
    for (const [t, p] of playRows(a).entries())
      for (const [i, v] of p.entries()) expect(v).toBeCloseTo(other[t]?.[i] ?? 0, 12)
  })

  it('records the regret in the trace', () => {
    const rec = recorder()
    O.runOnline(O.hedge({ learningRate: 0.5 }), O.linearLosses(biased.slice(0, 10)), {
      dim: n,
      experts: true,
      tracer: rec,
    })
    expect(metricSeries(rec.run(), 'regret').values).toHaveLength(11)
  })
})

describe('online convex optimisation', () => {
  // f_t(x) = |x - z_t|^2 with targets z_t in [-1, 1]^2; the best fixed point is their mean.
  const zs = toArray(uniform(key(9), [T, 2], { low: -1, high: 1 })) as number[][]
  const losses: O.OnlineLoss[] = zs.map((z) => (x: Value) => sum(square(sub(x, tensor(z)))))
  const mean = [0, 1].map((k) => zs.reduce((a, z) => a + (z[k] ?? 0), 0) / T)

  it('online gradient descent meets its regret bound against the best fixed point', () => {
    // D = 2 sqrt 2 (diameter of the ball of radius sqrt 2), G <= 2 (|x| + |z|) <= 4 sqrt 2.
    const D = 2 * Math.SQRT2
    const G = 4 * Math.SQRT2
    const eta = D / (G * Math.sqrt(T))
    const r = O.runOnline(O.onlineGradientDescent({ learningRate: eta, radius: Math.SQRT2 }), losses, {
      dim: 2,
      comparator: mean,
    })
    expect(toFlat(r.regret).at(-1) ?? Infinity).toBeLessThanOrEqual((D * D) / (2 * eta) + (eta * G * G * T) / 2)
    for (const p of playRows(r)) expect(Math.hypot(p[0] ?? 0, p[1] ?? 0)).toBeLessThanOrEqual(Math.SQRT2 + 1e-12)
  })

  it('online Newton step has logarithmic-order regret on exp-concave losses', () => {
    const r = O.runOnline(O.onlineNewtonStep({ gamma: 0.5, epsilon: 1, radius: Math.SQRT2 }), losses, {
      dim: 2,
      comparator: mean,
    })
    expect(toFlat(r.regret).at(-1) ?? Infinity).toBeLessThan(10 * Math.log(T))
    for (const p of playRows(r)) expect(Math.hypot(p[0] ?? 0, p[1] ?? 0)).toBeLessThanOrEqual(Math.SQRT2 + 1e-9)
  })

  it('FTRL with a quadratic regulariser plays the scaled negative gradient sum', () => {
    const l = O.ftrl({ learningRate: 0.25 })
    let s = l.init(2)
    s = l.update(s, new Float64Array([1, 2]), 1)
    s = l.update(s, new Float64Array([3, -2]), 2)
    expect(Array.from(l.play(s))).toEqual([-1, 0])
    const r = O.runOnline(O.ftrl({ learningRate: (t) => 0.5 / Math.sqrt(t), radius: Math.SQRT2 }), losses, {
      dim: 2,
      comparator: mean,
    })
    expect((toFlat(r.regret).at(-1) ?? Infinity) / T).toBeLessThan(0.1)
  })

  it('exponentiated gradient optimises a convex loss over the simplex', () => {
    const target = tensor([0.2, 0.5, 0.3])
    const f: O.OnlineLoss = (x) => sum(square(sub(x, target)))
    const r = O.runOnline(O.exponentiatedGradient({ learningRate: 0.5 }), Array(300).fill(f), { dim: 3 })
    const last = playRows(r).at(-1) ?? [0, 0, 0]
    for (const [i, v] of [0.2, 0.5, 0.3].entries()) expect(last[i]).toBeCloseTo(v, 4)
    expect(Number.isNaN(toFlat(r.regret)[0])).toBe(true)
  })
})
