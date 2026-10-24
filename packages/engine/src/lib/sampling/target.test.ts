import { describe, expect, it } from 'vitest'
import { numericalGradient } from '../autodiff/index.ts'
import { bijectors, dirichlet, gamma, multivariateNormal, normal } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { item, mul, neg, square, sum, tensor, toFlat } from '../tensor/index.ts'
import {
  banana,
  nealsFunnel,
  numericTarget,
  proposal,
  proposalFromDistribution,
  target,
  targetFromDistribution,
  unconstrainedTarget,
} from './target.ts'

describe('target', () => {
  it('evaluates the log density and its autodiff gradient', () => {
    const t = target((x) => mul(-0.5, sum(square(x))), 3, { names: ['a', 'b', 'c'] })
    expect(t.logDensity([1, 2, 2])).toBe(-4.5)
    expect(Array.from(t.valueAndGrad([1, 2, 2]).grad)).toEqual([-1, -2, -2])
    expect(t.names).toEqual(['a', 'b', 'c'])
    expect(t.hasGradient).toBe(true)
  })

  it('reads NaN as -Infinity and a non-finite gradient as outside the support', () => {
    const t = numericTarget(() => Number.NaN, 1, { grad: () => [Number.NaN] })
    expect(t.logDensity([0])).toBe(Number.NEGATIVE_INFINITY)
    const r = t.valueAndGrad([0])
    expect(r.value).toBe(Number.NEGATIVE_INFINITY)
    expect(Array.from(r.grad)).toEqual([0])
    const u = numericTarget(() => 1, 1, { grad: () => [Number.POSITIVE_INFINITY] })
    expect(u.valueAndGrad([0]).value).toBe(Number.NEGATIVE_INFINITY)
  })

  it('checks dimensions, names and gradients', () => {
    expect(() => target((x) => sum(x), 0)).toThrow()
    expect(() => target((x) => sum(x), 2, { names: ['a'] })).toThrow()
    expect(() => target((x) => sum(x), 2).logDensity([1])).toThrow()
    const t = numericTarget((x) => x[0] ?? 0, 1)
    expect(t.hasGradient).toBe(false)
    expect(() => t.valueAndGrad([1])).toThrow()
    expect(() => t.logDensity([1, 2])).toThrow()
  })
})

describe('targetFromDistribution', () => {
  it('samples a gamma on the log scale with the change of variables', () => {
    const g = gamma(3, 2)
    const t = targetFromDistribution(g)
    const u = 0.4
    expect(t.logDensity([u])).toBeCloseTo(item(g.logProb(Math.exp(u))) + u, 12)
    expect(t.constrain([u])[0]).toBeCloseTo(Math.exp(u), 14)
    expect(t.unconstrain([Math.exp(u)])[0]).toBeCloseTo(u, 14)
    const grad = t.valueAndGrad([u]).grad[0] ?? 0
    const fd =
      toFlat(
        numericalGradient((x) => bijectors.transformLogDensity((y) => g.logProb(y), bijectors.exp())(x), [u]),
      )[0] ?? 0
    expect(grad).toBeCloseTo(fd, 6)
  })

  it('maps a Dirichlet onto R^(K-1) and back to the simplex', () => {
    const d = dirichlet([2, 3, 4])
    const t = targetFromDistribution(d)
    expect(t.dim).toBe(2)
    const y = t.constrain([0.3, -0.2])
    expect(y.length).toBe(3)
    expect(y.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14)
    const back = t.unconstrain(y)
    expect(back[0]).toBeCloseTo(0.3, 12)
    expect(back[1]).toBeCloseTo(-0.2, 12)
    expect(t.constraints?.[0]?.kind).toBe('simplex')
  })

  it('keeps the raw density when asked, and refuses batched distributions', () => {
    const m = multivariateNormal([0, 1], {
      covariance: [
        [1, 0.5],
        [0.5, 2],
      ],
    })
    const t = targetFromDistribution(m, { unconstrained: false })
    expect(t.dim).toBe(2)
    expect(t.logDensity([0.1, 0.2])).toBeCloseTo(item(m.logProb([0.1, 0.2])), 12)
    expect(() => targetFromDistribution(normal([0, 1], 1))).toThrow()
  })
})

describe('unconstrainedTarget', () => {
  it('gives each coordinate its own bijector and adds the log-Jacobians', () => {
    // y0 ~ Exp(1) on (0, inf) and y1 ~ U(0, 2) on (0, 2).
    const t = unconstrainedTarget(
      (y) => neg(sum(mul(y, [1, 0]))),
      [{ kind: 'positive' }, { kind: 'interval', low: tensor(0), high: tensor(2) }],
      2,
    )
    const u = [0.5, -0.3]
    const s = 1 / (1 + Math.exp(0.3))
    const expected = -Math.exp(0.5) + 0.5 + Math.log(2 * s * (1 - s))
    expect(t.logDensity(u)).toBeCloseTo(expected, 12)
    const y = t.constrain(u)
    expect(y[0]).toBeCloseTo(Math.exp(0.5), 14)
    expect(y[1]).toBeCloseTo(2 * s, 14)
    expect(t.unconstrain(y)[1]).toBeCloseTo(-0.3, 12)
  })

  it('accepts one bijector for the whole vector, and rejects non-elementwise supports', () => {
    const t = unconstrainedTarget((y) => neg(sum(y)), bijectors.exp(), 2)
    expect(t.logDensity([0, 0])).toBeCloseTo(-2, 14)
    expect(() => unconstrainedTarget((y) => sum(y), [{ kind: 'simplex' }], 1)).toThrow()
    expect(() => unconstrainedTarget((y) => sum(y), [{ kind: 'real' }], 2)).toThrow()
  })
})

describe('standard targets', () => {
  it("evaluates Neal's funnel in both forms", () => {
    const f = nealsFunnel(3, { scale: 2 })
    const [v, a, b] = [0.4, -0.7, 1.2]
    const logN = (x: number, s2: number): number => -0.5 * Math.log(2 * Math.PI * s2) - (0.5 * x * x) / s2
    expect(f.logDensity([v, a, b])).toBeCloseTo(logN(v, 4) + logN(a, Math.exp(v)) + logN(b, Math.exp(v)), 12)
    expect(f.names).toEqual(['v', 'x[1]', 'x[2]'])
    const g = nealsFunnel(3, { nonCentred: true, scale: 2 })
    const y = g.constrain([0.2, 1, -1])
    expect(y[0]).toBeCloseTo(0.4, 14)
    expect(y[1]).toBeCloseTo(Math.exp(0.2), 14)
    expect(Array.from(g.unconstrain(y)).map((x) => Math.round(x * 1e12) / 1e12)).toEqual([0.2, 1, -1])
    expect(() => nealsFunnel(1)).toThrow()
  })

  it("gives the banana's gradient by autodiff (checked against finite differences)", () => {
    const b = banana({ curvature: 0.05, scale: 3 })
    const x = [1.3, -0.4]
    const h = 1e-6
    const g = b.valueAndGrad(x).grad
    for (let i = 0; i < 2; i++) {
      const up = [...x]
      const down = [...x]
      up[i] = (up[i] ?? 0) + h
      down[i] = (down[i] ?? 0) - h
      expect(g[i]).toBeCloseTo((b.logDensity(up) - b.logDensity(down)) / (2 * h), 6)
    }
  })
})

describe('proposals', () => {
  it('batch draws and densities consistently', () => {
    const q = proposalFromDistribution(
      multivariateNormal([0, 0], {
        covariance: [
          [1, 0],
          [0, 4],
        ],
      }),
    )
    const xs = q.sampleMany(key(0), 5)
    expect(xs.length).toBe(10)
    const many = q.logProbMany(xs)
    expect(many[2]).toBeCloseTo(q.logProb(xs.subarray(4, 6)), 12)
    expect(q.sample(key(1)).length).toBe(2)
    expect(() => proposalFromDistribution(normal([0, 1], 1))).toThrow()
  })

  it('wraps plain functions, checking draws', () => {
    const q = proposal({ dim: 2, sample: () => [1, 2], logProb: () => Number.NaN })
    expect(q.logProb([0, 0])).toBe(Number.NEGATIVE_INFINITY)
    expect(Array.from(q.logProbMany(new Float64Array(4)))).toEqual([Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY])
    expect(() => proposal({ dim: 2, sample: () => [1], logProb: () => 0 }).sample(key(0))).toThrow()
  })
})
