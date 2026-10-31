import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { lgamma } from '../numerics/special/index.ts'
import { item, tensor, toArray, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

const evidence = tensor([
  [2.5, 0.3, 0.1],
  [0.2, 4, 0.6],
])

describe('Dirichlet evidential classification', () => {
  for (const kind of ['mse', 'log', 'digamma'] as const) {
    it(`${kind} loss with the KL term passes the conformance check`, () =>
      expectConformant((e) => L.evidentialNll(e, [0, 2], { kind, klWeight: 0.3 }), [evidence]))
  }

  it('lowers the data term when the evidence moves to the right class', () => {
    for (const kind of ['mse', 'log', 'digamma'] as const) {
      const right = item(L.evidentialNll([[6, 0, 0]], [0], { kind }))
      const none = item(L.evidentialNll([[0, 0, 0]], [0], { kind }))
      const wrong = item(L.evidentialNll([[0, 6, 0]], [0], { kind }))
      expect(right, kind).toBeLessThan(none)
      expect(none, kind).toBeLessThan(wrong)
    }
  })

  it('has the closed-form expected squared error under the Dirichlet', () => {
    // alpha = [4, 2, 1], S = 7: sum (y - p)^2 + alpha (S - alpha) / (S^2 (S + 1)).
    const bias = (1 - 4 / 7) ** 2 + (2 / 7) ** 2 + (1 / 7) ** 2
    const variance = (4 * 3 + 2 * 5 + 1 * 6) / (49 * 8)
    expect(item(L.evidentialNll([[3, 1, 0]], [0]))).toBeCloseTo(bias + variance, 14)
  })

  it('charges the KL term only for evidence on the wrong classes', () => {
    const base = item(L.evidentialNll([[5, 0, 0]], [0]))
    expect(item(L.evidentialNll([[5, 0, 0]], [0], { klWeight: 1 }))).toBeCloseTo(base, 12)
    expect(item(L.evidentialNll([[5, 3, 0]], [0], { klWeight: 1 }))).toBeGreaterThan(
      item(L.evidentialNll([[5, 3, 0]], [0])),
    )
  })

  it('reports vacuity one without evidence and splits the entropy into aleatoric and epistemic parts', () => {
    const u = L.dirichletUncertainty([
      [0, 0, 0],
      [30, 0, 0],
      [10, 10, 10],
    ])
    const v = toArray(u.vacuity) as number[]
    expect(v[0]).toBe(1)
    expect(v[1]).toBeCloseTo(3 / 33, 14)
    const epistemic = toArray(u.epistemic) as number[]
    const aleatoric = toArray(u.aleatoric) as number[]
    // No evidence: the epistemic part is the largest; ample evidence of a confident class: both small; ample
    // evidence of a flat class: aleatoric dominates (close to log 3).
    expect(epistemic[0]).toBeGreaterThan(epistemic[2] ?? 0)
    expect(aleatoric[2]).toBeGreaterThan(aleatoric[1] ?? 1)
    expect(aleatoric[2]).toBeCloseTo(Math.log(3), 1)
    for (const [i, a] of aleatoric.entries()) expect(a + (epistemic[i] ?? 0)).toBeGreaterThanOrEqual(0)
  })
})

describe('deep evidential regression', () => {
  const [gamma, nu, alpha, beta, y] = [
    tensor([0.3, -0.5]),
    tensor([1.2, 0.7]),
    tensor([1.8, 2.5]),
    tensor([0.6, 1.1]),
    tensor([0.1, 0.4]),
  ]

  it('passes the conformance check in every parameter', () => {
    expectConformant(
      (g: Value, n: Value, a: Value, b: Value) => L.normalInverseGammaNll(g, n, a, b, y, { lambda: 0.2 }),
      [gamma, nu, alpha, beta],
    )
  })

  it('is the negative log-likelihood of a Student-t with 2 alpha degrees of freedom', () => {
    // St(mu = gamma, scale^2 = beta (1 + nu) / (nu alpha), df = 2 alpha).
    const t = (): number => {
      const [g, n, a, b, target] = [0.3, 1.2, 1.8, 0.6, 0.1]
      const df = 2 * a
      const scale2 = (b * (1 + n)) / (n * a)
      const z2 = (target - g) ** 2 / scale2
      const lgam = (x: number): number => item(lgamma(x))
      return -(
        lgam((df + 1) / 2) -
        lgam(df / 2) -
        0.5 * Math.log(df * Math.PI * scale2) -
        ((df + 1) / 2) * Math.log(1 + z2 / df)
      )
    }
    expect(item(L.normalInverseGammaNll([0.3], [1.2], [1.8], [0.6], [0.1]))).toBeCloseTo(t(), 13)
  })

  it('gives the aleatoric and epistemic variances of the prior', () => {
    const u = L.normalInverseGammaUncertainty([2], [3], [1])
    expect(item(u.aleatoric)).toBe(0.5)
    expect(item(u.epistemic)).toBe(0.25)
  })
})
