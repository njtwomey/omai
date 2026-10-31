import { describe, expect, it } from 'vitest'
import {
  betaRegression,
  type ExtendedFamily,
  gaussianLocationScale,
  hessIndex,
  multinomialLogit,
  negativeBinomialTheta,
  orderedCategorical,
  tweedieEstimated,
  zeroInflatedPoissonLogit,
} from './extended.ts'

const n = 6

/** Derivatives of the log-likelihood by central differences of the log-likelihood itself. */
function finiteDifferences(f: ExtendedFamily, y: Float64Array, w: Float64Array, eta: Float64Array[], aux: number[]) {
  const K = f.predictors.length
  const ll = (e: Float64Array[]): number => f.derivatives(y, w, e, aux, 2).logLik
  const grad = (e: Float64Array[], k: number, i: number, h: number): number => {
    const up = e.map((a) => Float64Array.from(a))
    const dn = e.map((a) => Float64Array.from(a))
    ;(up[k] as Float64Array)[i] = (up[k]?.[i] ?? 0) + h
    ;(dn[k] as Float64Array)[i] = (dn[k]?.[i] ?? 0) - h
    return (ll(up) - ll(dn)) / (2 * h)
  }
  const hess = (k: number, l: number, i: number): number => {
    const h = 1e-4
    const up = eta.map((a) => Float64Array.from(a))
    const dn = eta.map((a) => Float64Array.from(a))
    ;(up[l] as Float64Array)[i] = (up[l]?.[i] ?? 0) + h
    ;(dn[l] as Float64Array)[i] = (dn[l]?.[i] ?? 0) - h
    return (grad(up, k, i, 1e-5) - grad(dn, k, i, 1e-5)) / (2 * h)
  }
  return { K, grad: (k: number, i: number) => grad(eta, k, i, 1e-6), hess }
}

const cases: { name: string; family: ExtendedFamily; y: number[]; eta: number[][]; aux: number[] }[] = [
  {
    name: 'beta',
    family: betaRegression(),
    y: [0.1, 0.35, 0.5, 0.62, 0.8, 0.93],
    eta: [[-0.7, -0.3, 0.1, 0.4, 0.9, 1.2]],
    aux: [Math.log(8)],
  },
  {
    name: 'negative binomial',
    family: negativeBinomialTheta(),
    y: [0, 1, 3, 2, 7, 12],
    eta: [[-0.2, 0.3, 0.9, 0.7, 1.6, 2.2]],
    aux: [Math.log(2.5)],
  },
  {
    name: 'zero-inflated Poisson',
    family: zeroInflatedPoissonLogit(),
    y: [0, 0, 1, 2, 0, 5],
    eta: [
      [0.1, 0.3, 0.5, 0.7, 0.2, 1.4],
      [-0.5, 0.2, -1, -0.3, 0.8, -1.2],
    ],
    aux: [],
  },
  {
    name: 'ordered categorical',
    family: orderedCategorical(4),
    y: [0, 1, 2, 3, 1, 2],
    eta: [[-1.1, -0.2, 0.3, 1.4, 0.1, 0.9]],
    aux: [0.2, -0.3],
  },
  {
    name: 'multinomial',
    family: multinomialLogit(3),
    y: [0, 1, 2, 2, 1, 0],
    eta: [
      [0.3, -0.2, 0.8, 0.5, 0.1, -0.4],
      [-0.6, 0.4, 0.2, 1.1, -0.3, 0.7],
    ],
    aux: [],
  },
  {
    name: 'Tweedie',
    family: tweedieEstimated(),
    y: [0, 0.8, 0, 2.4, 1.1, 5.2],
    eta: [[-0.3, 0.1, -0.2, 0.7, 0.3, 1.4]],
    aux: [Math.log(1.3), 0.4],
  },
  {
    name: 'Gaussian location-scale',
    family: gaussianLocationScale(),
    y: [0.4, -1.1, 2.3, 0.9, -0.2, 1.7],
    eta: [
      [0.3, -0.5, 1.9, 0.2, 0.1, 1.2],
      [-0.2, 0.3, 0.5, -0.4, 0.1, 0.6],
    ],
    aux: [],
  },
]

describe('extended families: the gradient and Hessian of the log-likelihood in the predictors', () => {
  for (const c of cases) {
    it(`${c.name} equals finite differences of its log-likelihood`, () => {
      const y = Float64Array.from(c.y)
      const w = Float64Array.from({ length: n }, (_, i) => 1 + 0.25 * (i % 2))
      const eta = c.eta.map((a) => Float64Array.from(a))
      const d = c.family.derivatives(y, w, eta, c.aux, 2)
      expect(Number.isFinite(d.logLik)).toBe(true)
      const fd = finiteDifferences(c.family, y, w, eta, c.aux)
      for (let k = 0; k < fd.K; k++)
        for (let i = 0; i < n; i++) {
          expect(d.grad[k]?.[i] ?? Number.NaN, `grad ${k},${i}`).toBeCloseTo(fd.grad(k, i), 5)
        }
      for (let k = 0; k < fd.K; k++)
        for (let l = k; l < fd.K; l++)
          for (let i = 0; i < n; i++) {
            const analytic = d.hess[hessIndex(k, l, fd.K)]?.[i] ?? Number.NaN
            expect(analytic, `hess ${k}${l},${i}`).toBeCloseTo(fd.hess(k, l, i), 4)
          }
    })
  }

  it('the hand-written Gaussian location-scale derivatives equal those of the tensor version, third derivatives by differences', () => {
    const f = gaussianLocationScale()
    const y = Float64Array.from(cases[6]?.y ?? [])
    const w = new Float64Array(n).fill(1)
    const eta = (cases[6]?.eta ?? []).map((a) => Float64Array.from(a))
    const d = f.derivatives(y, w, eta, [], 3)
    expect(d.third).toBeDefined()
    const h = 1e-5
    for (let m = 0; m < 2; m++)
      for (let i = 0; i < n; i++) {
        const up = eta.map((a) => Float64Array.from(a))
        const dn = eta.map((a) => Float64Array.from(a))
        ;(up[m] as Float64Array)[i] = (up[m]?.[i] ?? 0) + h
        ;(dn[m] as Float64Array)[i] = (dn[m]?.[i] ?? 0) - h
        const a = f.derivatives(y, w, up, [], 2)
        const b = f.derivatives(y, w, dn, [], 2)
        for (let k = 0; k < 2; k++)
          for (let l = 0; l < 2; l++) {
            const fd = ((a.hess[hessIndex(k, l, 2)]?.[i] ?? 0) - (b.hess[hessIndex(k, l, 2)]?.[i] ?? 0)) / (2 * h)
            expect(d.third?.[(k * 2 + l) * 2 + m]?.[i] ?? Number.NaN, `third ${k}${l}${m},${i}`).toBeCloseTo(fd, 6)
          }
      }
  })

  it('reports an infeasible predictor as a log-likelihood of minus infinity', () => {
    const f = orderedCategorical(3)
    const d = f.derivatives(Float64Array.of(1), Float64Array.of(1), [Float64Array.of(Number.NaN)], [0], 2)
    expect(d.logLik).toBe(Number.NEGATIVE_INFINITY)
  })
})
