import { describe, expect, it } from 'vitest'
import { maternKernel, rbfKernel, scaleKernel, theta } from '../kernels/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm, tracePolicy } from '../trace/index.ts'
import { epAlgorithm, epMarginal, laplaceAlgorithm, laplaceMarginal } from './approximations.ts'

const x = [[-1.4], [-0.9], [-0.3], [0.2], [0.6], [1.3], [1.8]]
const y = [0, 0, 1, 0, 1, 1, 1]
const kernel = scaleKernel(maternKernel({ lengthScale: 0.9, nu: 2.5 }), 1.7)

const normalPdf = (z: number): number => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)
const normalCdf = (z: number): number => {
  // Abramowitz and Stegun 26.2.17-like series through erf from a high-order Taylor/continued approximation.
  const t = 1 / (1 + 0.2316419 * Math.abs(z))
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
  const p = 1 - normalPdf(z) * poly
  return z >= 0 ? p : 1 - p
}

describe('the Laplace approximation', () => {
  it('has a gradient that matches finite differences, for both likelihoods', () => {
    for (const likelihood of ['logit', 'probit'] as const) {
      const t = Array.from(toFlat(theta(kernel)))
      const { gradient } = laplaceMarginal(kernel, x, y, { likelihood })
      t.forEach((_, j) => {
        const h = 1e-5
        const f = (shift: number): number =>
          laplaceMarginal(kernel, x, y, { likelihood, theta: t.map((v, i) => (i === j ? v + shift : v)) }).value
        expect(gradient[j]).toBeCloseTo((f(h) - f(-h)) / (2 * h), 5)
      })
    }
  })

  it('finds the mode by Newton steps that raise the likelihood, one per step', () => {
    const rec = recorder()
    const gram = toFlat(kernel.k(x))
    const alg = laplaceAlgorithm(
      Array.from({ length: 7 }, (_, i) => Array.from(gram.slice(i * 7, (i + 1) * 7))),
      y,
    )
    const r = runAlgorithm(alg, { maxSteps: 50, tracer: rec, policy: tracePolicy({ metrics: true }) })
    expect(r.status).toBe('converged')
    const lml = metricSeries(rec.run(), 'logMarginalLikelihood').values
    for (let i = 2; i < lml.length; i++) expect(lml[i] as number).toBeGreaterThanOrEqual((lml[i - 1] as number) - 1e-9)
    // At the mode f = K a and the gradient of the log posterior vanishes: a = d log p / d f.
    const f = Array.from(r.state.f)
    expect(f[0] as number).toBeLessThan(0)
    expect(f[6] as number).toBeGreaterThan(0)
  })

  it('refuses labels other than 0 and 1', () => {
    expect(() => laplaceAlgorithm([[1]], [2])).toThrow(/labels/)
  })
})

describe('expectation propagation', () => {
  it('is exact for one observation', () => {
    for (const k of [0.3, 2, 9]) {
      const r = runAlgorithm(epAlgorithm([[k]], [1]), { maxSteps: 30 })
      expect(Math.exp(r.state.logMarginalLikelihood)).toBeCloseTo(0.5, 8)
    }
  })

  it('is close to the exact marginal likelihood for two observations, closer than the Laplace approximation', () => {
    const K = [
      [1.5, 0.9],
      [0.9, 1.5],
    ]
    const labels = [1, 0]
    // The exact value by a product Gauss-Hermite-like grid over the two latent values.
    const n = 400
    const lim = 9
    const h = (2 * lim) / n
    const L11 = Math.sqrt(1.5)
    const L21 = 0.9 / L11
    const L22 = Math.sqrt(1.5 - L21 * L21)
    let exact = 0
    for (let i = 0; i < n; i++) {
      const z1 = -lim + (i + 0.5) * h
      for (let j = 0; j < n; j++) {
        const z2 = -lim + (j + 0.5) * h
        const f1 = L11 * z1
        const f2 = L21 * z1 + L22 * z2
        exact += normalPdf(z1) * normalPdf(z2) * normalCdf(f1) * normalCdf(-f2) * h * h
      }
    }
    const ep = runAlgorithm(epAlgorithm(K, labels), { maxSteps: 400 })
    expect(ep.status).toBe('converged')
    expect(ep.state.logMarginalLikelihood).toBeCloseTo(Math.log(exact), 2)
    const laplace = runAlgorithm(laplaceAlgorithm(K, labels, { likelihood: 'probit' }), { maxSteps: 100 })
    expect(Math.abs(ep.state.logMarginalLikelihood - Math.log(exact))).toBeLessThan(
      Math.abs(laplace.state.logMarginalLikelihood - Math.log(exact)),
    )
  })

  it('updates one site per step, with sites that become positive, and a sweep ends with the likelihood recorded', () => {
    const gram = toFlat(rbfKernel().k(x))
    const K = Array.from({ length: 7 }, (_, i) => Array.from(gram.slice(i * 7, (i + 1) * 7)))
    const alg = epAlgorithm(K, y)
    let s = alg.init({} as never)
    s = alg.step(s, {} as never)
    expect(s.site).toBe(1)
    expect(s.tau[0]).toBeGreaterThan(0)
    expect(s.tau[1]).toBe(0)
    expect(s.logMarginalLikelihood).toBe(Number.NEGATIVE_INFINITY)
    for (let i = 1; i < 7; i++) s = alg.step(s, {} as never)
    expect(s.sweeps).toBe(1)
    expect(Number.isFinite(s.logMarginalLikelihood)).toBe(true)
    expect(() => epAlgorithm(K, y, { damping: 1 })).toThrow(/damping/)
  })

  it('has a gradient that matches finite differences', () => {
    const t = Array.from(toFlat(theta(kernel)))
    const { gradient } = epMarginal(kernel, x, y)
    t.forEach((_, j) => {
      const h = 1e-5
      const f = (shift: number): number =>
        epMarginal(kernel, x, y, { theta: t.map((v, i) => (i === j ? v + shift : v)) }).value
      expect(gradient[j]).toBeCloseTo((f(h) - f(-h)) / (2 * h), 4)
    })
  })
})
