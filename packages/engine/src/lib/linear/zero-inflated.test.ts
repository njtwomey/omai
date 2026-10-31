import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  type CountFamily,
  type CountModelKind,
  countModelAlgorithm,
  countModelObjective,
  countProblemOf,
  hurdle,
  zeroInflated,
} from './zero-inflated.ts'

const n = 80
const x = Array.from({ length: n }, (_, i) => [Math.sin(i * 1.3) * 1.2, Math.cos(i * 0.7)])
const y = x.map((r, i) => {
  if (Math.sin(i * 12.9) > 0.35) return 0
  const mu = Math.exp(0.7 + 0.4 * (r[0] ?? 0))
  return Math.max(0, Math.round(mu + 1.5 * Math.sin(i * 7.7)))
})

describe('count model likelihood', () => {
  for (const kind of ['zero-inflated', 'hurdle'] as CountModelKind[])
    for (const family of ['poisson', 'negative-binomial'] as CountFamily[])
      it(`has the gradient of its finite differences (${kind}, ${family})`, () => {
        const problem = countProblemOf({ x, y }, [0], 'test')
        const theta = [0.3, -0.2, 0.5, -0.4, 0.6, ...(family === 'negative-binomial' ? [-0.3] : [])]
        const e = countModelObjective(problem, kind, family, theta)
        const h = 1e-6
        theta.forEach((_, j) => {
          const up = [...theta]
          const dn = [...theta]
          up[j] = (up[j] ?? 0) + h
          dn[j] = (dn[j] ?? 0) - h
          const fd =
            (countModelObjective(problem, kind, family, up).objective -
              countModelObjective(problem, kind, family, dn).objective) /
            (2 * h)
          expect(e.gradient[j]).toBeCloseTo(fd, 4)
        })
      })

  it('rejects counts that are not non-negative integers', () => {
    expect(() => countProblemOf({ x: [[1], [2]], y: [1, -1] }, 'constant', 'test')).toThrow(/counts/)
    expect(() => countProblemOf({ x: [[1], [2]], y: [1, 1.5] }, 'constant', 'test')).toThrow(/counts/)
  })
})

describe('zero-inflated and hurdle estimators', () => {
  it('fits a hurdle model whose zero part is the share of zeros and whose count part solves the truncated score equation', () => {
    const f = hurdle({ tol: 1e-10 }).fit({ x, y })
    expect(f.params.converged).toBe(true)
    const share = y.filter((v) => v === 0).length / n
    expect(1 / (1 + Math.exp(-f.params.zeroIntercept))).toBeCloseTo(share, 8)
    // Intercept score of the zero-truncated Poisson: sum over positive counts of y - mu / (1 - exp(-mu)) = 0.
    const mu = toFlat(f.predictCount(x))
    let score = 0
    y.forEach((v, i) => {
      if (v > 0) score += v - (mu[i] ?? 0) / (1 - Math.exp(-(mu[i] ?? 0)))
    })
    expect(Math.abs(score)).toBeLessThan(1e-6)
    // The hurdle's probability of a zero is the zero part alone.
    expect(toFlat(f.predictZero([[0, 0]]))[0]).toBeCloseTo(share, 8)
  })

  it('has a zero-inflated zero probability above the count part and a mean below it', () => {
    const f = zeroInflated().fit({ x, y })
    expect(f.params.converged).toBe(true)
    const pi = toFlat(f.predictInflation([[0.5, 0.5]]))[0] ?? 0
    const mu = toFlat(f.predictCount([[0.5, 0.5]]))[0] ?? 0
    expect(toFlat(f.predictZero([[0.5, 0.5]]))[0]).toBeCloseTo(pi + (1 - pi) * Math.exp(-mu), 12)
    expect(toFlat(f.predict([[0.5, 0.5]]))[0]).toBeCloseTo((1 - pi) * mu, 12)
    expect(f.params.alpha).toBe(0)
    expect(f.params.covariance.length).toBe(4)
  })

  it('adds a dispersion that does not lower the likelihood, and lets the zero part depend on features', () => {
    const zi = zeroInflated({ inflation: 'all' }).fit({ x, y })
    const nb = zeroInflated({ family: 'negative-binomial' }).fit({ x, y })
    const poisson = zeroInflated().fit({ x, y })
    expect(nb.params.alpha).toBeGreaterThan(0)
    expect(zi.params.zeroCoef).toHaveLength(2)
    // Poisson is the limit alpha -> 0 of the negative binomial, so the likelihood can only be matched or improved.
    expect(nb.params.logLikelihood).toBeGreaterThan(poisson.params.logLikelihood - 1e-3)
    expect(zi.params.logLikelihood).toBeGreaterThan(poisson.params.logLikelihood - 1e-9)
  })

  it('treats integer weights as repeated rows', () => {
    const w = x.map((_, i) => (i % 3 === 0 ? 2 : 1))
    const a = zeroInflated({ tol: 1e-10 }).fit({ x, y, weights: w })
    const b = zeroInflated({ tol: 1e-10 }).fit({
      x: x.flatMap((r, i) => (w[i] === 2 ? [r, r] : [r])),
      y: y.flatMap((v, i) => (w[i] === 2 ? [v, v] : [v])),
    })
    expect(a.params.intercept).toBeCloseTo(b.params.intercept, 6)
    expect(a.params.logLikelihood).toBeCloseTo(b.params.logLikelihood, 6)
  })

  it('resumes from a serialised state and reports Wald tables', () => {
    const alg = countModelAlgorithm(countProblemOf({ x, y }, 'constant', 'test'))
    const full = runAlgorithm(alg, { maxSteps: 4 })
    const first = runAlgorithm(alg, { maxSteps: 2 })
    const rest = runAlgorithm(alg, {
      maxSteps: 2,
      start: { state: alg.serialise?.(first.state), step: first.steps, serialised: true },
    })
    expect(rest.state.theta).toEqual(full.state.theta)
    const table = zeroInflated().fit({ x, y }).exposition[0]
    expect(table?.kind).toBe('coefficientTable')
    if (table?.kind === 'coefficientTable')
      expect(table.names).toEqual(['count:x0', 'count:x1', 'count:intercept', 'zero:intercept'])
  })
})
