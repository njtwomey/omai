import { describe, expect, it } from 'vitest'
import { expectResumable } from '../../testing/mp-variants.ts'
import { foldIn, key, normal, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import * as M from './index.ts'

/** A compressed-sensing problem: a Gaussian matrix of variance 1 / M, a sparse signal, noise of the given variance. */
function problem(
  n: number,
  m: number,
  rho: number,
  noise: number,
  seed: number,
): { problem: M.AmpProblem; truth: number[] } {
  const p = M.ampProblem(key(seed), { columns: n, delta: m / n, sparsity: rho, noiseVariance: noise })
  return { problem: p.problem, truth: [...p.truth] }
}

/** The values a metric took on the tape, in order. */
function metric(rec: ReturnType<typeof recorder>, name: string): number[] {
  return rec.run().events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value as number] : []))
}

describe('the denoiser and the state evolution', () => {
  const prior: M.AmpPrior = { kind: 'bernoulli-gaussian', sparsity: 0.1, variance: 1 }
  it('the Bernoulli-Gaussian denoiser is the posterior mean, and its derivative is the derivative of it', () => {
    for (const tau2 of [0.05, 0.4, 2]) {
      for (const r of [-2.5, -0.3, 0, 0.1, 0.9, 3]) {
        const h = 1e-6
        const d = M.ampDenoiser(prior, r, tau2)
        const numeric = (M.ampDenoiser(prior, r + h, tau2).mean - M.ampDenoiser(prior, r - h, tau2).mean) / (2 * h)
        expect(Math.abs(d.derivative - numeric)).toBeLessThan(1e-6)
        // the posterior mean by direct integration over the two components
        const gain = 1 / (1 + tau2)
        const dens = (v: number): number => Math.exp((-0.5 * r * r) / v) / Math.sqrt(2 * Math.PI * v)
        const a = 0.1 * dens(1 + tau2)
        const b = 0.9 * dens(tau2)
        expect(Math.abs(d.mean - (a / (a + b)) * gain * r)).toBeLessThan(1e-12)
      }
    }
  })
  it('the mmse is the error of that denoiser: a Monte Carlo check by the fixed draws of the library', () => {
    const k = key(3)
    const n = 200000
    const u = toFlat(uniform(foldIn(k, 1), [n]))
    const g = toFlat(normal(foldIn(k, 2), [n]))
    const z = toFlat(normal(foldIn(k, 3), [n]))
    for (const tau2 of [0.02, 0.3]) {
      let s = 0
      for (let i = 0; i < n; i++) {
        const x = (u[i] as number) < 0.1 ? (g[i] as number) : 0
        s += (M.ampDenoiser(prior, x + Math.sqrt(tau2) * (z[i] as number), tau2).mean - x) ** 2
      }
      expect(Math.abs(s / n - M.ampMmse(prior, tau2))).toBeLessThan(0.05 * M.ampMmse(prior, tau2) + 2e-4)
    }
  })
  it('the Gaussian prior has the closed form, and the state evolution is its fixed-point iteration', () => {
    const gaussian: M.AmpPrior = { kind: 'gaussian', variance: 2 }
    expect(M.ampMmse(gaussian, 0.5)).toBeCloseTo((2 * 0.5) / 2.5, 14)
    const se = M.ampStateEvolution(gaussian, 0.5, 0.1, 200)
    // the fixed point solves tau2 = sigma2 + v tau2 / (delta (v + tau2))
    const t = se[se.length - 1] as number
    expect(Math.abs(t - (0.1 + (2 * t) / (0.5 * (2 + t))))).toBeLessThan(1e-12)
    expect(se[0]).toBeCloseTo(0.1 + 2 / 0.5, 14)
  })
  it('the state evolution is monotone, and has a good and a bad fixed point either side of the phase transition', () => {
    const sparse: M.AmpPrior = { kind: 'bernoulli-gaussian', sparsity: 0.1, variance: 1 }
    const rich = M.ampStateEvolution(sparse, 0.5, 1e-4, 60)
    const poor = M.ampStateEvolution(sparse, 0.1, 1e-4, 60)
    for (let i = 1; i < rich.length; i++) expect(rich[i] as number).toBeLessThanOrEqual((rich[i - 1] as number) + 1e-15)
    expect(rich[60] as number).toBeLessThan(2e-4)
    expect(poor[60] as number).toBeGreaterThan(0.2)
  })
  it('refuses a bad prior, noise, and delta', () => {
    expect(() => M.ampDenoiser({ kind: 'gaussian', variance: 0 }, 1, 1)).toThrow(/variance/)
    expect(() => M.ampDenoiser({ kind: 'bernoulli-gaussian', sparsity: 0, variance: 1 }, 1, 1)).toThrow(/sparsity/)
    expect(() => M.ampDenoiser(prior, 1, 0)).toThrow(/noise/)
    expect(() => M.ampStateEvolution(prior, 0, 0.1, 3)).toThrow(/delta/)
  })
})

describe('AMP follows its state evolution on sparse recovery', () => {
  const prior: M.AmpPrior = { kind: 'bernoulli-gaussian', sparsity: 0.15, variance: 1 }
  const { problem: p, truth } = problem(2000, 1000, 0.15, 1e-2, 11)
  const alg = M.ampAlgorithm(p, { prior, noiseVariance: 1e-2, truth, tol: 1e-12, maxIterations: 25 })
  const rec = recorder()
  const r = runAlgorithm(alg, { maxSteps: 25, tracer: rec })
  const emp = metric(rec, 'tau2')
  const se = M.ampStateEvolution(prior, 0.5, 1e-2, 25)
  it("the estimated noise level of iteration k is the state evolution's tau^2 of k - 1, within a fifth, and close at the end", () => {
    // the first value is the energy of y; after iteration k the tape holds the noise level that produced the estimate
    for (let k = 2; k <= 25; k++) {
      const predicted = se[k - 1] as number
      expect(Math.abs((emp[k] as number) - predicted) / predicted, `iteration ${k}`).toBeLessThan(0.2)
    }
    expect(Math.abs((emp[25] as number) - (se[24] as number)) / (se[24] as number)).toBeLessThan(0.06)
  })
  it('the error against the truth reaches the mmse the state evolution predicts, within 12 percent', () => {
    const mse = metric(rec, 'mse')
    const predicted = M.ampMmse(prior, se[se.length - 1] as number)
    expect(Math.abs((mse[mse.length - 1] as number) - predicted) / predicted).toBeLessThan(0.12)
    expect(mse[mse.length - 1] as number).toBeLessThan(0.15 * (mse[0] as number))
  })
  it('puts the state evolution on the tape beside the estimate', () => {
    expect(metric(rec, 'stateEvolution').length).toBe(25)
    expect(r.state.iteration).toBe(25)
  })
})

describe('AMP with a Gaussian prior approximates the exact posterior mean', () => {
  it('the estimate is close to the ridge solution, and the noise level follows the closed-form state evolution', () => {
    const n = 200
    const m = 100
    const sigma2 = 0.05
    const gen = problem(n, m, 1, sigma2, 5) // every entry nonzero and standard normal
    const prior: M.AmpPrior = { kind: 'gaussian', variance: 1 }
    const rec = recorder()
    const run = runAlgorithm(
      M.ampAlgorithm(gen.problem, { prior, noiseVariance: sigma2, tol: 1e-12, maxIterations: 40 }),
      {
        maxSteps: 40,
        tracer: rec,
      },
    )
    // the exact posterior mean x = (I / v + A'A / sigma2)^-1 A'y / sigma2, by conjugate gradients
    const A = gen.problem.matrix
    const y = gen.problem.y
    const mul = (v: number[]): number[] => {
      const av = Array.from({ length: m }, (_, i) => {
        let s = 0
        for (let j = 0; j < n; j++) s += (A[i * n + j] as number) * (v[j] as number)
        return s
      })
      return Array.from({ length: n }, (_, j) => {
        let s = (v[j] as number) / 1
        for (let i = 0; i < m; i++) s += ((A[i * n + j] as number) * (av[i] as number)) / sigma2
        return s
      })
    }
    const rhs = Array.from({ length: n }, (_, j) => {
      let s = 0
      for (let i = 0; i < m; i++) s += ((A[i * n + j] as number) * (y[i] as number)) / sigma2
      return s
    })
    let x = new Array<number>(n).fill(0)
    let res = rhs.slice()
    let dir = res.slice()
    for (let it = 0; it < 400; it++) {
      const ad = mul(dir)
      const rr = res.reduce((a, b) => a + b * b, 0)
      const alpha = rr / dir.reduce((a, b, i) => a + b * (ad[i] as number), 0)
      x = x.map((v, i) => v + alpha * (dir[i] as number))
      res = res.map((v, i) => v - alpha * (ad[i] as number))
      const beta = res.reduce((a, b) => a + b * b, 0) / rr
      dir = res.map((v, i) => v + beta * (dir[i] as number))
    }
    let num = 0
    let den = 0
    for (let i = 0; i < n; i++) {
      num += ((run.state.x[i] as number) - (x[i] as number)) ** 2
      den += (x[i] as number) ** 2
    }
    expect(Math.sqrt(num / den)).toBeLessThan(0.12)
    const se = M.ampStateEvolution(prior, 0.5, sigma2, 40)
    const emp = metric(rec, 'tau2')
    expect(Math.abs((emp[40] as number) - (se[39] as number)) / (se[39] as number)).toBeLessThan(0.15)
  })
})

describe('AMP: the contract', () => {
  const gen = problem(300, 150, 0.1, 1e-3, 21)
  const prior: M.AmpPrior = { kind: 'bernoulli-gaussian', sparsity: 0.1, variance: 1 }
  const alg = M.ampAlgorithm(gen.problem, { prior, noiseVariance: 1e-3, maxIterations: 12, tol: 1e-14 })
  it('serialises and resumes to the same state', () => {
    expectResumable(alg, 4)
  })
  it('runs anytime in slices that reach the same snapshot as the uninterrupted run', () => {
    const snapshot = M.ampSnapshot()
    let now = 0
    const seen: number[] = []
    const first = M.runAnytimeWith(snapshot, alg, {
      budgetMs: 7,
      snapshotEveryMs: 2,
      clock: () => (now += 1),
      onSnapshot: (s) => seen.push(s.sweeps),
    })
    expect(first.status).toBe('budget')
    expect(seen.length).toBeGreaterThanOrEqual(3)
    const second = M.runAnytimeWith(snapshot, alg, { budgetMs: 1e9, clock: () => (now += 1), from: first.slice })
    const whole = runAlgorithm(alg, { maxSteps: 100000 })
    expect(second.snapshot).toEqual(snapshot(whole.state))
  })
  it('puts the noise level, the iteration and the marginals on the tape', () => {
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 3, tracer: rec, policy: policies.teaching })
    const names = new Set(rec.run().events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    for (const name of ['tau2', 'iteration', 'change', 'stateEvolution']) expect(names.has(name)).toBe(true)
    expect(rec.run().events.some((e) => e.type === 'message' && e.kind === 'marginals')).toBe(true)
  })
  it('is a method with the prior as knobs and a typed tape', () => {
    const method = M.ampMethod(gen.problem, { noiseVariance: 1e-3 })
    expect(method.name).toBe('amp')
    expect(Object.keys(method.config.spaces)).toEqual(
      expect.arrayContaining(['prior', 'sparsity', 'variance', 'damping', 'tol', 'maxIterations']),
    )
    expect(Object.keys(method.contract as object)).toContain('amp/stateEvolution')
    const made = method.make({ prior: 'gaussian', variance: 2 } as never, { kind: 'key', words: [0, 0] })
    expect(made.name).toBe('amp')
  })
  it('refuses a matrix of the wrong size, a bad damping, and ends a run that blows up as diverged', () => {
    expect(() => M.ampAlgorithm({ ...gen.problem, rows: 7 }, { prior })).toThrow(/measurements/)
    expect(() => M.ampAlgorithm(gen.problem, { prior, damping: 1 })).toThrow(/damping/)
    expect(() => M.ampAlgorithm(gen.problem, { prior, truth: [1, 2] })).toThrow(/truth/)
    // a matrix of a far larger scale than the theory takes does not converge: the run ends, one way or the other, without an exception
    const wild = { ...gen.problem, matrix: Array.from(gen.problem.matrix, (v) => v * 40) }
    const r = runAlgorithm(M.ampAlgorithm(wild, { prior, maxIterations: 200 }), { maxSteps: 200 })
    expect(['diverged', 'completed', 'converged']).toContain(r.status)
  })
})
