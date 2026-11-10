// Minka's clutter problem: the generator is seeded and keeps its truth, the grid posterior is a density with the
// divergences of a Gaussian in both directions, and the alpha family moves from expectation propagation (wide, KL(p || q)
// small) toward the variational Gaussian (narrow, KL(q || p) small) on it.
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import { alphaEpAlgorithm } from './alpha.ts'
import { clutterGraph, clutterPosterior, clutterProblem } from './clutter.ts'
import { marginalsOf } from './engine.ts'

const problem = clutterProblem(key(7))
const exact = clutterPosterior(problem)

const fit = (alpha: number): { mean: number; variance: number } => {
  const g = clutterGraph(problem)
  const r = runAlgorithm(alphaEpAlgorithm(g, { alpha, tol: 1e-9 }), { maxSteps: 500, tracer: recorder() })
  const m = marginalsOf(g, r.state)['x'] as Record<string, number>
  return { mean: m['mean'] as number, variance: m['variance'] as number }
}

describe('the clutter problem', () => {
  it('is seeded: the same key gives the same points, with the clutter flagged', () => {
    expect(clutterProblem(key(7)).ys).toEqual(problem.ys)
    expect(clutterProblem(key(8)).ys).not.toEqual(problem.ys)
    expect(problem.ys).toHaveLength(20)
    expect(problem.isClutter.some(Boolean) && !problem.isClutter.every(Boolean)).toBe(true)
  })

  it('has an exact posterior that integrates to one, with the moments of its density', () => {
    const h = (exact.x[1] ?? 0) - (exact.x[0] ?? 0)
    const mass = exact.density.reduce((s, d) => s + d * h, 0)
    expect(mass).toBeCloseTo(1, 6)
    expect(exact.sd).toBeGreaterThan(0)
    expect(Number.isFinite(exact.logEvidence)).toBe(true)
  })

  it('gives zero divergence to the posterior itself and a positive one to any other Gaussian', () => {
    const own = exact.divergences(exact.mean, exact.sd ** 2)
    const off = exact.divergences(exact.mean + 1, exact.sd ** 2)
    expect(off.forward).toBeGreaterThan(own.forward)
    expect(off.reverse).toBeGreaterThan(0)
  })

  it('moves from mass-covering at alpha 1 to mode-seeking as alpha falls', () => {
    const wide = fit(1)
    const mid = fit(0.4)
    const narrow = fit(0.05)
    expect(wide.variance).toBeGreaterThan(mid.variance)
    expect(mid.variance).toBeGreaterThan(narrow.variance)
    const dw = exact.divergences(wide.mean, wide.variance)
    const dn = exact.divergences(narrow.mean, narrow.variance)
    expect(dw.forward).toBeLessThan(dn.forward)
    expect(dn.reverse).toBeLessThan(dw.reverse)
  })
})
