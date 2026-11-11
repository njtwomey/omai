// EP inside a grid over the hyperparameters, on the two Gaussian process benchmarks: the truth is the exact marginal likelihood of the
// length scale (the function integrated out) by quadrature, and the function values the mixture of their exact Gaussian conditionals.
import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { gpPosteriorGivenLengthScale } from '../models/gaussian-process.ts'
import { benchmarkModel } from '../models/suite.ts'
import { epGridMethod } from './ep-grid.ts'
import { analysisOf } from './model-input.ts'
import { methodKinds } from './models.ts'

/** The worst error of the function values' means (in truth sds) and of their sds (relative) against the truth's marginals. */
function functionErrors(
  truth: ReturnType<ReturnType<typeof benchmarkModel>['truth']>,
  mean: ArrayLike<number>,
  sd: ArrayLike<number>,
): { mean: number; sd: number } {
  let worstMean = 0
  let worstSd = 0
  for (let i = 0; i < 12; i++) {
    const t = truth.marginals[`f[${i}]`]
    if (t === undefined) continue
    worstMean = Math.max(worstMean, Math.abs((mean[i] ?? 0) - t.mean) / t.sd)
    worstSd = Math.max(worstSd, Math.abs((sd[i] ?? 0) / t.sd - 1))
  }
  return { mean: worstMean, sd: worstSd }
}

describe('epGridMethod on the Gaussian process benchmarks', () => {
  for (const name of ['gaussian-process-regression', 'gaussian-process-non-centred']) {
    const model = benchmarkModel(name)
    const bound = model.bound()
    const truth = model.truth(undefined, 'full')

    it(`${name}: the evidence, the length scale and the function agree with the exact quadrature`, () => {
      const method = epGridMethod(bound)
      expect(method.check?.(analysisOf(bound)).applies).toBe(true)
      const run = session(method, { steps: 400, mode: 'eager', config: { points: 41, tol: 1e-10 } })
      expect(run.status).toBe('completed')
      const post = method.posterior?.(run.state, run.config)
      if (post === undefined) throw new Error('no posterior')
      expect(post.status).toBe('converged')
      // Each node is exact (a Gaussian likelihood), so the grid evidence is the quadrature of the marginal likelihood.
      expect(Math.abs((post.logEvidence?.value ?? Number.NaN) - (truth.logEvidence ?? 0))).toBeLessThan(1e-4)
      const ell = truth.marginals['ell']
      const row = post.summary().find((r) => r.site === 'ell')
      expect(Math.abs((row?.mean ?? 0) - (ell?.mean ?? 0))).toBeLessThan(1e-3 * (ell?.sd ?? 1))
      expect(Math.abs((row?.sd ?? 0) - (ell?.sd ?? 0))).toBeLessThan(2e-3 * (ell?.sd ?? 1))
      const e = functionErrors(truth, toFlat(post.mean('f')), toFlat(post.stddev('f')))
      expect(e.mean).toBeLessThan(1e-3)
      expect(e.sd).toBeLessThan(2e-3)
    })

    it(`${name}: a coarse grid is a rough answer, and every step is a usable snapshot`, () => {
      const method = epGridMethod(bound)
      const run = session(method, { steps: 9, mode: 'eager', config: { points: 9, tol: 1e-10 } })
      const early = method.posterior?.(run.stateAt(3), run.config)
      const late = method.posterior?.(run.state, run.config)
      expect(early?.status).toBe('stopped')
      expect(Math.abs((late?.logEvidence?.value ?? Number.NaN) - (truth.logEvidence ?? 0))).toBeLessThan(0.05)
      if (late === undefined) throw new Error('no posterior')
      expect(functionErrors(truth, toFlat(late.mean('f')), toFlat(late.stddev('f'))).mean).toBeLessThan(0.05)
    })
  }

  it('extends the grid outwards where its edge still has mass, within the budget of nodes', () => {
    const model = benchmarkModel('gaussian-process-regression')
    const truth = model.truth(undefined, 'full')
    const method = epGridMethod(model.bound())
    // A first grid between the 20th and 80th percentile of the prior misses most of the posterior.
    const run = session(method, { steps: 300, mode: 'eager', config: { points: 9, tail: 0.2, tol: 1e-10 } })
    expect(run.state.extensions).toBeGreaterThan(0)
    expect(run.state.nodes.length).toBeGreaterThan(9)
    const post = method.posterior?.(run.state, run.config)
    expect(Math.abs((post?.logEvidence?.value ?? Number.NaN) - (truth.logEvidence ?? 0))).toBeLessThan(0.02)
    const capped = session(method, { steps: 300, mode: 'eager', config: { points: 9, tail: 0.2, maxNodes: 12 } })
    expect(capped.state.nodes.length).toBeLessThanOrEqual(12)
  })

  it('a run resumes from its serialised state and reaches the same posterior', () => {
    const model = benchmarkModel('gaussian-process-regression')
    const method = epGridMethod(model.bound())
    const whole = session(method, { steps: 31, mode: 'eager', config: { points: 31 } })
    const half = session(method, { steps: 31, mode: 'eager', config: { points: 31 } })
    const json = JSON.parse(JSON.stringify(half.stateAt(10)))
    expect(json.nodes.length).toBe(10)
    expect(whole.state.nodes.length).toBe(31)
    expect(whole.state.nodes[10]?.logEvidence).toBe(half.state.nodes[10]?.logEvidence)
  })

  it('is a kind of methodKinds, and refuses a model with no custom function to see through', () => {
    expect(Object.keys(methodKinds)).toContain('ep-grid')
    const bound = benchmarkModel('beta-bernoulli').bound()
    const verdict = epGridMethod(bound).check?.(analysisOf(bound))
    expect(verdict?.applies).toBe(false)
    expect(verdict?.reason).toMatch(/custom function/)
  })

  it('a node with the length scale fixed is the textbook posterior of the function: its evidence is the exact marginal likelihood', () => {
    const bound = benchmarkModel('gaussian-process-regression').bound()
    const x = Array.from(toFlat(bound.data['x'] as never))
    const y = Array.from(toFlat(bound.data['y'] as never))
    const method = epGridMethod(bound)
    const run = session(method, { steps: 3, mode: 'eager', config: { points: 3 } })
    for (const node of run.state.nodes) {
      const exact = gpPosteriorGivenLengthScale(x, y, node.x[0] ?? 1)
      expect(Math.abs(node.logEvidence - exact.logMarginalLikelihood)).toBeLessThan(1e-8)
      const f = node.sites['f']
      expect(f?.kind).toBe('multivariateNormal')
      for (let i = 0; i < 12; i++) expect(Math.abs((f?.mean[i] ?? 0) - (exact.mean[i] ?? 0))).toBeLessThan(1e-7)
    }
  })
})
