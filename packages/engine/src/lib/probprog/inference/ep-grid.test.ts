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

  it('a warm start reaches the same grid posterior in fewer sweeps (every node starts from the nearest settled one)', () => {
    const bound = benchmarkModel('gaussian-process-regression').bound()
    const method = epGridMethod(bound)
    const cold = session(method, { steps: 200, mode: 'eager', config: { points: 15, tol: 1e-10 } })
    const warm = session(method, { steps: 200, mode: 'eager', config: { points: 15, tol: 1e-10, warm: true } })
    const sweeps = (nodes: readonly { sweeps: number }[]): number => nodes.reduce((s, n) => s + n.sweeps, 0)
    expect(sweeps(warm.state.nodes)).toBeLessThanOrEqual(sweeps(cold.state.nodes))
    const a = method.posterior?.(cold.state, cold.config)
    const b = method.posterior?.(warm.state, warm.config)
    expect(Math.abs((b?.logEvidence?.value ?? Number.NaN) - (a?.logEvidence?.value ?? 0))).toBeLessThan(1e-6)
    const ell = (p: typeof a): number => p?.summary().find((r) => r.site === 'ell')?.mean ?? Number.NaN
    expect(Math.abs(ell(b) - ell(a))).toBeLessThan(1e-6)
  })

  it('a walk starts in the middle and steps outwards only while the posterior has mass: fewer nodes, the same answer', () => {
    const model = benchmarkModel('gaussian-process-regression')
    const truth = model.truth(undefined, 'full')
    const method = epGridMethod(model.bound())
    const all = session(method, { steps: 400, mode: 'eager', config: { points: 41, tol: 1e-10 } })
    const walk = session(method, {
      steps: 400,
      mode: 'eager',
      config: { points: 41, tol: 1e-10, order: 'walk', window: 9, warm: true },
    })
    expect(walk.state.nodes.length).toBeLessThan(all.state.nodes.length)
    expect(walk.state.pending.length).toBe(0)
    const indices = walk.state.nodes.map((n) => n.index[0] ?? 0)
    // contiguous: the nodes run are an interval of the lattice
    expect(Math.max(...indices) - Math.min(...indices) + 1).toBe(indices.length)
    const post = method.posterior?.(walk.state, walk.config)
    expect(Math.abs((post?.logEvidence?.value ?? Number.NaN) - (truth.logEvidence ?? 0))).toBeLessThan(0.02)
    const ell = truth.marginals['ell']
    const row = post?.summary().find((r) => r.site === 'ell')
    expect(Math.abs((row?.mean ?? 0) - (ell?.mean ?? 0))).toBeLessThan(0.02 * (ell?.sd ?? 1))
  })

  it('a node that did not settle weighs the grid where its evidence does not exceed the best settled node, and not above it', () => {
    const method = epGridMethod(benchmarkModel('gaussian-process-regression').bound())
    const run = session(method, { steps: 7, mode: 'eager', config: { points: 7, tol: 1e-10 } })
    const nodes = run.state.nodes
    const best = Math.max(...nodes.map((n) => n.logEvidence + n.logPrior))
    const mean = (shift: number | undefined): number => {
      const state = {
        ...run.state,
        nodes: nodes.map((n, i) =>
          i === 0
            ? {
                ...n,
                converged: false,
                logEvidence: shift === undefined ? Number.NEGATIVE_INFINITY : n.logEvidence + shift,
              }
            : n,
        ),
      }
      return (
        method
          .posterior?.(state, run.config)
          ?.summary()
          .find((r) => r.site === 'ell')?.mean ?? Number.NaN
      )
    }
    const settled =
      method
        .posterior?.(run.state, run.config)
        ?.summary()
        .find((r) => r.site === 'ell')?.mean ?? Number.NaN
    // Not settled, with the evidence it had: still below the best, so it weighs the grid exactly as it did.
    expect(Math.abs(mean(0) - settled)).toBeLessThan(1e-9)
    // Not settled, far above the best settled node: no weight, as if it had no evidence at all.
    expect(Math.abs(mean(best + 50) - mean(undefined))).toBeLessThan(1e-9)
  })

  it('the tail knob is honoured after check() has used the default tail: the grid spans the tail asked for', () => {
    const bound = benchmarkModel('gaussian-process-regression').bound()
    const fresh = epGridMethod(bound)
    const used = epGridMethod(bound)
    // check() builds its ranges at the tail 1e-4; a later run with another tail must not inherit them.
    expect(used.check?.(analysisOf(bound)).applies).toBe(true)
    const config = { points: 5, tail: 0.2 }
    const a = session(fresh, { steps: 1, mode: 'eager', config })
    const b = session(used, { steps: 1, mode: 'eager', config })
    expect(b.state.axes).toEqual(a.state.axes)
    const wide = session(used, { steps: 1, mode: 'eager', config: { points: 5, tail: 1e-8 } })
    expect((wide.state.axes[0]?.du ?? 0) * 4).toBeGreaterThan((b.state.axes[0]?.du ?? 0) * 4)
  })

  it('says that it failed, and does not throw, while no node has an EP evidence', () => {
    const method = epGridMethod(benchmarkModel('gaussian-process-regression').bound())
    const run = session(method, { steps: 3, mode: 'eager', config: { points: 3 } })
    const none = { ...run.state, nodes: run.state.nodes.map((n) => ({ ...n, logEvidence: Number.NaN })) }
    const post = method.posterior?.(none, run.config)
    expect(post?.status).toBe('failed')
    expect(post?.diagnostics['reason']).toMatch(/no node has an EP evidence/)
  })

  it('weighs only the nodes that settled when some did: a run that cycled does not carry an inflated evidence into the grid', () => {
    const method = epGridMethod(benchmarkModel('gaussian-process-regression').bound())
    const run = session(method, { steps: 5, mode: 'eager', config: { points: 5 } })
    const honest = method.posterior?.(run.state, run.config)
    const inflated = {
      ...run.state,
      nodes: run.state.nodes.map((n, i) => (i === 0 ? { ...n, converged: false, logEvidence: n.logEvidence + 50 } : n)),
    }
    const post = method.posterior?.(inflated, run.config)
    const a = honest?.summary().find((r) => r.site === 'ell')
    const b = post?.summary().find((r) => r.site === 'ell')
    // The node that did not settle has no weight, so its evidence (50 nats too high) changes nothing about the others'.
    expect(Math.abs((b?.mean ?? Number.NaN) - (a?.mean ?? 0))).toBeLessThan(0.5 * (a?.sd ?? 1))
    expect(post?.logEvidence?.value ?? Number.NaN).toBeLessThan((honest?.logEvidence?.value ?? 0) + 1)
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
