// Fractional EP on the softmax classifier, whose three class slopes share every observation. With one message per class score
// (`packScalars: false`) one Gaussian each leaves their spreads too narrow (about half the exact posterior's with 24 points); the
// `power` knob of `epSpecMethod` (fractional EP, the site divided by the power) changes the spread by a few percent only, so it is no
// substitute for the joint site (the scalar site once sent the undivided message, whose fixed point mixed the cavity and the
// projection and did widen the slopes: that was not fractional EP and is gone). Packed (the default for EP) the slopes are one vector and each observation one joint site
// (`multiclass-row`), which carries the correlations: plain EP has the exact spread and means, and the power is not needed.
import { describe, expect, it } from 'vitest'
import { session } from '../../trace/index.ts'
import { epMultiStartMethod, epSpecMethod, vmpSpecMethod } from '../inference/index.ts'
import { epAtom } from './ep-atom-list.ts'
import { atomBound } from './ep-atoms.ts'
import { atomBeliefs } from './run.ts'

const atom = epAtom('softmax-regression-learned')
const problem = atom.problem({ ...atom.defaults, seed: 3, n: 24 })
const exact = problem.exact ?? {}

/** The mean ratio of EP's sd to the exact sd over the three slopes, the largest mean error in exact sds, and whether the run settled. */
function against(power: number, packScalars = false): { sdRatio: number; meanError: number; converged: boolean } {
  const method = epSpecMethod(atomBound(problem), { packScalars })
  const run = session(method, {
    seed: 1,
    steps: 402,
    mode: 'eager',
    config: { tol: 1e-6, damping: 0.5, maxSweeps: 400, power },
  })
  const post = method.posterior?.(run.state, run.config)
  if (post === undefined) throw new Error('no posterior')
  const beliefs = atomBeliefs(problem, post as never)
  const names = problem.quantities.map((q) => q.name)
  const ratio = names.map((n) => (beliefs[n]?.sd ?? Number.NaN) / (exact[n]?.sd ?? 1))
  const error = names.map(
    (n) => Math.abs((beliefs[n]?.mean ?? Number.NaN) - (exact[n]?.mean ?? 0)) / (exact[n]?.sd ?? 1),
  )
  return {
    sdRatio: ratio.reduce((a, b) => a + b, 0) / ratio.length,
    meanError: Math.max(...error),
    converged: post.status === 'converged',
  }
}

describe('fractional EP (the power knob) on the softmax classifier', () => {
  it('packed, plain EP has the exact spreads and means (one joint site per observation): the power is not needed', () => {
    const plain = against(1, true)
    expect(plain.sdRatio).toBeGreaterThan(0.97)
    expect(plain.sdRatio).toBeLessThan(1.03)
    expect(plain.meanError).toBeLessThan(0.05)
    expect(plain.converged).toBe(true)
  }, 60_000)

  it('with a message per score, plain EP is too narrow, and fractional EP (the site divided by the power) stays about as narrow: the correlations are what is missing', () => {
    // The scalar site's message is divided by the power as every power rule's is (cavity + power * message is the projection), so the
    // fixed point is fractional EP's and the power does not stand in for the joint site: the spread moves by a few percent.
    const plain = against(1)
    const fractional = against(0.5)
    expect(plain.sdRatio).toBeLessThan(0.7)
    expect(Math.abs(fractional.sdRatio - plain.sdRatio)).toBeLessThan(0.05)
    expect(fractional.converged).toBe(true)
    expect(fractional.meanError).toBeLessThan(0.15)
    expect(plain.meanError).toBeLessThan(0.15)
  }, 60_000)

  it('a power below one is a knob of EP only: the variational and belief propagation methods do not have it', () => {
    const method = epSpecMethod(atomBound(problem))
    const knobs = (m: { config: unknown }): string[] =>
      Object.keys((m.config as { spaces: Record<string, unknown> }).spaces)
    expect(knobs(method)).toContain('power')
    expect(knobs(vmpSpecMethod(atomBound(problem)))).not.toContain('power')
  })

  it('the multi-start method takes the same power (one message per score): the spread of the first slope changes by a few percent', () => {
    const sdOf = (power: number): number => {
      const method = epMultiStartMethod(atomBound(problem), { packScalars: false })
      const run = session(method, {
        seed: 1,
        steps: 1,
        mode: 'eager',
        config: { starts: 1, tol: 1e-6, damping: 0.5, maxSweeps: 400, power },
      })
      const post = method.posterior?.(run.state, run.config)
      return post?.summary().find((r) => r.site === problem.quantities[0]?.name.replace(/\[.*$/, ''))?.sd ?? Number.NaN
    }
    expect(Math.abs(sdOf(0.5) / sdOf(1) - 1)).toBeLessThan(0.1)
    expect(sdOf(0.5)).not.toBe(sdOf(1))
  }, 60_000)
})
