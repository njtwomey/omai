// Fractional EP on the softmax classifier, whose three class slopes share every observation: one Gaussian each leaves their spreads
// too narrow (about half the exact posterior's with 24 points) and the run does not settle; a power of about 5 / rows (the
// `power` knob of `epSpecMethod`) widens them to the exact spread, the run settles, and the mean moves by a fraction of a sd.
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
function against(power: number): { sdRatio: number; meanError: number; converged: boolean } {
  const method = epSpecMethod(atomBound(problem))
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
  it('plain EP is too narrow, a power of 0.2 gives the exact spread, settles, and moves the mean by less than half a sd', () => {
    const plain = against(1)
    const fractional = against(0.2)
    expect(plain.sdRatio).toBeLessThan(0.7)
    expect(fractional.sdRatio).toBeGreaterThan(0.8)
    expect(fractional.sdRatio).toBeLessThan(1.25)
    expect(fractional.converged).toBe(true)
    expect(fractional.meanError).toBeLessThan(0.5)
    expect(plain.meanError).toBeLessThan(0.15)
  }, 60_000)

  it('a power below one is a knob of EP only: the variational and belief propagation methods do not have it', () => {
    const method = epSpecMethod(atomBound(problem))
    const knobs = (m: { config: unknown }): string[] =>
      Object.keys((m.config as { spaces: Record<string, unknown> }).spaces)
    expect(knobs(method)).toContain('power')
    expect(knobs(vmpSpecMethod(atomBound(problem)))).not.toContain('power')
  })

  it('the multi-start method takes the same power: a power below one widens the slopes there too', () => {
    const sdOf = (power: number): number => {
      const method = epMultiStartMethod(atomBound(problem))
      const run = session(method, {
        seed: 1,
        steps: 1,
        mode: 'eager',
        config: { starts: 1, tol: 1e-6, damping: 0.5, maxSweeps: 400, power },
      })
      const post = method.posterior?.(run.state, run.config)
      return post?.summary().find((r) => r.site === problem.quantities[0]?.name.replace(/\[.*$/, ''))?.sd ?? Number.NaN
    }
    expect(sdOf(0.2)).toBeGreaterThan(sdOf(1))
  }, 60_000)
})
