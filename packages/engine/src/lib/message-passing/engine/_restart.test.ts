import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../../trace/index.ts'
import { ImproperMessageError } from '../core/errors.ts'
import type { FactorOperator } from '../core/types.ts'
import * as M from '../index.ts'
import { largestOf, restartDiffers } from './_restart.ts'

/** An operator that never computes its message (an improper cavity, always): the run stalls, and restarts while it may. */
const reluctant: FactorOperator = {
  name: 'reluctant',
  description: 'never computes its message',
  ports: [{ name: 'x', accepts: ['gaussian'] }],
  support: { ep: 'exact' },
  ep: () => {
    throw new ImproperMessageError('the cavity has no density')
  },
}

const stalledGraph = (): M.CompiledGraph =>
  M.compileGraph(
    {
      variables: [{ name: 'x', family: { kind: 'gaussian' } }],
      factors: [M.gaussianPrior('x', 0, 1), { operator: 'reluctant', args: ['x'] }],
    },
    { ...M.defaultOperators(), reluctant },
  )

describe('the restarts of a stalled run', () => {
  it('a restart is always damped more than the attempt it replaces: three for a run under a half, two from a half, none from 0.9', () => {
    const count = (damping: number): number => [1, 2, 3, 4, 5].filter((k) => restartDiffers(damping, k)).length
    expect(count(0)).toBe(3)
    expect(count(0.3)).toBe(3)
    expect(count(0.5)).toBe(2)
    expect(count(0.7)).toBe(2)
    expect(count(0.8)).toBe(1)
    expect(count(0.9)).toBe(0)
    expect(count(0.95)).toBe(0)
  })

  it('a run damped by a half and asked for three restarts goes 0.8, 0.9 and stops (a third would repeat the second)', () => {
    const g = stalledGraph()
    const r = runAlgorithm(M.epAlgorithm(g, { damping: 0.5, restarts: 3, tol: 1e-10 }), { maxSteps: 400 })
    expect(r.state.restarts).toBe(2)
    expect(r.state.damping).toBe(0.9)
  })

  it('an undamped run goes 0.5, 0.8, 0.9 as before', () => {
    const g = stalledGraph()
    const r = runAlgorithm(M.epAlgorithm(g, { damping: 0, restarts: 3, tol: 1e-10 }), { maxSteps: 400 })
    expect(r.state.restarts).toBe(3)
    expect(r.state.damping).toBe(0.9)
    const one = runAlgorithm(M.epAlgorithm(g, { damping: 0, restarts: 1, tol: 1e-10 }), { maxSteps: 400 })
    expect(one.state.damping).toBe(0.5)
  })

  it('the largest residual is a loop: a million of them do not overflow the call stack, and a NaN is a NaN', () => {
    const many = new Float64Array(1_000_000)
    many[777_777] = 3
    expect(largestOf(many, 0)).toBe(3)
    expect(largestOf([], 0)).toBe(0)
    expect(largestOf([-2, -1])).toBe(-1)
    expect(largestOf([1, Number.NaN, 2], 0)).toBeNaN()
  })
})
