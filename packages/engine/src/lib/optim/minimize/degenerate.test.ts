import { describe, expect, it } from 'vitest'
import { OmaiError } from '../../base/index.ts'
import { isTensor, mul, square, sum, toFlat, type Value } from '../../tensor/index.ts'
import * as O from '../index.ts'

const bowl = (x: unknown): Value => sum(square(x as Value))
const tiny = (x: unknown): Value => mul(sum(square(x as Value)), 1e-7)

describe('the minimise family on degenerate input', () => {
  // The tolerance is absolute: gradients below it everywhere stop a run at its start (OL-6). A relative tolerance, opt
  // in, makes the stop `min(gradTol, gradTolRel * |g0|)`, so the same objective is minimised by every generic solver.
  it('stops at the start under the absolute gradient tolerance, and minimises with gradTolRel', () => {
    const solvers: [string, (o: O.VectorMethodOptions) => O.MinimizeResult][] = [
      ['gradientDescent', (o) => O.gradientDescent(tiny, [1, 1], o)],
      ['conjugateGradient', (o) => O.conjugateGradient(tiny, [1, 1], o)],
      ['newton', (o) => O.newton(tiny, [1, 1], o)],
      ['bfgs', (o) => O.bfgs(tiny, [1, 1], o)],
      ['lbfgs', (o) => O.lbfgs(tiny, [1, 1], o)],
      ['coordinateDescent', (o) => O.coordinateDescent(tiny, [1, 1], o)],
    ]
    for (const [name, run] of solvers) {
      const absolute = run({})
      expect(absolute.steps, name).toBe(0)
      expect(absolute.converged, name).toBe(true)
      const relative = run({ gradTolRel: 1e-3 })
      expect(relative.converged, name).toBe(true)
      expect(Math.max(...toFlat(relative.x as Value)), name).toBeLessThan(1e-2)
    }
    expect(() => O.bfgs(tiny, [1, 1], { gradTolRel: 0 })).toThrow(OmaiError)
  })

  // An empty parameter vector has nothing to minimise: `converged` after no steps, for every method (OL-26).
  it('returns converged after no steps for an empty parameter vector, whichever the method', () => {
    const none = { bounds: [] as [number, number][] }
    const runs: [string, () => O.MinimizeResult][] = [
      ['gradientDescent', () => O.gradientDescent(bowl, [])],
      ['conjugateGradient', () => O.conjugateGradient(bowl, [])],
      ['coordinateDescent', () => O.coordinateDescent(bowl, [])],
      ['newton', () => O.newton(bowl, [])],
      ['bfgs', () => O.bfgs(bowl, [])],
      ['lbfgs', () => O.lbfgs(bowl, [])],
      ['lbfgsb', () => O.lbfgsb(bowl, [])],
      ['nelderMead', () => O.nelderMead(bowl, [])],
      ['powell', () => O.powell(bowl, [])],
      ['cmaes', () => O.cmaes(bowl, [])],
      ['hookeJeeves', () => O.hookeJeeves(bowl, [])],
      ['compassSearch', () => O.compassSearch(bowl, [])],
      ['cobyla', () => O.cobyla(bowl, [])],
      ['simulatedAnnealing', () => O.simulatedAnnealing(bowl, [])],
      ['basinHopping', () => O.basinHopping(bowl, [])],
      ['dualAnnealing', () => O.dualAnnealing(bowl, [], none)],
      ['differentialEvolution', () => O.differentialEvolution(bowl, [], none)],
      ['particleSwarm', () => O.particleSwarm(bowl, [], none)],
    ]
    for (const [name, run] of runs) {
      const r = run()
      expect(r.status, name).toBe('converged')
      expect(r.converged, name).toBe(true)
      expect(r.steps, name).toBe(0)
      expect(r.fun, name).toBe(0)
    }
  })

  // A start that is not finite is never reported as a solution (OL-26).
  it('never reports a non-finite start as converged', () => {
    const runs: [string, () => O.MinimizeResult][] = [
      ['gradientDescent', () => O.gradientDescent(bowl, [Number.NaN, 1])],
      ['conjugateGradient', () => O.conjugateGradient(bowl, [Number.NaN, 1])],
      ['coordinateDescent', () => O.coordinateDescent(bowl, [Number.NaN, 1])],
      ['newton', () => O.newton(bowl, [Number.NaN, 1])],
      ['bfgs', () => O.bfgs(bowl, [Number.NaN, 1])],
      ['lbfgs', () => O.lbfgs(bowl, [Number.NaN, 1])],
      ['nelderMead', () => O.nelderMead(bowl, [Number.NaN, 1])],
      ['powell', () => O.powell(bowl, [Number.NaN, 1])],
    ]
    for (const [name, run] of runs) {
      let verdict: string
      try {
        verdict = run().status
      } catch (error) {
        expect(error, name).toBeInstanceOf(OmaiError)
        verdict = `threw ${(error as OmaiError).code}`
      }
      expect(['diverged', 'failed', 'threw value'], name).toContain(verdict)
    }
  })

  // Exact coordinate descent searches a bracket along each coordinate; an error of the user's objective is not a missing
  // bracket and must reach the caller (OL-30).
  it('lets an error thrown by the objective propagate out of exact coordinate descent', () => {
    const f = (x: unknown): Value => {
      // Called on traced values for the gradient and on plain tensors for the line searches along a coordinate.
      if (isTensor(x)) throw new Error('boom')
      return sum(square(x as Value))
    }
    expect(() => O.coordinateDescent(f, [1, 2], { update: 'exact' })).toThrow('boom')
  })
})
