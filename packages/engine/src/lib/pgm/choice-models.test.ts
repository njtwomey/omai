import { describe, expect, it } from 'vitest'
import { gumbel, key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as P from './index.ts'

const n = 6
const theta = [1.2, 0.6, 0.1, -0.3, -0.7, -0.9]

/** Rankings drawn from the Plackett-Luce model by the Gumbel trick, each over a random subset of items. */
function sample(count: number, size: number): number[][] {
  return Array.from({ length: count }, (_, g) => {
    const items = Array.from({ length: n }, (_, i) => i)
      .sort((a, b) => ((a * 7 + g * 3) % 11) - ((b * 7 + g * 3) % 11))
      .slice(0, size)
    const gum = toFlat(gumbel(key(500 + g), [size]))
    return items
      .map((i, j) => ({ i, s: (theta[i] ?? 0) + (gum[j] ?? 0) }))
      .sort((a, b) => b.s - a.s)
      .map((x) => x.i)
  })
}

describe('Bradley-Terry', () => {
  it('has strengths in the ratio of wins for two items', () => {
    const fit = P.fitBradleyTerry(2, [
      [0, 1],
      [0, 1],
      [0, 1],
      [1, 0],
    ])
    expect((fit.strength[0] ?? 0) / (fit.strength[1] ?? 1)).toBeCloseTo(3, 6)
    expect(P.winProbability(fit, 0, 1)).toBeCloseTo(0.75, 8)
    expect(fit.theta[0]).toBeCloseTo(-(fit.theta[1] ?? 0), 12)
  })
  it('rejects an item that never wins or never loses', () => {
    expect(() =>
      P.fitBradleyTerry(3, [
        [0, 1],
        [0, 2],
        [1, 0],
      ]),
    ).toThrow(/no win or no loss/)
  })
})

describe('Plackett-Luce', () => {
  const rankings = sample(600, 4)
  const problem: P.ChoiceProblem = { nItems: n, rankings }
  it('recovers the planted log strengths from sampled rankings', () => {
    const fit = P.fitPlackettLuce(problem)
    expect(fit.converged).toBe(true)
    const centred = theta.map((t) => t - theta.reduce((a, b) => a + b, 0) / n)
    fit.theta.forEach((t, i) => expect(Math.abs(t - (centred[i] ?? 0))).toBeLessThan(0.25))
  })
  it('increases the likelihood at every minorise-maximise step', () => {
    const algorithm = P.choiceAlgorithm(problem)
    let state = algorithm.init(key(0))
    let last = state.objective
    for (let i = 0; i < 25; i++) {
      state = algorithm.step(state, undefined as never)
      expect(state.objective).toBeLessThanOrEqual(last + 1e-12)
      last = state.objective
    }
  })
  it('reaches the same strengths from the generic solvers of the slot', () => {
    const base = P.fitPlackettLuce(problem, { tol: 1e-10 })
    for (const solver of [
      { kind: 'newton', gradTol: 1e-10 },
      { kind: 'bfgs', gradTol: 1e-10 },
      { kind: 'lbfgs', gradTol: 1e-10 },
    ]) {
      const fit = P.fitPlackettLuce(problem, { solver, tol: 1e-9 })
      expect(fit.solver).toBe(solver.kind)
      expect(fit.logLikelihood).toBeCloseTo(base.logLikelihood, 6)
      fit.theta.forEach((t, i) => expect(t).toBeCloseTo(base.theta[i] ?? 0, 4))
    }
  })
  it('serialises the traced fit', () => {
    const algorithm = P.choiceAlgorithm(problem, { solver: { kind: 'bfgs', gradTol: 1e-9 } })
    const run = runAlgorithm(algorithm, { maxSteps: 3 })
    expect(algorithm.deserialise?.(algorithm.serialise?.(run.state) as never)?.u).toEqual(run.state.u)
  })
})
