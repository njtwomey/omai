import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as P from '../index.ts'

type Beliefs = Readonly<Record<string, readonly number[]>>
const beliefsOf = (alg: ReturnType<typeof P.beliefPropagationAlgorithm>, state: P.BeliefPropagationState): Beliefs =>
  ((alg.observe?.(state, 0)?.frame as { beliefs: Beliefs } | undefined)?.beliefs ?? {}) as Beliefs

describe('belief propagation at finer granularity', () => {
  const grid = P.isingModel({ rows: 2, cols: 3, coupling: 0.3, field: { s_0_0: 0.4, s_1_2: -0.3 } })
  const chain = P.isingModel({ rows: 1, cols: 4, coupling: 0.7, field: { s_0_0: 0.5, s_0_3: -0.4 } })
  const run = (model: P.DiscreteModel, options: P.BeliefPropagationOptions, maxSteps = 100_000) => {
    const alg = P.beliefPropagationAlgorithm(model, { tol: 1e-13, ...options })
    return { alg, ...runAlgorithm(alg, { maxSteps }) }
  }

  it('ends in the same beliefs as whole sweeps, for every schedule and size of step', () => {
    const cases: [P.DiscreteModel, P.BeliefPropagationOptions][] = [
      [grid, { schedule: 'flooding' }],
      [grid, { schedule: 'sequential' }],
      [grid, { schedule: 'residual' }],
      [chain, { schedule: 'tree' }],
    ]
    for (const [model, options] of cases) {
      const sweep = run(model, options)
      expect(sweep.status).toBe('converged')
      const expected = beliefsOf(sweep.alg, sweep.state)
      for (const granularity of ['factor', 'message'] as const) {
        const fine = run(model, { ...options, granularity })
        expect(fine.status).toBe('converged')
        expect(fine.steps).toBeGreaterThan(sweep.steps)
        const got = beliefsOf(fine.alg, fine.state)
        for (const v of Object.keys(expected))
          for (const [i, p] of (expected[v] ?? []).entries()) expect(got[v]?.[i]).toBeCloseTo(p, 8)
      }
    }
  })

  it('takes one message per step, and an iteration is the plan of its steps', () => {
    const alg = P.beliefPropagationAlgorithm(chain, { granularity: 'message' })
    let state = alg.init(key(0))
    const cursors: number[] = []
    for (let i = 0; i < 40; i++) {
      state = alg.step(state, {} as never)
      cursors.push(state.cursor)
    }
    // Back at 0 only between iterations, so the steps of an iteration are all the updates (a variable and a factor message per edge).
    const period = cursors.indexOf(0) + 1
    expect(period).toBeGreaterThan(4)
    expect(cursors.filter((c) => c === 0).length).toBe(Math.floor(40 / period))
    expect(state.updates).toBe(40)
  })

  it("reports every variable's belief in the frame, in probabilities that sum to one", () => {
    const r = run(grid, {})
    const b = beliefsOf(r.alg, r.state)
    const exact = P.beliefPropagation(grid, { tol: 1e-13 })
    for (const [v, p] of Object.entries(b)) {
      expect(p.reduce((a, x) => a + x, 0)).toBeCloseTo(1, 10)
      expect(p[1]).toBeCloseTo(P.factorValues(exact.marginals[v] as P.Factor)[1] ?? 0, 8)
    }
  })

  it('says which message each step sent, by name, in the frame', () => {
    const alg = P.beliefPropagationAlgorithm(chain, { granularity: 'message', schedule: 'sequential' })
    let state = alg.init(key(0))
    expect((alg.observe?.(state, 0)?.frame as { sent: unknown[] }).sent).toEqual([])
    const kinds: string[] = []
    for (let i = 0; i < 12; i++) {
      state = alg.step(state, {} as never)
      const sent = (alg.observe?.(state, i)?.frame as { sent: { kind: string; from: string; to: string }[] }).sent
      expect(sent).toHaveLength(1)
      const [m] = sent
      // A variable message goes from a variable to a factor, a factor message the other way.
      expect(
        m?.kind === 'variableToFactor' ? m.to.startsWith('f') && !m.from.startsWith('f') : m?.from.startsWith('f'),
      ).toBe(true)
      kinds.push(m?.kind ?? '')
    }
    expect(new Set(kinds)).toEqual(new Set(['variableToFactor', 'factorToVariable']))
    // A whole sweep sends everything, so it lists nothing.
    const sweep = P.beliefPropagationAlgorithm(chain)
    const after = sweep.step(sweep.init(key(0)), {} as never)
    expect((sweep.observe?.(after, 1)?.frame as { sent: unknown[] }).sent).toEqual([])
  })

  it('keeps whole sweeps as the default', () => {
    const a = run(grid, {})
    const b = run(grid, { granularity: 'sweep' })
    expect(b.steps).toBe(a.steps)
    expect(a.state.cursor).toBe(0)
  })
})
