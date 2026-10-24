import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  forwardPlan,
  graphplan,
  hAdd,
  hFF,
  hMax,
  partialOrderPlan,
  regressionPlan,
  type StripsAction,
  strips,
  validatePlan,
} from '../index.ts'

/** A random STRIPS problem over up to 6 facts and 7 actions. */
const randomProblem = fc
  .record({
    facts: fc.integer({ min: 2, max: 6 }),
    actions: fc.array(
      fc.record({
        pre: fc.array(fc.nat(5), { maxLength: 2 }),
        add: fc.array(fc.nat(5), { minLength: 1, maxLength: 2 }),
        del: fc.array(fc.nat(5), { maxLength: 2 }),
      }),
      { minLength: 1, maxLength: 7 },
    ),
    init: fc.array(fc.nat(5), { minLength: 1, maxLength: 3 }),
    goal: fc.array(fc.nat(5), { minLength: 1, maxLength: 3 }),
  })
  .map(({ facts, actions, init, goal }) => {
    const f = (i: number): string => `p${i % facts}`
    const acts: StripsAction[] = actions.map((a, k) => {
      const add = [...new Set(a.add.map(f))]
      return { name: `a${k}`, pre: a.pre.map(f), add, del: a.del.map(f).filter((x) => !add.includes(x)) }
    })
    return strips(init.map(f), goal.map(f), acts)
  })

describe('planning properties', () => {
  it('orders h-max <= h-FF <= h-add, with h-max admissible', () => {
    fc.assert(
      fc.property(randomProblem, (p) => {
        const optimal = forwardPlan(p, { search: 'breadth-first' })
        const max = hMax(p)(p.initial)
        const add = hAdd(p)(p.initial)
        const ff = hFF(p)(p.initial)
        expect(max).toBeLessThanOrEqual(ff)
        expect(ff).toBeLessThanOrEqual(add)
        if (optimal.status === 'solved') expect(max).toBeLessThanOrEqual(optimal.plan.length)
        else expect(optimal.status).toBe('failure')
      }),
      { numRuns: 200 },
    )
  })

  it('optimal planners agree, and every planner returns valid plans or none', () => {
    fc.assert(
      fc.property(randomProblem, (p) => {
        const bfs = forwardPlan(p, { search: 'breadth-first' })
        const astar = forwardPlan(p, { heuristic: 'h-max' })
        const back = regressionPlan(p, { heuristic: 'h-max' })
        const blind = regressionPlan(p, { heuristic: 'blind', search: 'breadth-first' })
        expect(astar.status).toBe(bfs.status)
        expect(back.status).toBe(bfs.status)
        expect(blind.status).toBe(bfs.status)
        if (bfs.status !== 'solved') return
        const n = bfs.plan.length
        for (const r of [astar, back, blind]) {
          expect(r.plan.length).toBe(n)
          expect(validatePlan(p, r.plan).valid).toBe(true)
        }
        for (const heuristic of ['h-add', 'h-ff', 'goal-count'] as const) {
          const r = forwardPlan(p, { heuristic, search: 'greedy' })
          expect(validatePlan(p, r.plan).valid).toBe(true)
        }
        const g = graphplan(p)
        expect(g.status).toBe('solved')
        expect(validatePlan(p, g.plan).valid).toBe(true)
        expect(g.steps.length).toBeLessThanOrEqual(n)
        const pop = partialOrderPlan(p, { maxExpansions: 2000 })
        if (pop.status === 'solved') expect(validatePlan(p, pop.linearization).valid).toBe(true)
      }),
      { numRuns: 200 },
    )
  })

  it('GraphPlan proves unsolvable problems unsolvable', () => {
    fc.assert(
      fc.property(randomProblem, (p) => {
        if (forwardPlan(p, { search: 'breadth-first' }).status === 'failure')
          expect(graphplan(p).status).toBe('failure')
      }),
      { numRuns: 200 },
    )
  })
})
