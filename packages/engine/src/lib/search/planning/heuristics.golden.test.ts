import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { hAdd, hFF, hMax, pddl, relaxedPlan } from '../index.ts'

describe('relaxation heuristics against pyperplan', () => {
  for (const c of loadFixture('search/planning/heuristics').cases) {
    it(c.name, () => {
      const p = c.params
      const problem = pddl(p['domain'] as string, p['problem'] as string)
      const states = p['states'] as string[][]
      const add = hAdd(problem)
      const max = hMax(problem)
      const ff = hFF(problem)
      expect(states.map(add)).toEqual(p['hAdd'])
      expect(states.map(max)).toEqual(p['hMax'])
      for (const s of states) {
        // h-max <= h-FF <= h-add, and the relaxed plan reaches the goal ignoring deletes.
        expect(ff(s)).toBeGreaterThanOrEqual(max(s))
        expect(ff(s)).toBeLessThanOrEqual(add(s))
        const plan = relaxedPlan(problem, s) ?? []
        const reached = new Set(s)
        const byName = new Map(problem.actions.map((a) => [a.name, a]))
        for (let pass = 0; pass < plan.length; pass++) {
          for (const name of plan) {
            const a = byName.get(name)
            if (a?.pre.every((f) => reached.has(f))) for (const f of a.add) reached.add(f)
          }
        }
        expect(problem.goal.every((g) => reached.has(g))).toBe(true)
        expect(plan.length).toBe(ff(s))
      }
    })
  }
})
