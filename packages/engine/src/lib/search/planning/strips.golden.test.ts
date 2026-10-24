import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { forwardPlan, graphplan, pddl, regressionPlan, validatePlan } from '../index.ts'

describe('STRIPS planning against pyperplan', () => {
  for (const c of loadFixture('search/planning/strips').cases) {
    it(c.name, () => {
      const p = c.params
      const problem = pddl(p['domain'] as string, p['problem'] as string)
      expect(problem.initial).toEqual(p['initial'])
      expect(problem.goal).toEqual(p['goal'])
      const length = p['length'] as number
      // Optimal planners reach pyperplan's optimal length.
      for (const r of [
        forwardPlan(problem, { search: 'breadth-first' }),
        forwardPlan(problem, { heuristic: 'h-max' }),
        regressionPlan(problem, { heuristic: 'h-max' }),
      ]) {
        expect(r.status).toBe('solved')
        expect(r.plan.length).toBe(length)
        expect(validatePlan(problem, r.plan).valid).toBe(true)
      }
      // Satisficing planners find valid plans no shorter.
      for (const r of [
        forwardPlan(problem, { heuristic: 'h-add', search: 'greedy' }),
        forwardPlan(problem, { heuristic: 'h-ff', search: 'greedy' }),
        forwardPlan(problem, { heuristic: 'h-add', search: 'weighted-a-star', weight: 3 }),
      ]) {
        expect(validatePlan(problem, r.plan).valid).toBe(true)
        expect(r.plan.length).toBeGreaterThanOrEqual(length)
      }
      // GraphPlan's parallel plan serialises into a valid plan; its steps never exceed the optimal length.
      const g = graphplan(problem)
      expect(g.status).toBe('solved')
      expect(validatePlan(problem, g.plan).valid).toBe(true)
      expect(g.steps.length).toBeLessThanOrEqual(length)
      expect(g.plan.length).toBeGreaterThanOrEqual(length)
      // pyperplan's own plan is valid in our grounding.
      expect(validatePlan(problem, p['plan'] as string[]).valid).toBe(true)
    })
  }
})
