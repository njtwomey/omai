import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  type Assignment,
  ac3,
  backtrackingSearch,
  type Csp,
  type CspConstraint,
  constraint,
  csp,
  cspProblem,
  depthFirstSearch,
  isSolution,
  minConflicts,
} from './index.ts'

/** A random CSP: up to 6 variables with small integer domains, random binary relations and an optional ternary one. */
const randomCsp: fc.Arbitrary<Csp<number>> = fc
  .record({
    n: fc.integer({ min: 1, max: 6 }),
    d: fc.integer({ min: 1, max: 4 }),
    seed: fc.array(fc.boolean(), { minLength: 400, maxLength: 400 }),
    ternary: fc.boolean(),
  })
  .map(({ n, d, seed, ternary }) => {
    const vars = Array.from({ length: n }, (_, i) => `x${i}`)
    const domains = Object.fromEntries(vars.map((v) => [v, Array.from({ length: d }, (_, k) => k)]))
    let bit = 0
    const next = (): boolean => seed[bit++ % seed.length] ?? false
    const constraints: CspConstraint<number>[] = []
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!next()) continue
        const allowed = new Set<string>()
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) if (next() || next()) allowed.add(`${a},${b}`)
        constraints.push(constraint([vars[i] as string, vars[j] as string], (v) => allowed.has(`${v[0]},${v[1]}`)))
      }
    }
    if (ternary && n >= 3)
      constraints.push(constraint(['x0', 'x1', 'x2'], (v) => ((v[0] ?? 0) + (v[1] ?? 0) + (v[2] ?? 0)) % 2 === 0))
    return csp(vars, domains, constraints)
  })

/** Every solution by brute force. */
function bruteForce(problem: Csp<number>): string[] {
  const out: string[] = []
  const vars = problem.variables
  const walk = (i: number, a: Record<string, number>): void => {
    if (i === vars.length) {
      if (isSolution(problem, a)) out.push(JSON.stringify(vars.map((v) => a[v])))
      return
    }
    for (const x of problem.domains[vars[i] as string] ?? []) walk(i + 1, { ...a, [vars[i] as string]: x })
  }
  walk(0, {})
  return out.sort()
}

const key = (problem: Csp<number>, s: Assignment<number>): string => JSON.stringify(problem.variables.map((v) => s[v]))

describe('CSP properties', () => {
  it('every configuration of backtracking finds exactly the solutions, each satisfying every constraint', () => {
    fc.assert(
      fc.property(randomCsp, (problem) => {
        const truth = bruteForce(problem)
        for (const inference of ['none', 'forward-checking', 'mac'] as const) {
          for (const variableOrder of ['static', 'mrv', 'degree', 'mrv-degree'] as const) {
            for (const valueOrder of ['static', 'lcv'] as const) {
              const r = backtrackingSearch(problem, { inference, variableOrder, valueOrder, all: true })
              expect(r.solutions.map((s) => key(problem, s)).sort()).toEqual(truth)
              for (const s of r.solutions) expect(isSolution(problem, s)).toBe(true)
              expect(r.status).toBe(truth.length > 0 ? 'solved' : 'failure')
            }
          }
        }
        // The incremental formulation searched depth-first finds a solution exactly when one exists.
        const dfs = depthFirstSearch(cspProblem(problem))
        expect(dfs.status).toBe(truth.length > 0 ? 'solved' : 'failure')
      }),
      { numRuns: 150 },
    )
  })

  it('forward checking and MAC never assign more values than plain backtracking in static order', () => {
    fc.assert(
      fc.property(randomCsp, (problem) => {
        const plain = backtrackingSearch(problem, { all: true }).assignments
        const fc_ = backtrackingSearch(problem, { all: true, inference: 'forward-checking' }).assignments
        const mac = backtrackingSearch(problem, { all: true, inference: 'mac' }).assignments
        expect(fc_).toBeLessThanOrEqual(plain)
        expect(mac).toBeLessThanOrEqual(fc_)
      }),
      { numRuns: 150 },
    )
  })

  it('AC-3 is sound (keeps every value of every solution) and its domains are arc consistent', () => {
    fc.assert(
      fc.property(randomCsp, (problem) => {
        const truth = bruteForce(problem).map((s) => JSON.parse(s) as number[])
        const r = ac3(problem)
        if (!r.consistent) {
          expect(truth).toEqual([])
          return
        }
        for (const s of truth) {
          problem.variables.forEach((v, i) => {
            expect(r.domains[v]).toContain(s[i])
          })
        }
        // A second pass from the reduced domains removes nothing more (a fixed point).
        expect(ac3(problem, { domains: r.domains }).domains).toEqual(r.domains)
      }),
      { numRuns: 200 },
    )
  })

  it('min-conflicts returns only solutions', () => {
    fc.assert(
      fc.property(randomCsp, fc.nat(1000), (problem, seed) => {
        const r = minConflicts(problem, { seed, maxSteps: 300 })
        if (r.status === 'solved') expect(isSolution(problem, r.assignment ?? {})).toBe(true)
        else expect(r.assignment).toBeNull()
      }),
      { numRuns: 100 },
    )
  })
})
