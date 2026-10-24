import { describe, expect, it } from 'vitest'
import { messages, policies, recorder, runAlgorithm, validateAgainstSchema } from '../trace/index.ts'
import {
  cdcl,
  cdclAlgorithm,
  cnf,
  dpll,
  dpllAlgorithm,
  maxsat,
  maxsatAlgorithm,
  parseDimacs,
  toDimacs,
  walksat,
} from './index.ts'

describe('SAT solvers', () => {
  it('reads and writes DIMACS', () => {
    const f = parseDimacs('c a comment\np cnf 4 3\n1 -2 0\n2 3\n -4 0\n4 0\n')
    expect(f.clauses).toEqual([[1, -2], [2, 3, -4], [4]])
    expect(parseDimacs(toDimacs(f, ['round trip']))).toEqual(f)
    expect(() => parseDimacs('1 2 0')).toThrow(/header/)
    expect(() => parseDimacs('p cnf 2 1\n1 2')).toThrow(/terminated/)
    expect(() => cnf([[0]])).toThrow(/literal/)
  })

  it('traces CDCL decisions, conflicts and learned clauses', () => {
    // Pigeonhole 4 into 3: unsatisfiable, and needs learning.
    const x = (p: number, h: number): number => 3 * p + h + 1
    const clauses: number[][] = []
    for (let p = 0; p < 4; p++) clauses.push([x(p, 0), x(p, 1), x(p, 2)])
    for (let h = 0; h < 3; h++)
      for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) clauses.push([-x(p, h), -x(q, h)])
    const f = cnf(clauses)
    const rec = recorder()
    const r = cdcl(f, { tracer: rec, policy: policies.teaching })
    expect(r.status).toBe('unsat')
    expect(r.stats.conflicts).toBeGreaterThan(0)
    const run = rec.run()
    expect(messages(run, 'learn').length).toBe(r.stats.learned)
    expect(messages(run, 'decide').length).toBe(r.stats.decisions)
    const schema = cdclAlgorithm(f).schema
    if (schema !== undefined) expect(validateAgainstSchema(run.events, schema)).toEqual([])
    const d = dpll(f)
    expect([d.status, d.stats.backtracks > 0]).toEqual(['unsat', true])
  })

  it('handles empty formulas, empty clauses and budgets', () => {
    expect(cdcl(cnf([], 3)).model).toEqual([-1, -2, -3])
    expect(cdcl(cnf([[]], 1)).status).toBe('unsat')
    expect(dpll(cnf([[]], 1)).status).toBe('unsat')
    expect(dpll(cnf([])).status).toBe('sat')
    expect(walksat(cnf([[1], [-1]]), { maxFlips: 20 }).status).toBe('unknown')
    expect(() => cdcl(cnf([[1]]), { assumptions: [5] })).toThrow(/assumption/)
    const hard = cnf(
      Array.from({ length: 30 }, (_, i) => [((i * 7) % 12) + 1, -(((i * 5) % 12) + 1), ((i * 3) % 12) + 1]),
    )
    expect(['sat', 'unknown']).toContain(cdcl(hard, { maxConflicts: 1 }).status)
    const steps = runAlgorithm(
      dpllAlgorithm(
        cnf([
          [1, 2],
          [-1, 2],
        ]),
      ),
      { maxSteps: 100 },
    )
    expect(steps.state.status).toBe('sat')
  })

  it('solves MaxSAT and reports unsatisfiable hard parts', () => {
    expect(maxsat({ numVars: 2, hard: [[1], [-1]], soft: [] }).status).toBe('unsat')
    expect(
      maxsat({
        numVars: 2,
        hard: [],
        soft: [
          { clause: [1], weight: 2 },
          { clause: [-1], weight: 1 },
        ],
      }).cost,
    ).toBe(1)
    expect(() => maxsat({ numVars: 1, hard: [[2]], soft: [] })).toThrow(/literal/)
    expect(() => maxsatAlgorithm({ numVars: 1, hard: [], soft: [{ clause: [1], weight: 0 }] })).toThrow(/weight/)
  })
})
