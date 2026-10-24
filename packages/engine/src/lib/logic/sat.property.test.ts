import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { bruteForceCount, bruteForceSat, cnf as cnfArb } from '../../testing/logic.ts'
import { allModels, cdcl, countModels, dpll, maxsat, satisfies, unsatCore, walksat } from './index.ts'

describe('SAT properties', () => {
  it('models satisfy every clause and UNSAT agrees with brute force', () => {
    fc.assert(
      fc.property(cnfArb(7, 24), (f) => {
        const truth = bruteForceSat(f.numVars, f.clauses)
        for (const r of [
          cdcl(f),
          cdcl(f, { restarts: 'luby', restartBase: 1 }),
          dpll(f),
          dpll(f, { branching: 'dlis', pureLiterals: false }),
        ]) {
          expect(r.status).toBe(truth ? 'sat' : 'unsat')
          if (r.model !== null) expect(satisfies(f, r.model)).toBe(true)
        }
        const w = walksat(f, { seed: 1, maxFlips: 2000 })
        if (w.status === 'sat') expect(satisfies(f, w.model ?? [])).toBe(true)
        else expect(w.status).toBe('unknown')
      }),
      { numRuns: 200 },
    )
  })

  it('counts and enumerates models exactly', () => {
    fc.assert(
      fc.property(cnfArb(6, 12), (f) => {
        const n = bruteForceCount(f.numVars, f.clauses)
        expect(countModels(f)).toBe(n)
        expect(allModels(f).length).toBe(n)
      }),
      { numRuns: 150 },
    )
  })

  it('assumptions: failed assumptions are an unsatisfiable subset', () => {
    fc.assert(
      fc.property(cnfArb(6, 14), fc.array(fc.integer({ min: 1, max: 6 }), { maxLength: 4 }), (f, picks) => {
        const assumptions = [...new Set(picks.filter((v) => v <= f.numVars))].map((v, i) => (i % 2 ? -v : v))
        const units = assumptions.map((a) => [a])
        const truth = bruteForceSat(f.numVars, [...f.clauses, ...units])
        const r = cdcl(f, { assumptions })
        expect(r.status).toBe(truth ? 'sat' : 'unsat')
        if (r.status === 'unsat' && bruteForceSat(f.numVars, f.clauses)) {
          const failed = r.failedAssumptions ?? []
          expect(failed.every((a) => assumptions.includes(a))).toBe(true)
          expect(bruteForceSat(f.numVars, [...f.clauses, ...failed.map((a) => [a])])).toBe(false)
        }
      }),
      { numRuns: 150 },
    )
  })

  it('cores are minimal unsatisfiable subsets', () => {
    fc.assert(
      fc.property(cnfArb(5, 14), (f) => {
        const core = unsatCore(f)
        if (bruteForceSat(f.numVars, f.clauses)) {
          expect(core).toBeNull()
          return
        }
        const clauses = (core ?? []).map((i) => f.clauses[i] ?? [])
        expect(bruteForceSat(f.numVars, clauses)).toBe(false)
        for (let k = 0; k < clauses.length; k++) {
          expect(
            bruteForceSat(
              f.numVars,
              clauses.filter((_, j) => j !== k),
            ),
          ).toBe(true)
        }
      }),
      { numRuns: 150 },
    )
  })

  it('MaxSAT optimum equals the brute-force minimum', () => {
    fc.assert(
      fc.property(
        cnfArb(5, 4),
        cnfArb(5, 8),
        fc.array(fc.integer({ min: 1, max: 4 }), { minLength: 8, maxLength: 8 }),
        (h, s, w) => {
          const n = Math.max(h.numVars, s.numVars)
          const soft = s.clauses.map((clause, i) => ({ clause, weight: w[i] ?? 1 }))
          let best = Number.POSITIVE_INFINITY
          for (let m = 0; m < 1 << n; m++) {
            const model = Array.from({ length: n }, (_, i) => ((m >> i) & 1 ? i + 1 : -(i + 1)))
            if (!satisfies({ numVars: n, clauses: h.clauses }, model)) continue
            const cost = soft
              .filter((c) => !satisfies({ numVars: n, clauses: [c.clause] }, model))
              .reduce((a, c) => a + c.weight, 0)
            best = Math.min(best, cost)
          }
          const r = maxsat({ numVars: n, hard: h.clauses, soft })
          expect(r.cost).toBe(best)
          expect(r.status).toBe(Number.isFinite(best) ? 'optimal' : 'unsat')
        },
      ),
      { numRuns: 150 },
    )
  })
})
