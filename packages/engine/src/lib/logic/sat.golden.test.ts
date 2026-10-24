import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import {
  allModels,
  type Cnf,
  cdcl,
  countModels,
  dpll,
  maxsat,
  parseDimacs,
  type SoftClause,
  satisfies,
  unsatCore,
  walksat,
} from './index.ts'

describe('SAT against python-sat', () => {
  for (const c of loadFixture('logic/sat').cases) {
    it(c.name, () => {
      const p = c.params
      if (c.fn === 'dimacs') {
        const f = parseDimacs(p['text'] as string)
        expect(f.numVars).toBe(p['numVars'])
        expect(f.clauses).toEqual(p['clauses'])
        return
      }
      const f: Cnf = { numVars: p['numVars'] as number, clauses: p['clauses'] as number[][] }
      const status = p['sat'] ? 'sat' : 'unsat'
      for (const r of [cdcl(f), cdcl(f, { restarts: 'none' }), dpll(f), dpll(f, { branching: 'jeroslow-wang' })]) {
        expect(r.status).toBe(status)
        if (r.model !== null) expect(satisfies(f, r.model)).toBe(true)
      }
      if (p['sat'] === true && f.numVars <= 20) {
        const w = walksat(f, { seed: 3, maxFlips: 200_000 })
        expect(w.status).toBe('sat')
      }
      if (p['count'] !== undefined) {
        expect(countModels(f)).toBe(p['count'])
        if ((p['count'] as number) <= 200) {
          const ms = allModels(f)
          expect(ms.length).toBe(p['count'])
          expect(new Set(ms.map((m) => m.join(','))).size).toBe(ms.length)
          for (const m of ms) expect(satisfies(f, m)).toBe(true)
        }
      }
      if (c.fn === 'core') {
        const core = unsatCore(f)
        if (p['sat'] === true) expect(core).toBeNull()
        else {
          const muses = (p['muses'] as number[][]).map((m) => m.join(','))
          expect(muses).toContain((core ?? []).join(','))
          expect(muses).toContain((p['musx'] as number[]).join(','))
        }
      }
    })
  }
})

describe('MaxSAT against RC2', () => {
  for (const c of loadFixture('logic/maxsat').cases) {
    it(c.name, () => {
      const p = c.params
      const problem = {
        numVars: p['numVars'] as number,
        hard: p['hard'] as number[][],
        soft: p['soft'] as SoftClause[],
      }
      const r = maxsat(problem)
      if (p['cost'] === null) {
        expect(r.status).toBe('unsat')
        return
      }
      expect(r.status).toBe('optimal')
      expect(r.cost).toBe(p['cost'])
      const model = r.model ?? []
      expect(satisfies({ numVars: problem.numVars, clauses: problem.hard }, model)).toBe(true)
      const falsified = problem.soft
        .filter((s) => !satisfies({ numVars: problem.numVars, clauses: [s.clause] }, model))
        .reduce((acc, s) => acc + s.weight, 0)
      expect(falsified).toBe(p['cost'])
    })
  }
})
