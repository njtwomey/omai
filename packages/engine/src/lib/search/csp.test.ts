import { describe, expect, it } from 'vitest'
import { queensCsp, sendMoreMoney, sudokuCsp } from '../../testing/search.ts'
import { messages, policies, recorder, runAlgorithm, validateAgainstSchema } from '../trace/index.ts'
import {
  ac3,
  ac3Algorithm,
  allDifferent,
  backtrackingAlgorithm,
  backtrackingSearch,
  binaryConstraint,
  csp,
  isSolution,
  minConflicts,
  minConflictsAlgorithm,
  violations,
} from './index.ts'

describe('CSP construction', () => {
  it('rejects malformed problems', () => {
    expect(() => csp(['a', 'a'], { a: [1] }, [])).toThrow()
    expect(() => csp(['a'], {}, [])).toThrow()
    expect(() => csp(['a'], { a: [1] }, [binaryConstraint('a', 'b', () => true)])).toThrow()
  })

  it('reports violations and solutions', () => {
    const p = csp(['a', 'b', 'c'], { a: [1, 2], b: [1, 2], c: [1, 2, 3] }, allDifferent(['a', 'b', 'c']))
    expect(violations(p, { a: 1, b: 1, c: 1 }).length).toBe(3)
    expect(isSolution(p, { a: 1, b: 2, c: 3 })).toBe(true)
    expect(isSolution(p, { a: 1, b: 2, c: 4 })).toBe(false)
  })

  it('handles empty problems and empty domains', () => {
    expect(backtrackingSearch(csp([], {}, [])).assignment).toEqual({})
    expect(backtrackingSearch(csp(['a'], { a: [] }, [])).status).toBe('failure')
    expect(ac3(csp(['a'], { a: [] }, [])).consistent).toBe(false)
    expect(() => minConflicts(csp(['a'], { a: [] }, []))).toThrow()
  })
})

describe('solvers', () => {
  it('solve n-queens with every configuration and min-conflicts', () => {
    const q = queensCsp(10)
    for (const inference of ['none', 'forward-checking', 'mac'] as const) {
      const r = backtrackingSearch(q, { inference, variableOrder: 'mrv', valueOrder: 'lcv' })
      expect(isSolution(q, r.assignment ?? {})).toBe(true)
    }
    const m = minConflicts(queensCsp(20), { seed: 4 })
    expect(m.status).toBe('solved')
    expect(isSolution(queensCsp(20), m.assignment ?? {})).toBe(true)
  })

  it('prunes with inference and counts the effort', () => {
    const q = queensCsp(8)
    const plain = backtrackingSearch(q)
    const fc = backtrackingSearch(q, { inference: 'forward-checking' })
    const mac = backtrackingSearch(q, { inference: 'mac' })
    expect(fc.assignments).toBeLessThan(plain.assignments)
    expect(mac.assignments).toBeLessThanOrEqual(fc.assignments)
    expect(fc.pruned).toBeGreaterThan(0)
    expect(plain.backtracks).toBeGreaterThan(0)
    expect(plain.checks).toBeGreaterThan(0)
  })

  it('solves n-ary problems (SEND + MORE = MONEY) with generalised arc consistency', () => {
    const r = backtrackingSearch(sendMoreMoney(), { inference: 'mac', variableOrder: 'mrv' })
    const a = r.assignment ?? {}
    const word = (w: string): number => [...w].reduce((n, ch) => 10 * n + (a[ch] ?? 0), 0)
    expect(word('SEND') + word('MORE')).toBe(word('MONEY'))
  })

  it('stops on budgets', () => {
    const hard = sudokuCsp('.'.repeat(81))
    expect(backtrackingSearch(hard, { all: true, maxAttempts: 200 }).status).toBe('budget')
    const unsat = queensCsp(3)
    expect(minConflicts(unsat, { maxSteps: 50, seed: 1 }).status).toBe('budget')
    expect(backtrackingSearch(queensCsp(8), { all: true, maxSolutions: 5 }).solutions.length).toBe(5)
  })

  it('traces assignments, backtracks and solutions as declared messages', () => {
    const q = queensCsp(5)
    const algorithm = backtrackingAlgorithm(q, { inference: 'forward-checking' })
    const rec = recorder()
    runAlgorithm(algorithm, { maxSteps: 1000, tracer: rec, policy: policies.teaching })
    expect(validateAgainstSchema(rec.events, algorithm.schema ?? (undefined as never))).toEqual([])
    expect(messages(rec.run(), 'assign').length).toBeGreaterThan(0)
    expect(messages(rec.run(), 'solution').length).toBe(1)
    const arc = ac3Algorithm(queensCsp(5, 0))
    const rec3 = recorder()
    const r = ac3(queensCsp(5, 0), { tracer: rec3, policy: policies.teaching })
    expect(validateAgainstSchema(rec3.events, arc.schema ?? (undefined as never))).toEqual([])
    const removed = messages(rec3.run(), 'revise').flatMap((m) => (m.payload as { removed: number[] }).removed)
    expect(removed.length).toBe(5 * 5 - Object.values(r.domains).reduce((s, d) => s + d.length, 0) - 4)
    const mc = minConflictsAlgorithm(queensCsp(6))
    const rec2 = recorder()
    runAlgorithm(mc, { maxSteps: 500, tracer: rec2, policy: policies.teaching, seed: 2 })
    expect(validateAgainstSchema(rec2.events, mc.schema ?? (undefined as never))).toEqual([])
  })

  it('uses a starting assignment for min-conflicts', () => {
    const q = queensCsp(4)
    const r = minConflicts(q, { initial: { '0': 1, '1': 3, '2': 0, '3': 2 } })
    expect(r.assignment).toEqual({ '0': 1, '1': 3, '2': 0, '3': 2 })
    expect(r.assignments).toBe(4)
  })
})
