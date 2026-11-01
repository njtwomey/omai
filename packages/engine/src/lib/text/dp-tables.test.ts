import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { sellers } from './approximate-search.ts'
import { dpKinds } from './dp-options.ts'
import { dpAlgorithm, dpContract, dpTable } from './dp-tables.ts'
import { editCounts, lcsTable, levenshtein, longestCommonSubsequence } from './similarity.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

describe('dpAlgorithm', () => {
  it('fills one row per step and traces back in one more', () => {
    const run = runAlgorithm(dpAlgorithm('kitten', 'sitting'), { maxSteps: 20 })
    expect([run.status, run.steps]).toEqual(['converged', 7])
    expect(run.state.table[6 * 8 + 7]).toBe(levenshtein('kitten', 'sitting'))
    const weighted = dpTable('kitten', 'sitting', { weights: { substitution: 2 } })
    expect(weighted.table[6 * 8 + 7]).toBe(levenshtein('kitten', 'sitting', { substitution: 2 }))
  })

  it('agrees with levenshtein, the LCS table, edit counts and sellers on random strings', () => {
    fc.assert(
      fc.property(
        fc.string({ unit: fc.constantFrom('a', 'b', 'c'), maxLength: 12 }),
        fc.string({ unit: fc.constantFrom('a', 'b', 'c'), maxLength: 12 }),
        (a, b) => {
          const t = dpTable(a, b)
          expect(t.table[(t.rows + 1) * (t.columns + 1) - 1]).toBe(levenshtein(a, b))
          const lcs = dpTable(a, b, { kind: 'lcs' })
          const lt = lcsTable(a, b)
          expect(lcs.table[(lcs.rows + 1) * (lcs.columns + 1) - 1]).toBe(lt[a.length]?.[b.length])
          expect(longestCommonSubsequence(a, b)).toHaveLength(lt[a.length]?.[b.length] ?? -1)
          const c = editCounts(a, b)
          expect(c.hits + c.substitutions + c.deletions).toBe(a.length)
          expect(c.hits + c.substitutions + c.insertions).toBe(b.length)
          expect(c.substitutions + c.deletions + c.insertions).toBe(levenshtein(a, b))
          for (const o of sellers(a.slice(0, 5), b, 2)) {
            expect(o.distance).toBeLessThanOrEqual(2)
            expect(levenshtein(a.slice(0, 5), b.slice(o.start, o.end))).toBeLessThanOrEqual(o.distance)
          }
        },
      ),
      { numRuns: 60 },
    )
  })

  it('keeps the documented results of the wrappers', () => {
    expect(longestCommonSubsequence('ABCBDAB', 'BDCABA').join('')).toBe('BCBA')
    expect(editCounts(['the', 'cat', 'sat'], ['the', 'cat', 'sit', 'down'])).toEqual({
      hits: 2,
      substitutions: 1,
      deletions: 0,
      insertions: 1,
    })
    expect(sellers('survey', 'surgery', 2).map((o) => [o.start, o.end, o.distance])).toEqual([
      [0, 5, 2],
      [0, 6, 2],
      [0, 7, 2],
    ])
    expect(() => sellers('a', 'b', -1)).toThrow(/non-negative/)
  })

  it('serialises the table, resumes mid-fill and logs the rows', () => {
    const alg = dpAlgorithm('intention', 'execution', { kind: 'edit-counts' })
    const whole = runAlgorithm(alg, { maxSteps: 30 })
    const part = runAlgorithm(alg, { maxSteps: 4 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 30,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 4, serialised: true },
    })
    expect(rest.state.counts).toEqual(whole.state.counts)
    expect(Array.from(rest.state.table)).toEqual(Array.from(whole.state.table))
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 30, tracer: rec, policy: { ...policies.minimal, metrics: true, detail: 'full' } })
    const cells = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'dp/cell' ? [e.value] : []))
    expect(cells).toHaveLength(whole.steps)
    expect(Object.keys(dpContract)).toHaveLength(3)
  })
})

describe('dpKinds', () => {
  it('names the tables and builds options', () => {
    expect(dpKinds.names).toEqual(['levenshtein', 'lcs', 'edit-counts', 'sellers'])
    expect(dpKinds.make({ kind: 'sellers', k: 2 })).toEqual({ kind: 'sellers', k: 2 })
    expect(dpKinds.properties({ kind: 'sellers' }).freeStart).toBe(true)
    const t = dpTable('abc', 'abd', dpKinds.make({ kind: 'levenshtein', substitution: 3 }))
    expect(t.table[t.table.length - 1]).toBe(2)
  })
})
