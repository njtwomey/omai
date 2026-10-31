import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../trace/index.ts'
import {
  canonicalPatterns,
  containsPattern,
  exhaustivePatterns,
  gsp,
  gspAlgorithm,
  patternText,
  prefixSpan,
  prefixSpanAlgorithm,
  sequenceSupport,
} from './index.ts'

// Pei et al. (2001), table 1, with a b c d e f g as 1 to 7.
const a = 1
const b = 2
const c = 3
const d = 4
const e = 5
const f = 6
const g = 7
const PEI = [
  [[a], [a, b, c], [a, c], [d], [c, f]],
  [[a, d], [c], [b, c], [a, e]],
  [[e, f], [a, b], [d, f], [c], [b]],
  [[e], [g], [a, f], [c], [b], [c]],
]
const counts = (ps: readonly { pattern: readonly (readonly number[])[]; count: number }[]): Record<string, number> =>
  Object.fromEntries(ps.map((p) => [patternText(p.pattern), p.count]))

describe('sequential patterns on the PrefixSpan paper example', () => {
  const found = counts(prefixSpan(PEI, { minSupport: 0.5 }))

  it("has the paper's length-1 patterns with their supports", () => {
    for (const [name, n] of [
      ['(1)', 4],
      ['(2)', 4],
      ['(3)', 4],
      ['(4)', 3],
      ['(5)', 3],
      ['(6)', 3],
    ] as const) {
      expect(found[name]).toBe(n)
    }
    expect(found['(7)']).toBeUndefined()
  })

  it("has the paper's longer patterns, among them <(ab)>, <a(bc)a>-style extensions and <abc>", () => {
    expect(found['(1 2)']).toBe(2)
    expect(found['(1)(2)(3)']).toBe(2)
    expect(found['(1)(3)(2)']).toBe(3)
    expect(found['(1)(2 3)']).toBe(2)
  })

  it('PrefixSpan, GSP and exhaustive enumeration agree', () => {
    const truth = exhaustivePatterns(PEI, { minSupport: 0.5 })
    expect(prefixSpan(PEI, { minSupport: 0.5 })).toEqual(truth)
    expect(gsp(PEI, { minSupport: 0.5 })).toEqual(truth)
  })

  it('every reported count is a true count', () => {
    for (const p of prefixSpan(PEI, { minSupport: 0.5 })) expect(sequenceSupport(PEI, p.pattern)).toBe(p.count)
  })
})

describe('gap constraints', () => {
  const data = [
    [[1], [9], [2]],
    [[1], [2]],
    [[1], [8], [8], [2]],
  ]
  it('containment respects the minimum and maximum gap and the span', () => {
    expect(containsPattern(data[0] ?? [], [[1], [2]])).toBe(true)
    expect(containsPattern(data[0] ?? [], [[1], [2]], { maxGap: 1 })).toBe(false)
    expect(containsPattern(data[1] ?? [], [[1], [2]], { minGap: 2 })).toBe(false)
    expect(containsPattern(data[2] ?? [], [[1], [2]], { maxSpan: 2 })).toBe(false)
    expect(containsPattern(data[2] ?? [], [[1], [2]], { maxSpan: 3 })).toBe(true)
  })

  it('GSP with a maximum gap counts only adjacent occurrences and matches the exhaustive count', () => {
    const constraints = { maxGap: 1 }
    const truth = exhaustivePatterns(data, { minSupport: 0.3 }, constraints)
    expect(gsp(data, { minSupport: 0.3 }, constraints)).toEqual(truth)
    expect(counts(truth)['(1)(2)']).toBe(1)
  })
})

describe('the algorithms', () => {
  it('PrefixSpan extends one prefix per step', () => {
    const { state, steps } = runAlgorithm(prefixSpanAlgorithm(PEI, { minSupport: 0.5 }), { maxSteps: 10000, seed: 0 })
    expect(state.stack.length).toBe(0)
    // one step to start, one per pattern expanded (the empty prefix and every pattern found)
    expect(steps).toBe(state.found.length + 2)
    expect(canonicalPatterns(state.found).length).toBe(state.found.length)
  })

  it('GSP takes one step per level', () => {
    const { state } = runAlgorithm(gspAlgorithm(PEI, { minSupport: 0.5 }), { maxSteps: 100, seed: 0 })
    expect(state.done).toBe(true)
    expect(state.k).toBe(Math.max(...state.found.map((p) => p.pattern.flat().length)) + 1)
  })
})
