import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../trace/index.ts'
import {
  aprioriSd,
  aprioriSdAlgorithm,
  cn2Sd,
  columnSelectors,
  discoverSubgroups,
  subgroupCounts,
  subgroupQuality,
} from './index.ts'

// A deterministic table: the target is 1 exactly when a=x and b=p, plus two flipped rows as noise.
const rows = (): { columns: Record<string, string[]>; y: number[] } => {
  const a: string[] = []
  const b: string[] = []
  const c: string[] = []
  const y: number[] = []
  for (let i = 0; i < 120; i++) {
    a.push(['x', 'y', 'z'][i % 3] as string)
    b.push(['p', 'q'][Math.floor(i / 3) % 2] as string)
    c.push(['u', 'v', 'w', 't'][Math.floor(i / 5) % 4] as string)
    y.push(a[i] === 'x' && b[i] === 'p' ? 1 : 0)
  }
  y[0] = 0 // a flipped positive
  y[1] = 1 // a flipped negative
  return { columns: { a, b, c }, y }
}

describe('quality functions', () => {
  it('weighted relative accuracy is p/N - nP/N^2', () => {
    const k = { size: 20, positives: 15, total: 100, totalPositives: 30 }
    expect(subgroupQuality('wracc')(k)).toBeCloseTo(15 / 100 - (20 * 30) / 10000, 12)
    expect(subgroupQuality('klosgen')(k)).toBeCloseTo(Math.sqrt(0.2) * (0.75 - 0.3), 12)
    expect(subgroupQuality('relative-accuracy')(k)).toBeCloseTo(0.45, 12)
    expect(subgroupQuality('lift')(k)).toBeCloseTo(2.5, 12)
  })

  it('the chi-squared statistic is zero at independence and signed by the direction', () => {
    expect(subgroupQuality('chi-squared')({ size: 20, positives: 6, total: 100, totalPositives: 30 })).toBeCloseTo(
      0,
      12,
    )
    const up = subgroupQuality('chi-squared')({ size: 20, positives: 15, total: 100, totalPositives: 30 })
    const down = subgroupQuality('chi-squared')({ size: 20, positives: 1, total: 100, totalPositives: 30 })
    expect(up).toBeGreaterThan(0)
    expect(down).toBeLessThan(0)
    // Pearson's statistic on the 2x2 table [[15, 5], [15, 65]]: N (ad - bc)^2 / (r1 r2 c1 c2)
    expect(up).toBeCloseTo((100 * (15 * 65 - 5 * 15) ** 2) / (20 * 80 * 30 * 70), 10)
  })

  it('counts weight rows', () => {
    expect(subgroupCounts([1, 1, 0], [1, 0, 1], [2, 1, 1])).toEqual({
      size: 3,
      positives: 2,
      total: 4,
      totalPositives: 3,
    })
  })
})

describe('subgroup discovery', () => {
  const { columns, y } = rows()
  const selectors = columnSelectors(columns)

  it('finds the planted conjunction and exhaustive search agrees with a wide beam', () => {
    const exhaustive = discoverSubgroups(
      selectors,
      { kind: 'binary', positive: y },
      { k: 5, depth: 2, search: 'exhaustive' },
    )
    const beam = discoverSubgroups(selectors, { kind: 'binary', positive: y }, { k: 5, depth: 2, width: 50 })
    expect(exhaustive[0]?.description).toBe('a=x & b=p')
    expect(beam.map((s) => s.quality)).toEqual(exhaustive.map((s) => s.quality))
    expect(exhaustive[0]?.size).toBe(Math.round(120 / 6))
  })

  it('the quality equals WRAcc computed from the coverage', () => {
    const [best] = discoverSubgroups(
      selectors,
      { kind: 'binary', positive: y },
      { k: 1, depth: 2, search: 'exhaustive' },
    )
    const k = subgroupCounts(best?.coverage ?? [], y)
    expect(best?.quality).toBeCloseTo(subgroupQuality('wracc')(k), 12)
  })

  it('a numeric target: the subgroup of a shifted mean', () => {
    const values = y.map((v, i) => 10 * v + (i % 7) * 0.01)
    const [best] = discoverSubgroups(selectors, { kind: 'numeric', values }, { k: 1, depth: 2, width: 50 })
    expect(best?.description).toBe('a=x & b=p')
    expect(best?.positives).toBeGreaterThan(9)
  })

  it('numeric columns get equal-frequency selectors', () => {
    const sel = columnSelectors({ v: [1, 2, 3, 4, 5, 6, 7, 8] }, { bins: 4 })
    expect(sel.map((s) => s.name)).toEqual(['v<=2.5', 'v>2.5', 'v<=4.5', 'v>4.5', 'v<=6.5', 'v>6.5'])
  })
})

describe('covering variants', () => {
  // Two disjoint planted subgroups: a=x and b=q explain the positives separately.
  const a: string[] = []
  const b: string[] = []
  const c: string[] = []
  const y: number[] = []
  for (let i = 0; i < 200; i++) {
    a.push(i % 4 === 0 ? 'x' : 'n')
    b.push(i % 4 === 1 ? 'q' : 'n')
    c.push(['u', 'v'][i % 2] as string)
    y.push(a[i] === 'x' || b[i] === 'q' ? 1 : 0)
  }
  const selectors = columnSelectors({ a, b, c })

  it('CN2-SD returns a diverse pair: the second rule is the other subgroup', () => {
    const rules = cn2Sd(selectors, y, { rules: 2, depth: 1 })
    expect(rules.map((r) => r.description).sort()).toEqual(['a=x', 'b=q'])
    // The first covers more positives per unit weight; after reweighting the second still has positive quality.
    expect(rules[1]?.quality).toBeGreaterThan(0)
  })

  it('Apriori-SD finds the same two subgroups', () => {
    const rules = aprioriSd(selectors, y, { rules: 2, depth: 2, minSupport: 0.2 })
    expect(rules.map((r) => r.description).sort()).toEqual(['a=x', 'b=q'])
  })

  it('Apriori-SD generates candidates level by level and then picks one rule per step', () => {
    const { state, steps } = runAlgorithm(aprioriSdAlgorithm(selectors, y, { rules: 2, depth: 2, minSupport: 0.2 }), {
      maxSteps: 50,
      seed: 0,
    })
    expect(steps).toBe(2 + 2)
    expect(state.rules.length).toBe(2)
    expect(state.candidates.length).toBeGreaterThan(2)
  })

  it('stops when no rule has positive quality', () => {
    const rules = cn2Sd(
      selectors,
      y.map(() => 1),
      { rules: 5 },
    )
    expect(rules).toEqual([])
  })
})
