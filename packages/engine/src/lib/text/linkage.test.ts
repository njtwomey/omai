import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { linkageTables } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  clusterMatches,
  comparisonVectors,
  type FieldComparator,
  fellegiSunter,
  fellegiSunterAlgorithm,
  linkageScores,
  matchPosterior,
  matchWeight,
  type RecordPair,
  sortedNeighbourhoodBlocking,
  standardBlocking,
} from './linkage.ts'
import { soundex } from './phonetic.ts'

const exact = (a: string, b: string): number => (a === b ? 1 : 0)

describe('blocking', () => {
  it('pairs records sharing a key, once, in order', () => {
    const left = [
      ['a', 'x'],
      ['b', 'y'],
      ['c', 'x'],
    ]
    const right = [
      ['d', 'x'],
      ['e', 'z'],
      ['f', 'y'],
    ]
    expect(standardBlocking(left, right, (r) => r[1] ?? '')).toEqual([
      [0, 0],
      [1, 2],
      [2, 0],
    ])
    expect(standardBlocking(left, right, (r) => [r[1] ?? '', 'any'])).toHaveLength(9)
  })

  it('sorted neighbourhood pairs records close in the sort order, across tables only', () => {
    const left = [['smith'], ['jones'], ['young']]
    const right = [['smyth'], ['joans'], ['zed']]
    const pairs = sortedNeighbourhoodBlocking(left, right, (r) => r[0] ?? '', 2)
    expect(pairs).toContainEqual([1, 1])
    expect(pairs).toContainEqual([0, 0])
    expect(pairs).not.toContainEqual([0, 1])
    expect(sortedNeighbourhoodBlocking(left, right, (r) => r[0] ?? '', 6)).toHaveLength(9)
    expect(() => sortedNeighbourhoodBlocking(left, right, (r) => r[0] ?? '', 1)).toThrow()
  })
})

describe('comparison levels', () => {
  it('cuts a similarity at descending thresholds', () => {
    const fields: FieldComparator[] = [{ field: 0, compare: (a, b) => Number(b) / Number(a), thresholds: [0.9, 0.5] }]
    const w = comparisonVectors(
      [
        [0, 0],
        [0, 1],
        [0, 2],
      ],
      [['10']],
      [['9.5'], ['6'], ['2']],
      fields,
    )
    expect(w.map((x) => x[0])).toEqual([2, 1, 0])
  })
})

describe('Fellegi-Sunter on two planted tables', () => {
  const { data, truth } = generate(linkageTables, {
    key: key(3),
    knobs: { entities: 400, overlap: 0.6, extras: 80, typoRate: 0.05 },
  })
  const extra = truth.extra ?? {}
  const nLeft = extra['nLeft'] as number
  const records = (data.documents ?? []).map((d) => d.split('|'))
  const left = records.slice(0, nLeft)
  const right = records.slice(nLeft)
  const matches = extra['matches'] as RecordPair[]
  const surname = (r: readonly string[]): string => soundex((r[0] ?? '').split(' ').at(-1) ?? '')
  const pairs = standardBlocking(left, right, (r) => surname(r))
  const fields: FieldComparator[] = [
    { field: 0, thresholds: [0.95, 0.85] },
    { field: 1, compare: exact, thresholds: [1] },
    { field: 2, compare: exact, thresholds: [1] },
  ]
  const vectors = comparisonVectors(pairs, left, right, fields)
  const fit = fellegiSunter(vectors, [3, 2, 2])

  it('blocking keeps most true pairs while cutting the comparisons by an order of magnitude', () => {
    const kept = linkageScores(pairs, matches)
    expect(kept.recall).toBeGreaterThan(0.85)
    expect(pairs.length).toBeLessThan((left.length * right.length) / 8)
  })

  it('EM recovers the planted agreement rates without labels', () => {
    // City and year agree for matches 90% of the time (the generator corrupts 10% of each); by chance 1/8 and 1/60.
    const [, cityM] = fit.model.m[1] as number[]
    const [, yearM] = fit.model.m[2] as number[]
    expect(cityM).toBeGreaterThan(0.85)
    expect(cityM).toBeLessThan(0.95)
    expect(yearM).toBeGreaterThan(0.85)
    expect(yearM).toBeLessThan(0.95)
    expect((fit.model.u[1] as number[])[1]).toBeLessThan(0.3)
    expect(fit.model.lambda).toBeGreaterThan(0.02)
    expect(fit.model.lambda).toBeLessThan(0.5)
  })

  it('calling the pairs with posterior above one half finds the planted matches', () => {
    const called = pairs.filter((_, k) => matchPosterior(fit.model, vectors[k] as Int32Array) > 0.5)
    const s = linkageScores(called, matches)
    expect(s.precision).toBeGreaterThan(0.9)
    expect(s.recall).toBeGreaterThan(0.7)
    // Entities: matched records share a cluster, the unmatched stay alone.
    const entity = clusterMatches(called, left.length, right.length)
    const [l, r] = matches[0] as RecordPair
    expect(entity[l]).toBe(entity[left.length + r])
  })

  it('has a posterior that is a monotone function of the weight, and EM raises the likelihood each step', () => {
    const lo = Int32Array.of(0, 0, 0)
    const hi = Int32Array.of(2, 1, 1)
    expect(matchWeight(fit.model, hi)).toBeGreaterThan(matchWeight(fit.model, lo))
    expect(matchPosterior(fit.model, hi)).toBeGreaterThan(matchPosterior(fit.model, lo))
    const alg = fellegiSunterAlgorithm(vectors, [3, 2, 2])
    const run = runAlgorithm(alg, { seed: 0, maxSteps: 6 })
    let s = alg.init(key(0))
    let last = s.logLikelihood
    for (let i = 0; i < run.steps; i++) {
      s = alg.step(s, { t: i, key: key(0) } as never)
      expect(s.logLikelihood).toBeGreaterThanOrEqual(last - 1e-9)
      last = s.logLikelihood
    }
    const back = alg.deserialise?.(alg.serialise?.(s) as never)
    expect(back?.lambda).toBe(s.lambda)
  })
})

describe('scores', () => {
  it('counts pairwise precision and recall', () => {
    const s = linkageScores(
      [
        [0, 0],
        [1, 2],
        [3, 3],
      ],
      [
        [0, 0],
        [1, 1],
        [3, 3],
      ],
    )
    expect(s.truePositives).toBe(2)
    expect(s.precision).toBeCloseTo(2 / 3, 12)
    expect(s.recall).toBeCloseTo(2 / 3, 12)
  })
})
