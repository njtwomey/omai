import { describe, expect, it } from 'vitest'
import { iris } from '../datasets/index.ts'
import { recordsOf } from '../exposition/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { Log, messages } from '../trace/index.ts'
import { classicTreeClassifier, pessimisticErrors, pessimisticPrune } from './classic.ts'

// Quinlan's play-tennis data: outlook (sunny, overcast, rain), temperature, humidity, wind; played or not.
const tennis = {
  x: [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [2, 1, 0, 0],
    [2, 2, 1, 0],
    [2, 2, 1, 1],
    [1, 2, 1, 1],
    [0, 1, 0, 0],
    [0, 2, 1, 0],
    [2, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [2, 1, 0, 1],
  ],
  y: [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
}

describe('classic tree growers', () => {
  it('ID3 reproduces the textbook play-tennis tree: outlook, then humidity or wind', () => {
    const f = classicTreeClassifier('id3', { categorical: [0, 1, 2, 3] }).fit(tennis)
    const t = f.params
    expect(t.feature[0]).toBe(0)
    expect(t.children[0]).toHaveLength(3)
    expect(t.branches[0]).toEqual([[0], [1], [2]])
    // The overcast child is pure; sunny splits on humidity, rain on wind.
    const [sunny, overcast, rain] = t.children[0] ?? []
    expect(t.children[overcast ?? 0]).toHaveLength(0)
    expect(t.feature[sunny ?? 0]).toBe(2)
    expect(t.feature[rain ?? 0]).toBe(3)
    expect(f.score?.(tennis)).toBe(1)
    expect(Array.from(f.apply([[1, 0, 0, 0]]))).toEqual([overcast])
  })

  it('records multiway splits as branches and logs them', () => {
    const f = classicTreeClassifier('id3', { categorical: [0, 1, 2, 3] }).fit(tennis, {
      logs: { 'tree/split': Log(), 'tree/leaves': Log() },
    })
    const record = recordsOf(f.exposition, 'tree')[0]
    expect(record?.nodes[0]?.split?.branches).toHaveLength(3)
    expect(record?.nodes[0]?.candidates).toHaveLength(4)
    const splits = messages(f.run, 'tree/split')
    expect(splits.length).toBe(f.params.feature.length)
  })

  it('C4.5 splits categorical features by branches and continuous ones by threshold, and prunes', () => {
    const flowers = iris()
    const grown = classicTreeClassifier('c45', { prune: false, minSamplesLeaf: 2 }).fit(flowers)
    const pruned = classicTreeClassifier('c45', { prune: true, confidence: 0.1 }).fit(flowers)
    expect(grown.score?.(flowers)).toBeGreaterThan(0.97)
    expect(pruned.params.feature.length).toBeLessThanOrEqual(grown.params.feature.length)
    expect(pruned.score?.(flowers)).toBeGreaterThan(0.93)
    expect(grown.params.threshold[0]).not.toBe(-2)
    // Gain ratio picks the petal features first.
    expect([2, 3]).toContain(grown.params.feature[0])
    const t = classicTreeClassifier('c45', { categorical: [0, 1, 2, 3], prune: false }).fit(tennis).params
    expect(t.children[0]?.length).toBeGreaterThan(1)
  })

  it('pessimistic pruning collapses a split that does not reduce the error bound', () => {
    const tree = {
      mode: 'c45' as const,
      classes: [0, 1],
      nFeatures: 1,
      cuts: [null],
      feature: [0, -1, -1],
      threshold: [0.5, -2, -2],
      children: [[1, 2], [], []],
      branches: [[[], []], [], []],
      value: [
        [5, 5],
        [3, 2],
        [2, 3],
      ],
      impurity: [1, 0.97, 0.97],
      nSamples: [10, 5, 5],
      depth: [0, 1, 1],
    }
    expect(pessimisticPrune(tree, 0.25).children).toEqual([[]])
    const clean = {
      ...tree,
      value: [
        [5, 5],
        [5, 0],
        [0, 5],
      ],
      impurity: [1, 0, 0],
    }
    expect(pessimisticPrune(clean, 0.25).children).toEqual([[1, 2], [], []])
    expect(pessimisticErrors(14, 0, 0.25)).toBeCloseTo(14 * (1 - 0.25 ** (1 / 14)), 12)
  })

  it('CHAID merges categories with similar class distributions and splits on the significant feature', () => {
    // Three categories of feature 0: 0 and 1 behave alike, 2 differs; feature 1 is noise.
    const x = Array.from({ length: 90 }, (_, i) => [i % 3, (i * 7) % 4])
    const y = x.map(([c = 0], i) => (c === 2 ? (i % 10 === 0 ? 0 : 1) : i % 10 === 0 ? 1 : 0))
    const f = classicTreeClassifier('chaid', { categorical: [0, 1], alphaSplit: 0.05 }).fit({ x, y })
    expect(f.params.feature[0]).toBe(0)
    expect(f.params.branches[0]).toEqual([[0, 1], [2]])
    expect(f.score?.({ x, y })).toBeGreaterThan(0.85)
    const merges = messages(f.run, 'tree/merge')
    expect(Array.isArray(merges)).toBe(true)
    // With nothing significant, the root stays a leaf.
    const noise = classicTreeClassifier('chaid', { categorical: [0, 1] }).fit({
      x: x.map((_, i) => [Math.floor(i / 2) % 3, Math.floor(i / 2) % 3]),
      y: x.map((_, i) => (i % 2 === 0 ? 1 : 0)),
    })
    expect(noise.params.children[0]).toHaveLength(0)
  })

  it('runs as a session and probes the structure early, and keys do not matter', () => {
    const est = classicTreeClassifier('id3', { categorical: [0, 1, 2, 3] })
    const s = est.session(tennis, { key: key(0), logs: 'all' })
    s.advance({ steps: 2 })
    const early = s.probe(1, 'model/structure') as { readonly kind: string; readonly nodes: readonly unknown[] }
    expect(early.kind).toBe('tree')
    expect(toFlat(s.fittedAt(1).predict(tennis.x))).toHaveLength(14)
    s.advance()
    expect(s.fittedAt(s.t).params).toEqual(est.fit(tennis).params)
  })
})
