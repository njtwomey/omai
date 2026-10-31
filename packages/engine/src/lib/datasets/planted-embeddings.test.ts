import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { analogyCorpus, linkageTables, plantedRelations } from './index.ts'

describe('analogyCorpus', () => {
  const { data, truth } = generate(analogyCorpus, {
    key: key(1),
    knobs: { concepts: 4, attributes: 3, sentences: 500 },
  })
  const extra = truth.extra ?? {}

  it('has one target per sentence, drawn from the concept-attribute grid', () => {
    expect(data.documents?.length).toBe(500)
    const targets = new Set(extra['targets'] as string[])
    expect(targets.size).toBe(12)
    for (const d of data.documents ?? []) expect(d.split(' ').filter((w) => targets.has(w)).length).toBe(1)
  })

  it('plants parallelograms: every quadruple shares a concept and an attribute pairwise', () => {
    const concept = extra['conceptOf'] as Record<string, number>
    const attribute = extra['attributeOf'] as Record<string, number>
    const quads = extra['analogies'] as [string, string, string, string][]
    expect(quads.length).toBe(4 * 3 * 3 * 2)
    for (const [a, b, c, d] of quads) {
      expect(concept[a]).toBe(concept[b])
      expect(concept[c]).toBe(concept[d])
      expect(attribute[a]).toBe(attribute[c])
      expect(attribute[b]).toBe(attribute[d])
      expect(concept[a]).not.toBe(concept[c])
      expect(attribute[a]).not.toBe(attribute[b])
    }
  })

  it("keeps a target's concept words with its concept: contexts are the concept's own", () => {
    const concept = extra['conceptOf'] as Record<string, number>
    for (const d of data.documents?.slice(0, 100) ?? []) {
      const words = d.split(' ')
      const target = words.find((w) => w in concept) as string
      for (const w of words) if (w.startsWith('c')) expect(w.startsWith(`c${concept[target]}_`)).toBe(true)
    }
  })
})

describe('plantedRelations', () => {
  const { data, truth } = generate(plantedRelations, { key: key(2), knobs: { groups: 3, perGroup: 8, heldOut: 0.4 } })
  const extra = truth.extra ?? {}
  const all = extra['allTriples'] as [number, number, number][]
  const test = extra['heldOut'] as [number, number, number][]
  const train = Array.from(
    toFlat(data.heads),
    (h, i) => [h, toFlat(data.relations)[i] ?? 0, toFlat(data.tails)[i] ?? 0] as const,
  )
  const has = (set: readonly (readonly number[])[], h: number, r: number, t: number): boolean =>
    set.some((f) => f[0] === h && f[1] === r && f[2] === t)

  it('splits the facts into disjoint training and test sets that cover them', () => {
    expect(train.length + test.length).toBe(all.length)
    expect(test.length).toBeGreaterThan(0)
    for (const f of test) expect(has(train, f[0], f[1], f[2])).toBe(false)
  })

  it('obeys its patterns over all the facts', () => {
    const rel = (r: number): [number, number][] => all.filter((f) => f[1] === r).map((f) => [f[0], f[2]])
    for (const [a, b] of rel(0)) expect(has(all, b, 0, a)).toBe(true)
    for (const [a, b] of rel(1)) expect(has(all, b, 2, a)).toBe(true)
    for (const [a, b] of rel(2)) expect(has(all, b, 1, a)).toBe(true)
    // the cycle: one next per entity, and no fact (a, next, b) with (b, next, a) unless the group has two entities
    expect(rel(3).length).toBe(3 * 8)
    for (const [a, b] of rel(3)) expect(has(all, b, 3, a)).toBe(false)
    // the composition: parent then next
    for (const [a, c] of rel(4)) {
      const via = rel(1)
        .filter(([h]) => h === a)
        .map(([, m]) => m)
      expect(via.some((m) => has(all, m, 3, c))).toBe(true)
    }
  })

  it('holds out only facts whose partner stays in training', () => {
    for (const [h, r, t] of test) {
      if (r === 0) expect(has(train, t, 0, h)).toBe(true)
      if (r === 1) expect(has(train, t, 2, h)).toBe(true)
    }
  })
})

describe('linkageTables', () => {
  const { data, truth } = generate(linkageTables, {
    key: key(3),
    knobs: { entities: 60, overlap: 0.5, extras: 10, typoRate: 0.05 },
  })
  const extra = truth.extra ?? {}

  it('has the known matches under the same label', () => {
    const nLeft = extra['nLeft'] as number
    const matches = extra['matches'] as [number, number][]
    const labels = toFlat(data.labels as never)
    expect(nLeft).toBe(60)
    expect(data.documents?.length).toBe(60 + (extra['nRight'] as number))
    for (const [l, r] of matches) expect(labels[l]).toBe(labels[nLeft + r])
    expect(matches.length).toBeGreaterThan(15)
    expect((extra['nRight'] as number) - matches.length).toBe(10)
  })

  it('perturbs the right records but not beyond recognition', () => {
    const matches = extra['matches'] as [number, number][]
    const nLeft = extra['nLeft'] as number
    const docs = data.documents ?? []
    let same = 0
    for (const [l, r] of matches) if (docs[l] === docs[nLeft + r]) same++
    expect(same).toBeLessThan(matches.length)
    expect(same).toBeGreaterThan(matches.length / 4)
  })
})
