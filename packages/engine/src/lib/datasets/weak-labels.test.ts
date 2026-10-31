import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { clusterAssumption, crowdLabels, multipleInstanceBags } from './weak-labels.ts'

describe('multipleInstanceBags', () => {
  it('has a witness in every positive bag and none in a negative one, with the bag label on each instance', () => {
    const { data, truth } = generate(multipleInstanceBags, {
      key: key(1),
      knobs: { bags: 80, bagSize: 6, witnessRate: 0.2 },
    })
    const instance = toFlat(truth.latents?.['instanceLabels'] as Tensor)
    const bagLabel = (truth.extra ?? {})['bagLabels'] as number[]
    const y = toFlat(data.y)
    for (let b = 0; b < 80; b++) {
      const witnesses = instance.slice(b * 6, b * 6 + 6).reduce((a, v) => a + v, 0)
      expect(witnesses > 0).toBe(bagLabel[b] === 1)
      for (let i = 0; i < 6; i++) expect(y[b * 6 + i]).toBe(bagLabel[b])
    }
    expect(bagLabel.some((v) => v === 1) && bagLabel.some((v) => v === 0)).toBe(true)
  })

  it('separates witnesses from other instances by the stated distance on the first feature direction', () => {
    const { data, truth } = generate(multipleInstanceBags, {
      key: key(2),
      knobs: { bags: 400, bagSize: 5, features: 1, separation: 4, witnessRate: 0.5 },
    })
    const instance = toFlat(truth.latents?.['instanceLabels'] as Tensor)
    const x = toFlat(data.x)
    const mean = (flag: number): number => {
      const v = x.filter((_, i) => instance[i] === flag)
      return v.reduce((a, b) => a + b, 0) / v.length
    }
    expect(mean(1) - mean(0)).toBeGreaterThan(3.7)
    expect(mean(1) - mean(0)).toBeLessThan(4.3)
  })

  it('is deterministic in the key', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 50 }), (seed) => {
        const a = generate(multipleInstanceBags, { key: key(seed), knobs: { bags: 10 } })
        const b = generate(multipleInstanceBags, { key: key(seed), knobs: { bags: 10 } })
        expect(Array.from(toFlat(a.data.x))).toEqual(Array.from(toFlat(b.data.x)))
      }),
      { numRuns: 10 },
    )
  })
})

describe('clusterAssumption', () => {
  it('follows the cluster classes exactly at purity 1, and its posterior is then the Bayes rule', () => {
    const { data, truth } = generate(clusterAssumption, { key: key(3), knobs: { n: 600, purity: 1, spread: 0.4 } })
    const cluster = toFlat(truth.clusters as Tensor)
    const map = (truth.extra ?? {})['clusterClass'] as number[]
    const y = toFlat(data.y)
    for (let i = 0; i < 600; i++) expect(y[i]).toBe(map[cluster[i] ?? 0])
    const post = toFlat(truth.posterior?.(data.x) as Tensor)
    let right = 0
    for (let i = 0; i < 600; i++) if ((post[i * 2 + 1] ?? 0) > (post[i * 2] ?? 0) === (y[i] === 1)) right++
    expect(right / 600).toBeGreaterThan(0.99)
  })

  it('lowers the accuracy of the cluster rule by the planted breakage, and the posterior rows sum to one', () => {
    const { data, truth } = generate(clusterAssumption, { key: key(4), knobs: { n: 4000, purity: 0.6, classes: 2 } })
    const cluster = toFlat(truth.clusters as Tensor)
    const map = (truth.extra ?? {})['clusterClass'] as number[]
    const y = toFlat(data.y)
    const follows = y.filter((v, i) => v === map[cluster[i] ?? 0]).length / 4000
    // With probability 0.6 the class is the cluster's, otherwise uniform over two classes: 0.6 + 0.4 / 2.
    expect(follows).toBeGreaterThan(0.77)
    expect(follows).toBeLessThan(0.83)
    const post = toFlat(truth.posterior?.(data.x) as Tensor)
    for (let i = 0; i < 20; i++) expect((post[i * 2] ?? 0) + (post[i * 2 + 1] ?? 0)).toBeCloseTo(1, 12)
  })
})

describe('crowdLabels worker kinds', () => {
  it('spammers answer uniformly and adversaries are wrong more often than right', () => {
    const { data, truth } = generate(crowdLabels, {
      key: key(5),
      knobs: { items: 1500, workers: 6, perItem: 6, classes: 3, skill: 20, spammers: 0.34, adversaries: 0.17 },
    })
    const kinds = (truth.extra ?? {})['workerKinds'] as string[]
    expect(kinds.filter((k) => k === 'spammer')).toHaveLength(2)
    expect(kinds.filter((k) => k === 'adversary')).toHaveLength(1)
    const users = toFlat(data.users)
    const items = toFlat(data.items)
    const values = toFlat(data.values)
    const classes = truth.latents?.['trueLabels'] as number[]
    const rate = (kind: string): number => {
      let hit = 0
      let total = 0
      users.forEach((w, j) => {
        if (kinds[w] !== kind) return
        total++
        if (values[j] === classes[items[j] ?? 0]) hit++
      })
      return hit / total
    }
    expect(rate('honest')).toBeGreaterThan(0.8)
    expect(Math.abs(rate('spammer') - 1 / 3)).toBeLessThan(0.05)
    expect(rate('adversary')).toBeLessThan(0.2)
  })

  it('defaults reproduce honest workers only', () => {
    const { truth } = generate(crowdLabels, { key: key(6), knobs: { items: 20, workers: 5 } })
    expect(((truth.extra ?? {})['workerKinds'] as string[]).every((k) => k === 'honest')).toBe(true)
  })
})
